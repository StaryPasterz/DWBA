# ionization.py
"""
DWBA Electron-Impact Ionization Calculator
===========================================

Calculates total and single differential ionization cross sections (SDCS)
treating ionization as excitation to continuum states.

Theory
------
e + A → e + A⁺ + e' is calculated using DWBA T-matrix formalism:
- The final "bound state" is replaced by a continuum wave χ(k_eject, r)
- SDCS is integrated over ejected energy: E_eject ∈ [0, (E_inc - IP)/2]

Kinematics
----------
    E_inc + E_bound = E_scatt + E_eject
    IP = -E_bound

Partial Waves
-------------
Summation over:
- Ejected angular momentum l_eject (0 to l_eject_max)
- Projectile partial waves l_i, l_f
- Magnetic sublevels M_i, m_eject

Normalization
-------------
- Bound states: normalized to 1
- Continuum waves: unit asymptotic amplitude

Cross Section Formula
---------------------
TDCS = (2*pi)^4 * (k_scatt * k_ej / k_i) * |T|^2

This follows the TDWBA convention with continuum waves normalized to delta(k-k').
Exchange uses swapped detection angles for indistinguishable electrons.

Units
-----
- Input: eV
- Internal: Hartree atomic units
- Output: cm²

References
----------
- S. Jones, D.H. Madison, Phys. Rev. A 67, 052701 (2003)
- D. Bote, F. Salvat, Phys. Rev. A 77, 042701 (2008)
- Llovet, Powell, Salvat, Jablonski, J. Phys. Chem. Ref. Data 43, 013102 (2014)

Logging
-------
Uses logging_config. Set DWBA_LOG_LEVEL=DEBUG for verbose output.
"""


from __future__ import annotations
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from grid import (
    RadialGrid,
    make_r_grid,
    ev_to_au,
    k_from_E_eV,
    compute_safe_L_max,
    compute_required_r_max,
)
from potential_core import (
    CorePotentialParams,
    V_core_on_grid,
)
from bound_states import (
    solve_bound_states,
    BoundOrbital,
)
# Reuse precompute logic and config from driver
from driver import (
    precompute_continuum_waves, 
    OSCILLATORY_CONFIG, 
    get_worker_count,
    log_calculation_params
)
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential
)
from dwba_matrix_elements import (
    radial_ME_all_L,
    radial_ME_all_L_gpu,
    HAS_CUPY,
    check_cupy_runtime
)
from dwba_coupling import (
    calculate_ionization_coefficients
)
from sigma_total import (
    sigma_au_to_cm2
)
from scipy.special import sph_harm
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class IonizationChannelSpec:
    """
    Parameters for the ionization channel:
      Target(n_i, l_i) -> Ion + e_eject
    """
    l_i: int       # Initial target angular momentum
    n_index_i: int # Initial target n_index
    N_equiv: int   # Number of equivalent electrons
    
    l_eject_max: int  # Max angular momentum of ejected electron to sum
    L_max: int        # Multipole max for matrix elements
    L_i_total: int    # Total L of initial target (usually l_i for H/He+)
    
    # Projectile partial wave base limit (auto-scaled from k_i as needed)
    L_max_projectile: int = 50
    
    # Convergence threshold for adaptive L_max (relative change in sigma)
    convergence_threshold: float = 0.01  # 1% relative change triggers stop

@dataclass(frozen=True)
class IonizationResult:
    E_incident_eV: float
    IP_eV: float
    sigma_total_au: float
    sigma_total_cm2: float
    # Detailed SDCS dSigma/dE
    sdcs_data: Optional[List[Tuple[float, float]]] = None # (E_eject_eV, dcs_cm2/eV)
    # Optional TDCS d^3Sigma/(dOmega_scatt dOmega_eject dE)
    # Stored as list of dicts:
    # [{"angles_deg": (theta_scatt, theta_eject, phi_eject), "values": [(E_eject_eV, tdcs_cm2_per_ev_sr2), ...]}, ...]
    tdcs_data: Optional[List[Dict[str, Any]]] = None
    # Partial waves (L_projectile -> Sigma), integrated over E_eject
    partial_waves: Optional[Dict[str, float]] = None
    # Run metadata for diagnostics/reproducibility
    metadata: Optional[Dict[str, Any]] = None


def _auto_L_max(k_i_au: float, L_requested: int, r_max: float = 200.0,
                L_cap: int = 100, L_floor: int = 3) -> int:
    """
    Auto-scale projectile L_max using classical turning point physics.

    The centrifugal barrier creates a turning point at r_t(L) = (L + 0.5) / k.
    For accurate asymptotic fitting, we require r_max >= C × r_t(L_max).
    
    Parameters
    ----------
    k_i_au : float
        Incident wave number (atomic units).
    L_requested : int
        User-requested L_max base value.
    r_max : float
        Maximum radius of computational grid.
    L_cap : int
        Hard upper limit.
    L_floor : int
        Minimum L_max to ensure s-, p-, and d-wave contributions are included.
        This prevents L_max=0 at very low energies near threshold.
        Default: 3 (includes l=0,1,2,3 partial waves).
        
    Returns
    -------
    int
        Safe L_max respecting turning point constraint and minimum floor.
    """
    # Physics-based turning point limit
    L_turning = compute_safe_L_max(k_i_au, r_max, safety_factor=2.5)
    
    # Dynamic convergence estimate
    L_dynamic = int(k_i_au * 8.0) + 5
    
    # Take max of request/dynamic, then cap by turning point and hard limit
    L_max_proj = max(L_requested, L_dynamic)
    L_max_proj = min(L_max_proj, L_turning)
    L_max_proj = min(L_max_proj, L_cap)
    
    # Ensure minimum floor for low-energy threshold behavior
    # At very low energies, k → 0, but we still need at least L_floor partial waves
    return max(L_max_proj, L_floor)


# ============================================================================
# Worker Function for Single Energy Point
# ============================================================================

def _compute_sdcs_at_energy(
    E_eject_eV: float,
    E_incident_eV: float,
    E_total_final_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    grid: RadialGrid,
    V_core: np.ndarray,
    orb_i: BoundOrbital,
    U_inc_vals: np.ndarray, 
    V_hartree_inc: np.ndarray,
    use_gpu: bool,
    chi_i_cache: Optional[Dict[int, ContinuumWave]] = None,
    tdcs_angles: Optional[List[Tuple[float, float, float]]] = None
) -> Tuple[float, Dict[str, float], Optional[List[float]]]:
    """
    Compute the Single Differential Cross Section (SDCS) at a specific ejected 
    electron energy using Partial Wave DWBA. The SDCS is angle-integrated over
    both outgoing electrons using spherical-harmonic orthonormality.
    
    This function implements three key optimizations:
    1. Ejected wave caching - chi_eject computed once per l_ej
    2. Scattered wave caching - chi_scatt cached once per energy point
    3. Adaptive L_max - convergence detection using coherent SDCS
    
    Parameters
    ----------
    E_eject_eV : float
        Ejected electron energy in eV.
    E_incident_eV : float
        Incident electron energy in eV.
    E_total_final_eV : float
        Total final state energy (E_incident - IP) in eV.
    chan : IonizationChannelSpec
        Ionization channel specification.
    core_params : CorePotentialParams
        Core potential parameters.
    grid : RadialGrid
        Radial grid for calculations.
    V_core : np.ndarray
        Core potential on grid.
    orb_i : BoundOrbital
        Initial bound orbital.
    U_inc_vals : np.ndarray
        Incident distorting potential values.
    V_hartree_inc : np.ndarray
        Hartree potential for incident channel.
    use_gpu : bool
        Whether to use GPU acceleration for integrals.
    chi_i_cache : Optional[Dict[int, ContinuumWave]]
        Pre-computed incident waves (keyed by l).
    tdcs_angles : Optional[List[Tuple[float, float, float]]]
        Optional list of (theta_scatt, theta_eject, phi_eject) in radians.
        If provided, TDCS is computed for each angle triplet.
        Exchange term uses swapped detection angles (indistinguishable electrons).
        
    Returns
    -------
    Tuple[float, Dict[str, float], Optional[List[float]]]
        (SDCS value in a.u., partial wave contributions dict, optional TDCS list)
    """
    E_scatt_eV = E_total_final_eV - E_eject_eV
    
    k_i_au = k_from_E_eV(E_incident_eV)
    k_scatt_au = k_from_E_eV(E_scatt_eV)
    k_eject_au = k_from_E_eV(E_eject_eV)
    
    z_ion_inc = core_params.Zc - 1.0
    z_ion_final = core_params.Zc
    
    # Reconstruct potential objects from arrays (needed for parallel workers)
    U_inc_obj = DistortingPotential(U_of_r=U_inc_vals, V_hartree_of_r=V_hartree_inc)
    U_ion_vals = V_core
    U_ion_obj = DistortingPotential(U_of_r=U_ion_vals, V_hartree_of_r=np.zeros_like(V_core))
    
    sigma_energy_au = 0.0
    partial_L_contribs = {}

    # Coherent SDCS sums per (l_ej, l_f, Mi, Mf). Needed to preserve
    # interference between different projectile partial waves l_i.
    sdcs_amp: List[Dict[Tuple[int, int, int], List[complex]]] = []
    for _ in range(chan.l_eject_max + 1):
        sdcs_amp.append({})

    tdcs_amp = None
    if tdcs_angles:
        tdcs_amp = []
        for _ in tdcs_angles:
            tdcs_amp.append({})
    
    Li = chan.l_i
    L_max_proj = _auto_L_max(k_i_au, chan.L_max_projectile)

    FACTOR_2PI_4 = (2.0 * np.pi) ** 4
    prefac_common = (k_scatt_au * k_eject_au / k_i_au) * (float(chan.N_equiv) / float(2 * Li + 1))
    prefac_sigma = prefac_common * FACTOR_2PI_4

    def _spin_combo(f_val: complex, g_val: complex) -> float:
        """Spin-averaged |f+g|^2 and |f-g|^2 combination."""
        amp_plus = f_val + g_val
        amp_minus = f_val - g_val
        return 0.25 * (abs(amp_plus) ** 2) + 0.75 * (abs(amp_minus) ** 2)
    
    # ========================================================================
    # OPTIMIZATION 1: Ejected Wave Cache
    # Pre-compute all ejected waves for this energy point.
    # These are reused across all projectile partial waves.
    # ========================================================================
    chi_eject_cache: Dict[int, Optional[ContinuumWave]] = {}
    for l_ej in range(chan.l_eject_max + 1):
        try:
            chi_eject_cache[l_ej] = solve_continuum_wave(
                grid, U_ion_obj, l_ej, E_eject_eV, z_ion=z_ion_final
            )
        except Exception:
            chi_eject_cache[l_ej] = None

    # Cache scattered waves once per energy (reused across all l_i and l_ej).
    chi_scatt_cache: Dict[int, Optional[ContinuumWave]] = {}
    l_f_max_scatt = L_max_proj + chan.L_max
    for l_f in range(l_f_max_scatt + 1):
        try:
            chi_scatt_cache[l_f] = solve_continuum_wave(
                grid, U_ion_obj, l_f, E_scatt_eV, z_ion=z_ion_final
            )
        except Exception:
            chi_scatt_cache[l_f] = None
    n_scatt_fail = sum(1 for v in chi_scatt_cache.values() if v is None)
    if n_scatt_fail:
        logger.debug(
            "Ionization E_ej=%.2f eV: scattered waves failed for %d/%d l_f values",
            E_eject_eV, n_scatt_fail, len(chi_scatt_cache)
        )
    
    # ========================================================================
    # OPTIMIZATION 2: Adaptive L_max with Convergence Detection
    # Track cumulative sigma and stop when relative change is small.
    # ========================================================================
    combo_sum_total = 0.0
    sigma_coherent_prev = 0.0
    consecutive_small_changes = 0
    CONVERGENCE_WINDOW = 3  # Stop after 3 consecutive small contributions
    
    for l_i_proj in range(L_max_proj + 1):
        # --- Get Incident Wave ---
        if chi_i_cache is not None and l_i_proj in chi_i_cache:
            chi_i = chi_i_cache[l_i_proj]
        else:
            try:
                chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
            except Exception:
                chi_i = None
        
        if chi_i is None:
            break  # Can't continue without incident wave
        
        sigma_this_L = 0.0  # Contribution from this L
        
        # Loop over ejected electron angular momenta
        for l_ej in range(chan.l_eject_max + 1):
            chi_eject = chi_eject_cache.get(l_ej)
            if chi_eject is None:
                continue
                
            Lf = l_ej
            
            # Compatible final projectile l_f range
            # Use a wider range to avoid truncating higher partial waves.
            lf_min = 0
            lf_max = max(L_max_proj, l_i_proj + chan.L_max)
            target_parity_change = (Li + Lf) % 2
            
            combo_sum = 0.0
            for l_f_proj in range(lf_min, lf_max + 1):
                if (l_i_proj + l_f_proj) % 2 != target_parity_change:
                    continue
                
                # Get or compute scattered wave
                chi_scatt_wave = chi_scatt_cache.get(l_f_proj)
                
                if chi_scatt_wave is None:
                    continue
                
                # Compute radial integrals
                if use_gpu:
                    integrals = radial_ME_all_L_gpu(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i, bound_f=chi_eject, 
                        cont_i=chi_i, cont_f=chi_scatt_wave,
                        L_max=chan.L_max,
                        use_oscillatory_quadrature=True,
                        oscillatory_method=OSCILLATORY_CONFIG["method"],
                        CC_nodes=OSCILLATORY_CONFIG["CC_nodes"],
                        phase_increment=OSCILLATORY_CONFIG["phase_increment"],
                        gpu_block_size=OSCILLATORY_CONFIG["gpu_block_size"],
                        min_grid_fraction=OSCILLATORY_CONFIG["min_grid_fraction"],
                        k_threshold=OSCILLATORY_CONFIG["k_threshold"],
                        gpu_memory_mode=OSCILLATORY_CONFIG["gpu_memory_mode"],
                        gpu_memory_threshold=OSCILLATORY_CONFIG["gpu_memory_threshold"]
                    )
                else:
                    integrals = radial_ME_all_L(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i, bound_f=chi_eject, 
                        cont_i=chi_i, cont_f=chi_scatt_wave,
                        L_max=chan.L_max,
                        use_oscillatory_quadrature=True,
                        oscillatory_method=OSCILLATORY_CONFIG["method"],
                        CC_nodes=OSCILLATORY_CONFIG["CC_nodes"],
                        phase_increment=OSCILLATORY_CONFIG["phase_increment"],
                        min_grid_fraction=OSCILLATORY_CONFIG["min_grid_fraction"],
                        k_threshold=OSCILLATORY_CONFIG["k_threshold"]
                    )

                # Accumulate angle-integrated coefficients over magnetic substates
                for Mi in range(-Li, Li + 1):
                    for Mf in range(-Lf, Lf + 1):
                        coeffs = calculate_ionization_coefficients(
                            integrals.I_L_direct,
                            integrals.I_L_exchange,
                            l_i_proj,
                            l_f_proj,
                            l_ej,
                            k_i_au,
                            k_scatt_au,
                            k_eject_au,
                            Li,
                            Mi,
                            Mf,
                            include_eject_norm=True  # Include ejected-wave normalization (DWBA)
                        )
                        if coeffs.f_coeff == 0.0 and coeffs.g_coeff == 0.0:
                            continue
                        # Incoherent per-l_i contribution (for diagnostics).
                        combo_sum += _spin_combo(coeffs.f_coeff, coeffs.g_coeff)

                        # Coherent SDCS sum over l_i for fixed (l_f, Mi, Mf).
                        key = (l_f_proj, Mi, Mf)
                        if key not in sdcs_amp[l_ej]:
                            sdcs_amp[l_ej][key] = [coeffs.f_coeff, coeffs.g_coeff]
                            combo_sum_total += _spin_combo(coeffs.f_coeff, coeffs.g_coeff)
                        else:
                            old_f, old_g = sdcs_amp[l_ej][key]
                            old_val = _spin_combo(old_f, old_g)
                            new_f = old_f + coeffs.f_coeff
                            new_g = old_g + coeffs.g_coeff
                            sdcs_amp[l_ej][key] = [new_f, new_g]
                            combo_sum_total += _spin_combo(new_f, new_g) - old_val

                        if tdcs_amp is not None:
                            # Angular factors for TDCS
                            mu_f_dir = Mf - Mi
                            mu_f_exc = Mi - Mf
                            for idx, (theta_scatt, theta_eject, phi_eject) in enumerate(tdcs_angles):
                                Y_eject = np.conj(sph_harm(Mf, l_ej, phi_eject, theta_eject))
                                Y_eject_swap = np.conj(sph_harm(Mf, l_ej, 0.0, theta_scatt))

                                if abs(mu_f_dir) <= l_f_proj:
                                    Y_scatt_dir = sph_harm(-mu_f_dir, l_f_proj, 0.0, theta_scatt)
                                else:
                                    Y_scatt_dir = 0.0

                                if abs(mu_f_exc) <= l_f_proj:
                                    Y_scatt_exc_swap_base = sph_harm(-mu_f_exc, l_f_proj, phi_eject, theta_eject)
                                    Y_scatt_exc_swap = ((-1.0) ** mu_f_exc) * Y_scatt_exc_swap_base
                                else:
                                    Y_scatt_exc_swap = 0.0

                                f_amp = coeffs.f_coeff * Y_scatt_dir * Y_eject
                                # Exchange uses swapped detection angles (indistinguishable electrons).
                                g_amp = coeffs.g_coeff * Y_scatt_exc_swap * Y_eject_swap

                                key = (Mi, Mf)
                                if key not in tdcs_amp[idx]:
                                    tdcs_amp[idx][key] = [0.0 + 0.0j, 0.0 + 0.0j]
                                tdcs_amp[idx][key][0] += f_amp
                                tdcs_amp[idx][key][1] += g_amp

            # Apply ionization kinematic factor
            # ----------------------------------------------------------------
            # TDCS/SDCS normalization (Jones & Madison 2003; Bote & Salvat 2008):
            # d^3σ/(dΩ_scatt dΩ_eject dE) = (2π)^4 (k_scatt k_ej / k_i) |T|^2
            # Here we use the angle-integrated form for SDCS.
            # Reference: Jones & Madison (2003), Bote & Salvat (2008)
            # ----------------------------------------------------------------
            sigma_shell_au = prefac_sigma * combo_sum
            sigma_this_L += sigma_shell_au
            
        # End loop over l_ej
        
        # Record contribution from this projectile L
        key = f"L{l_i_proj}"
        partial_L_contribs[key] = sigma_this_L
        
        # ====================================================================
        # ADAPTIVE CONVERGENCE CHECK
        # If relative change < threshold for several consecutive L, stop.
        # ====================================================================
        sigma_coherent = prefac_sigma * combo_sum_total
        if l_i_proj > 3 and sigma_coherent_prev > 1e-40:
            rel_change = abs(sigma_coherent - sigma_coherent_prev) / sigma_coherent_prev
            if rel_change < chan.convergence_threshold:
                consecutive_small_changes += 1
                if consecutive_small_changes >= CONVERGENCE_WINDOW:
                    # Converged - stop iterating
                    break
            else:
                consecutive_small_changes = 0

        sigma_coherent_prev = sigma_coherent
            
    # ========================================================================
    # Coherent SDCS sum over l_i, then apply kinematic factor.
    # ========================================================================
    sigma_energy_au = prefac_sigma * combo_sum_total

    # ========================================================================
    # BORN TOP-UP (Geometric Series Extrapolation)
    # Accounts for higher partial waves (L > L_computed) assuming geometric decay.
    # ========================================================================
    l_indices = sorted([int(k[1:]) for k in partial_L_contribs.keys() if k.startswith("L")])
    if len(l_indices) >= 3:
        try:
            L_last = l_indices[-1]
            val_L = partial_L_contribs.get(f"L{L_last}", 0.0)
            val_Lm1 = partial_L_contribs.get(f"L{L_last-1}", 0.0)
            val_Lm2 = partial_L_contribs.get(f"L{L_last-2}", 0.0)
            
            # Check for monotonic decay
            if val_L > 0 and val_Lm1 > val_L and val_Lm2 > val_Lm1:
                q = val_L / val_Lm1
                q_prev = val_Lm1 / val_Lm2
                
                # Stability check: q < 1 and consistent
                if q < 0.95 and abs(q - q_prev) < 0.2:
                    tail_correction = val_L * q / (1.0 - q)
                    if tail_correction > 0:
                        sigma_energy_au += tail_correction
                        partial_L_contribs["born_topup"] = tail_correction
                        logger.debug(
                            "Ionization SDCS top-up: added %.2e (L>%d, q=%.3f)",
                            tail_correction, L_last, q
                        )
        except Exception:
            pass  # Safety fallback
            
    tdcs_values = None
    if tdcs_amp is not None:
        tdcs_values = []
        for amp_dict in tdcs_amp:
            tdcs_sum = 0.0
            for amp_pair in amp_dict.values():
                f_val, g_val = amp_pair
                tdcs_sum += _spin_combo(f_val, g_val)
            tdcs_values.append(prefac_common * FACTOR_2PI_4 * tdcs_sum)

    return sigma_energy_au, partial_L_contribs, tdcs_values


# ============================================================================
# Helper for Parallel Execution
# ============================================================================

def _wrapper_sdcs_helper(args: tuple) -> Tuple[float, Dict[str, float], Optional[List[float]]]:
    """
    Wrapper function for ProcessPoolExecutor/imap.
    Unpacks the argument tuple and calls _compute_sdcs_at_energy.
    """
    return _compute_sdcs_at_energy(*args)


# ============================================================================
# Main Entry Point
# ============================================================================

def compute_ionization_cs(
    E_incident_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-4,
    r_max: float = 200.0,
    n_points: int = 3000,
    n_energy_steps: int = 10,
    use_exchange: bool = False,
    use_polarization: bool = False,
    exchange_method: str = "fumc",
    n_workers: Optional[int] = None,
    tdcs_angles_deg: Optional[List[Tuple[float, float, float]]] = None,
    # Optimization Injection
    _precalc_grid: Optional[RadialGrid] = None,
    _precalc_V_core: Optional[np.ndarray] = None,
    _precalc_orb_i: Optional[BoundOrbital] = None
) -> IonizationResult:
    """
    Calculate Total Ionization Cross Section (TICS) via Partial Wave DWBA.
    
    This function integrates the Single Differential Cross Section (SDCS) 
    over ejected electron energies from 0 to (E - IP)/2.
    
    Optimizations:
    - Parallel energy integration (CPU workers)
    - Pre-computed incident continuum waves
    - Ejected wave caching within each energy worker
    - Auto-scaled L_max with coherent convergence detection
    - Born Top-Up for high-L extrapolation
    
    Parameters
    ----------
    E_incident_eV : float
        Incident electron energy in eV.
    chan : IonizationChannelSpec
        Ionization channel specification.
    core_params : CorePotentialParams
        Core potential parameters.
    r_min : float
        Minimum radial grid point (bohr).
    r_max : float
        Maximum radial grid point (bohr).
    n_points : int
        Number of radial grid points.
    n_energy_steps : int
        Number of energy integration steps.
    use_exchange : bool
        Deprecated. Exchange potential is not used for ionization (ignored).
    use_polarization : bool
        Include polarization potential.
    exchange_method : str
        Deprecated. Retained for API compatibility (ignored).
    n_workers : Optional[int]
        Number of parallel workers (default: CPU count).
    tdcs_angles_deg : Optional[List[Tuple[float, float, float]]]
        Optional list of (theta_scatt, theta_eject, phi_eject) in degrees.
        If provided, computes TDCS for each E_eject on these angles.
    _precalc_grid : Optional[RadialGrid]
        Precomputed grid for reuse across energies.
    _precalc_V_core : Optional[np.ndarray]
        Precomputed core potential on the same grid.
    _precalc_orb_i : Optional[BoundOrbital]
        Precomputed initial bound orbital.
        
    Returns
    -------
    IonizationResult
        Contains total cross section, SDCS/TDCS data, partial waves, and metadata.
    """
    t_start = time.perf_counter()
    if use_exchange or exchange_method != "fumc":
        logger.debug(
            "Ionization: use_exchange/exchange_method ignored (static U_j only). "
            "use_exchange=%s, exchange_method=%s",
            use_exchange, exchange_method
        )
    if use_polarization:
        logger.warning(
            "Ionization: polarization potential is heuristic and not part of the article DWBA."
        )

    # ========================================================================
    # 1. Setup Grid & Potential
    # ========================================================================
    if _precalc_grid is not None and _precalc_V_core is not None:
        grid = _precalc_grid
        V_core = _precalc_V_core
    else:
        grid = make_r_grid(r_min, r_max, n_points)
        V_core = V_core_on_grid(grid, core_params)

    # ========================================================================
    # 2. Initial Bound State
    # ========================================================================
    if _precalc_orb_i is not None:
        orb_i = _precalc_orb_i
    else:
        states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+2)
        orb_i: Optional[BoundOrbital] = None
        for s in states_i:
            if s.n_index == chan.n_index_i:
                orb_i = s
                break
        if orb_i is None:
            raise ValueError(f"Initial bound state n={chan.n_index_i} l={chan.l_i} not found.")

    E_bound_au = orb_i.energy_au
    IP_au = -E_bound_au
    IP_eV = IP_au / ev_to_au(1.0)
    
    if E_incident_eV <= IP_eV:
        k_i_au_in = float(k_from_E_eV(E_incident_eV))
        L_max_proj = _auto_L_max(k_i_au_in, chan.L_max_projectile)
        metadata = {
            "model": "static+polarization" if use_polarization else "static",
            "use_polarization": use_polarization,
            "grid": {
                "r_min": float(grid.r[0]),
                "r_max": float(grid.r[-1]),
                "n_points": len(grid.r),
            },
            "numerics": {
                "n_energy_steps": n_energy_steps,
                "L_max_projectile_base": chan.L_max_projectile,
                "L_max_projectile_used": L_max_proj,
                "L_max_projectile_cap": 100,
                "L_max": chan.L_max,
                "l_eject_max": chan.l_eject_max,
                "convergence_threshold": chan.convergence_threshold,
            },
            "use_gpu": False,
        }
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0, [], None, None, metadata)

    E_total_final_eV = E_incident_eV - IP_eV
    
    # ========================================================================
    # 3. Energy Integration Grid
    # ========================================================================
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    
    # Clamp minimum energy for continuum solver stability
    SAFE_MIN_EV = 0.1
    steps = np.maximum(steps, SAFE_MIN_EV)
    steps = np.unique(steps)  # Remove duplicates
    
    if len(steps) < 2:
        # Recovery if range collapsed
        steps = np.linspace(SAFE_MIN_EV, max(SAFE_MIN_EV + 0.5, E_total_final_eV / 2.0), n_energy_steps + 1)
    
    # ========================================================================
    # 4. Build Distorting Potentials
    # ========================================================================
    k_i_au_in = float(k_from_E_eV(E_incident_eV))
    # Article Eq. 456-463: Static potentials U_j = V_core + V_Hartree
    U_inc_obj, _ = build_distorting_potentials(
        grid, V_core, orb_i, orb_i, 
        k_i_au=k_i_au_in, 
        use_exchange=False,  # Article uses static potentials only
        use_polarization=use_polarization
    )
    
    # ========================================================================
    # 5. Pre-compute Incident Waves (Once)
    # ========================================================================
    logger.info("Ionization Scan E_inc=%.1f eV: Integrating dSigma/dE...", E_incident_eV)
    
    z_ion_inc = core_params.Zc - 1.0
    L_max_proj = _auto_L_max(k_i_au_in, chan.L_max_projectile)
    logger.info("Auto-L: E=%.1f eV (k=%.2f) -> L_max_proj=%d", E_incident_eV, k_i_au_in, L_max_proj)
    if L_max_proj >= 100:
        logger.warning("Auto-L: L_max_proj hit cap=100; consider raising base L_max_projectile.")
    logger.debug("Pre-calc: Caching incident waves up to L=%d...", L_max_proj)
    t_pre = time.perf_counter()
    chi_i_cache = precompute_continuum_waves(
        L_max_proj, E_incident_eV, 
        z_ion_inc, U_inc_obj, grid
    )
    logger.debug("Pre-calc: Done in %.3fs. Cached %d waves.", time.perf_counter() - t_pre, len(chi_i_cache))
    
    # ========================================================================
    # 6. Parallel Energy Integration
    # ========================================================================
    
    # Selection logic for execution path
    USE_GPU = False
    if HAS_CUPY:
        if check_cupy_runtime():
            USE_GPU = True
        else:
            logger.warning("CuPy detected but runtime check failed. Fallback to CPU.")
    
    log_calculation_params("GPU" if USE_GPU else "CPU Parallel", L_max_proj)
            
    tdcs_angles_rad = None
    if tdcs_angles_deg:
        tdcs_angles_rad = [
            (np.deg2rad(th_s), np.deg2rad(th_e), np.deg2rad(ph_e))
            for (th_s, th_e, ph_e) in tdcs_angles_deg
        ]

    # Prepare task arguments
    tasks = []
    for E_eject_eV in steps:
        tasks.append((
            E_eject_eV, E_incident_eV, E_total_final_eV,
            chan, core_params, grid, V_core, orb_i,
            U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
            USE_GPU,
            chi_i_cache,
            tdcs_angles_rad
        ))
    
    sdcs_values_au = []
    tdcs_values_au = [] if tdcs_angles_rad else None
    partials_per_step: List[Dict[str, float]] = []
    total_steps = len(tasks)
    
    if USE_GPU:
        # GPU path: Sequential execution (GPU already parallelizes internally)
        logger.info("Mode: GPU Acceleration (Sequential Energy Scan)")
        for idx, task in enumerate(tasks):
            E_ej = task[0]
            val, partials, tdcs_vals = _compute_sdcs_at_energy(*task)
            sdcs_values_au.append(val)
            partials_per_step.append(partials)
            if tdcs_values_au is not None:
                tdcs_values_au.append(tdcs_vals)
            logger.debug("Step %d/%d: E_ej=%.2f eV -> SDCS=%.2e", idx+1, total_steps, E_ej, val)
    else:
        # CPU path: Parallel execution across energy points
        if n_workers is None:
            n_workers = get_worker_count(silent=True)  # Use global config
        
        # logger.info("Mode: CPU Parallel (Workers=%d)", n_workers)
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(_wrapper_sdcs_helper, task): idx 
                    for idx, task in enumerate(tasks)
                }
                
                # Collect results as they complete
                results = [None] * total_steps
                completed = 0
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        val, partials, tdcs_vals = future.result()
                        results[idx] = (val, tdcs_vals, partials)
                        completed += 1
                        logger.debug("[%d/%d] E_ej=%.2f eV -> SDCS=%.2e", completed, total_steps, steps[idx], val)
                    except Exception as e:
                        logger.error("Step %d failed: %s", idx, e)
                        results[idx] = (0.0, None, {})
                
                sdcs_values_au = [r[0] for r in results]
                partials_per_step = [r[2] for r in results]
                if tdcs_values_au is not None:
                    tdcs_values_au = [r[1] for r in results]
                    for idx, vals in enumerate(tdcs_values_au):
                        if vals is None:
                            tdcs_values_au[idx] = [0.0] * len(tdcs_angles_rad)
                
        except KeyboardInterrupt:
            logger.warning("Ionization: Interrupt detected! Terminating workers...")
            raise

    # ========================================================================
    # 7. Integrate SDCS to Get Total Cross Section
    # ========================================================================
    # Integration from 0 to (E-IP)/2 covers all unique final state configurations
    # for indistinguishable electrons. No factor of 2 is needed.
    
    steps_au = np.array([ev_to_au(e) for e in steps])
    total_sigma_au = np.trapz(sdcs_values_au, steps_au)
    total_sigma_cm2 = sigma_au_to_cm2(total_sigma_au)

    partial_waves_out: Optional[Dict[str, float]] = None
    if partials_per_step:
        partial_waves_out = {}
        keys = set()
        for partials in partials_per_step:
            keys.update(partials.keys())
        for key in sorted(keys):
            vals = np.array([p.get(key, 0.0) for p in partials_per_step])
            partial_waves_out[key] = sigma_au_to_cm2(np.trapz(vals, steps_au))
        if partial_waves_out:
            top = sorted(partial_waves_out.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.debug("Ionization partial waves: %s", ", ".join(f"{k}={v:.1e}" for k, v in top))
    
    elapsed = time.perf_counter() - t_start
    logger.info("Total Ionization Sigma = %.2e cm^2 (computed in %.1fs)", total_sigma_cm2, elapsed)
    
    # ========================================================================
    # 8. Prepare Output
    # ========================================================================
    sdcs_data_list = list(zip(
        steps, 
        [sigma_au_to_cm2(s * ev_to_au(1.0)) for s in sdcs_values_au]
    ))

    tdcs_data = None
    if tdcs_angles_deg and tdcs_values_au is not None:
        tdcs_data = []
        for angle_idx, angles_deg in enumerate(tdcs_angles_deg):
            values = []
            for e_idx, E_eject_eV in enumerate(steps):
                tdcs_val_au = tdcs_values_au[e_idx][angle_idx]
                tdcs_val_cm2 = sigma_au_to_cm2(tdcs_val_au * ev_to_au(1.0))
                values.append((E_eject_eV, tdcs_val_cm2))
            tdcs_data.append({
                "angles_deg": tuple(float(x) for x in angles_deg),
                "values": values
            })

    metadata = {
        "model": "static+polarization" if use_polarization else "static",
        "use_polarization": use_polarization,
        "grid": {
            "r_min": float(grid.r[0]),
            "r_max": float(grid.r[-1]),
            "n_points": len(grid.r),
        },
        "numerics": {
            "n_energy_steps": n_energy_steps,
            "L_max_projectile_base": chan.L_max_projectile,
            "L_max_projectile_used": L_max_proj,
            "L_max_projectile_cap": 100,
            "L_max": chan.L_max,
            "l_eject_max": chan.l_eject_max,
            "convergence_threshold": chan.convergence_threshold,
        },
        "use_gpu": USE_GPU,
    }

    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au,
        total_sigma_cm2,
        sdcs_data_list,
        tdcs_data,
        partial_waves_out,
        metadata
    )
