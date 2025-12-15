# ionization.py
#
# DWBA electron-impact ionization calculator.
#
# THEORY & IMPLEMENTATION
# -----------------------
# We treat ionization e + A -> e + A+ + e' as "excitation" to a continuum state.
# 
# 1. Kinematics:
#    E_inc + E_bound = E_scatt + E_eject
#    IP = -E_bound
#    We integrate the Single Differential Cross Section (SDCS) dσ/dE_eject
#    from E_eject = 0 to (E_inc - IP)/2.
#
# 2. Amplitudes:
#    We use the same T-matrix formalism as excitation, but the "final bound state"
#    φ_f is replaced by a distorted continuum wave χ_{l_eject}(k_e, r).
#
# 3. Partial Waves:
#    We sum over:
#    - Ejected angular momentum l_eject (0..l_eject_max)
#    - Projectile partial waves l_i, l_f (just like in driver.py)
#    - Magnetic sublevels M_i (target) and m_eject (ejected electron)
#
# 4. Normalization:
#    - Bound states are normalized to 1.
#    - Continuum waves χ(k,r) are normalized to unit asymptotic amplitude.
#    - To get cross sections per unit energy (dσ/dE), we apply the density
#      of states factor for the ejected electron:
#         ρ(E) = 1 / (π * k_eject)   [in a.u.]
#
# UNITS:
#    Input: eV.
#    Internal: Hartree Atomic Units.
#    Output: cm^2.
#

from __future__ import annotations
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from grid import (
    RadialGrid,
    make_r_grid,
    ev_to_au,
    k_from_E_eV,
)
from potential_core import (
    CorePotentialParams,
    V_core_on_grid,
)
from bound_states import (
    solve_bound_states,
    BoundOrbital,
)
# Reuse precompute logic from driver (assumed available/compatible)
from driver import precompute_continuum_waves
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential,
    U_distorting 
)
from dwba_matrix_elements import (
    radial_ME_all_L,
    radial_ME_all_L_gpu,
    HAS_CUPY,
    check_cupy_runtime,
    RadialDWBAIntegrals,
)
from dwba_coupling import (
    calculate_amplitude_contribution,
    Amplitudes
)
from sigma_total import (
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
    dcs_dwba
)
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
    
    # Projectile partial wave limit (can be set high; adaptive convergence will truncate)
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
    # Partial Waves contributions (L_projectile -> Sigma)
    partial_waves: Optional[Dict[str, float]] = None

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
    chi_i_cache: Optional[Dict[int, ContinuumWave]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Single Differential Cross Section (SDCS) at a specific ejected 
    electron energy using Partial Wave DWBA.
    
    This function implements three key optimizations:
    1. Ejected wave caching - chi_eject computed once per l_ej
    2. Scattered wave caching - chi_scatt cached within the partial wave loop
    3. Adaptive L_max - convergence detection to stop early
    
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
        
    Returns
    -------
    Tuple[float, Dict[str, float]]
        (SDCS value in a.u., partial wave contributions dict)
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
    theta_grid = np.linspace(0, np.pi, 200)
    
    Li = chan.l_i
    L_max_proj = chan.L_max_projectile
    
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
    
    # ========================================================================
    # OPTIMIZATION 2: Adaptive L_max with Convergence Detection
    # Track cumulative sigma and stop when relative change is small.
    # ========================================================================
    sigma_accumulated = 0.0
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
            lf_min = max(0, l_i_proj - 10)
            lf_max = l_i_proj + 10
            target_parity_change = (Li + Lf) % 2
            
            # Initialize amplitude accumulators for magnetic substates
            amps_shell = {}
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    amps_shell[(Mi, Mf)] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )
            
            # Cache scattered waves for this energy/L combination
            chi_scatt_cache: Dict[int, Optional[ContinuumWave]] = {}

            for l_f_proj in range(lf_min, lf_max + 1):
                if (l_i_proj + l_f_proj) % 2 != target_parity_change:
                    continue
                
                # Get or compute scattered wave
                if l_f_proj in chi_scatt_cache:
                    chi_scatt_wave = chi_scatt_cache[l_f_proj]
                else:
                    try:
                        chi_scatt_wave = solve_continuum_wave(
                            grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final
                        )
                    except Exception:
                        chi_scatt_wave = None
                    chi_scatt_cache[l_f_proj] = chi_scatt_wave
                
                if chi_scatt_wave is None:
                    continue
                
                # Compute radial integrals
                if use_gpu:
                    integrals = radial_ME_all_L_gpu(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i, bound_f=chi_eject, 
                        cont_i=chi_i, cont_f=chi_scatt_wave,
                        L_max=chan.L_max
                    )
                else:
                    integrals = radial_ME_all_L(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i, bound_f=chi_eject, 
                        cont_i=chi_i, cont_f=chi_scatt_wave,
                        L_max=chan.L_max
                    )

                # Accumulate amplitudes over magnetic substates
                for Mi in range(-Li, Li+1):
                    for Mf in range(-Lf, Lf+1):
                        partial_amp = calculate_amplitude_contribution(
                            theta_grid, 
                            integrals.I_L_direct, integrals.I_L_exchange,
                            l_i_proj, l_f_proj, k_i_au, k_scatt_au,
                            Li, Lf, Mi, Mf
                        )
                        amps_shell[(Mi, Mf)].f_theta += partial_amp.f_theta
                        amps_shell[(Mi, Mf)].g_theta += partial_amp.g_theta

            # Compute DCS for this l_i_proj and l_eject combination
            # ----------------------------------------------------------------
            # IONIZATION KINEMATICS:
            # TDCS = (k_f * k_ej / k_i) * |T|²
            # dcs_dwba computes (k_f/k_i) * |T|², so we multiply by k_ej.
            # ----------------------------------------------------------------
            dcs_shell = np.zeros_like(theta_grid, dtype=float)
            for (Mi, Mf), amp in amps_shell.items():
                chan_dcs = dcs_dwba(
                    theta_grid, amp.f_theta, amp.g_theta, 
                    k_i_au, k_scatt_au, Li, chan.N_equiv
                )
                dcs_shell += chan_dcs
            
            # Apply ionization kinematic factor
            dcs_shell *= k_eject_au
            
            sigma_shell_au = integrate_dcs_over_angles(theta_grid, dcs_shell)
            sigma_this_L += sigma_shell_au
            
        # End loop over l_ej
        
        # Record contribution from this projectile L
        sigma_energy_au += sigma_this_L
        key = f"L{l_i_proj}"
        partial_L_contribs[key] = sigma_this_L
        
        # ====================================================================
        # ADAPTIVE CONVERGENCE CHECK
        # If relative change < threshold for several consecutive L, stop.
        # ====================================================================
        if l_i_proj > 3 and sigma_accumulated > 1e-40:
            rel_change = abs(sigma_this_L) / sigma_accumulated
            if rel_change < chan.convergence_threshold:
                consecutive_small_changes += 1
                if consecutive_small_changes >= CONVERGENCE_WINDOW:
                    # Converged - stop iterating
                    break
            else:
                consecutive_small_changes = 0
        
        sigma_accumulated = sigma_energy_au
            
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
        except Exception:
            pass  # Safety fallback
            
    return sigma_energy_au, partial_L_contribs


# ============================================================================
# Helper for Parallel Execution
# ============================================================================

def _wrapper_sdcs_helper(args: tuple) -> Tuple[float, Dict[str, float]]:
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
    exchange_method: str = 'fumc',
    n_workers: Optional[int] = None
) -> IonizationResult:
    """
    Calculate Total Ionization Cross Section (TICS) via Partial Wave DWBA.
    
    This function integrates the Single Differential Cross Section (SDCS) 
    over ejected electron energies from 0 to (E - IP)/2.
    
    Optimizations:
    - Parallel energy integration (CPU workers)
    - Pre-computed incident continuum waves
    - Ejected wave caching within each energy worker
    - Adaptive L_max convergence detection
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
        Include exchange potential.
    use_polarization : bool
        Include polarization potential.
    exchange_method : str
        Exchange method ('fumc' or 'slater').
    n_workers : Optional[int]
        Number of parallel workers (default: CPU count).
        
    Returns
    -------
    IonizationResult
        Contains total cross section, SDCS data, and partial wave info.
    """
    t_start = time.perf_counter()

    # ========================================================================
    # 1. Setup Grid & Potential
    # ========================================================================
    grid = make_r_grid(r_min, r_max, n_points)
    V_core = V_core_on_grid(grid, core_params)

    # ========================================================================
    # 2. Initial Bound State
    # ========================================================================
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
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0, [], {})

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
    U_inc_obj, _ = build_distorting_potentials(
        grid, V_core, orb_i, orb_i, 
        k_i_au=k_i_au_in, 
        use_exchange=use_exchange,
        use_polarization=use_polarization,
        exchange_method=exchange_method
    )
    
    # ========================================================================
    # 5. Pre-compute Incident Waves (Once)
    # ========================================================================
    logger.info("Ionization Scan E_inc=%.1f eV: Integrating dSigma/dE...", E_incident_eV)
    
    z_ion_inc = core_params.Zc - 1.0
    logger.debug("Pre-calc: Caching incident waves up to L=%d...", chan.L_max_projectile)
    t_pre = time.perf_counter()
    chi_i_cache = precompute_continuum_waves(
        chan.L_max_projectile, E_incident_eV, 
        z_ion_inc, U_inc_obj, grid
    )
    logger.debug("Pre-calc: Done in %.3fs. Cached %d waves.", time.perf_counter() - t_pre, len(chi_i_cache))
    
    # ========================================================================
    # 6. Parallel Energy Integration
    # ========================================================================
    
    # Determine GPU availability
    USE_GPU = HAS_CUPY and check_cupy_runtime()
    
    # Prepare task arguments
    tasks = []
    for E_eject_eV in steps:
        tasks.append((
            E_eject_eV, E_incident_eV, E_total_final_eV,
            chan, core_params, grid, V_core, orb_i,
            U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
            USE_GPU,
            chi_i_cache
        ))
    
    sdcs_values_au = []
    total_steps = len(tasks)
    
    if USE_GPU:
        # GPU path: Sequential execution (GPU already parallelizes internally)
        logger.info("Mode: GPU Acceleration (Sequential Energy Scan)")
        for idx, task in enumerate(tasks):
            E_ej = task[0]
            val, partials = _compute_sdcs_at_energy(*task)
            sdcs_values_au.append(val)
            logger.debug("Step %d/%d: E_ej=%.2f eV -> SDCS=%.2e", idx+1, total_steps, E_ej, val)
    else:
        # CPU path: Parallel execution across energy points
        if n_workers is None:
            n_workers = os.cpu_count() or 1
        
        logger.info("Mode: CPU Parallel (Workers=%d)", n_workers)
        
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
                        val, partials = future.result()
                        results[idx] = val
                        completed += 1
                        logger.debug("[%d/%d] E_ej=%.2f eV -> SDCS=%.2e", completed, total_steps, steps[idx], val)
                    except Exception as e:
                        logger.error("Step %d failed: %s", idx, e)
                        results[idx] = 0.0
                
                sdcs_values_au = results
                
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
    
    elapsed = time.perf_counter() - t_start
    logger.info("Total Ionization Sigma = %.2e cm^2 (computed in %.1fs)", total_sigma_cm2, elapsed)
    
    # ========================================================================
    # 8. Prepare Output
    # ========================================================================
    sdcs_data_list = list(zip(
        steps, 
        [sigma_au_to_cm2(s * ev_to_au(1.0)) for s in sdcs_values_au]
    ))

    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au,
        total_sigma_cm2,
        sdcs_data_list,
        {}  # Full partial wave data could be aggregated if needed
    )
