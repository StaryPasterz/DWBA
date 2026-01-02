# driver.py
"""
DWBA Excitation Cross Section Calculator
=========================================

High-level orchestration for electron-impact excitation cross section calculations
using the Distorted Wave Born Approximation (DWBA) method.

Pipeline
--------
1. Calculate bound states (initial & final) using SAE model potential
2. Construct distorting potentials (static + optional polarization)
3. Partial wave loop with adaptive L_max convergence
4. T-matrix radial integrals via multipole expansion
5. Angular coupling (direct & exchange amplitudes)
6. Cross section integration with Born top-up extrapolation

Execution Modes
---------------
- GPU Accelerated: Uses CuPy for radial integrals (if available)
- CPU Parallel: Multiprocessing for partial wave summation

Units
-----
- Internal: Hartree atomic units (Ha, a₀)
- API Input/Output: eV, cm²

Logging
-------
Uses logging_config module. Set DWBA_LOG_LEVEL=DEBUG for verbose output.
"""


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import time
import concurrent.futures
import os

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
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential,
)
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from dwba_matrix_elements import (
    radial_ME_all_L,
    radial_ME_all_L_gpu,
    HAS_CUPY,
    check_cupy_runtime,
    RadialDWBAIntegrals,
    OscillatoryMethod,
)
from sigma_total import (
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
    sigma_cm2_to_au,
    dcs_dwba
)
from dwba_coupling import (
    calculate_amplitude_contribution,
    Amplitudes
)
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

# =============================================================================
# GLOBAL OSCILLATORY CONFIGURATION
# =============================================================================
# Can be set from DW_main.py before calculations. Used by all radial_ME calls.
OSCILLATORY_CONFIG = {
    "method": "advanced",
    "CC_nodes": 5,
    "phase_increment": 1.5708,
    "min_grid_fraction": 0.1,
    "k_threshold": 0.5,
    # GPU block size: 0 = auto-tune based on VRAM, >0 = explicit size
    "gpu_block_size": 0,
    # GPU memory strategy: "auto" (check memory), "full" (force full matrix), "block" (force block-wise)
    "gpu_memory_mode": "auto",
    # Memory threshold for auto mode: fraction of free GPU memory to use
    "gpu_memory_threshold": 0.7,
    # CPU worker count: "auto" = auto-detect (min(cpu_count, 8)), int > 0 = explicit count
    "n_workers": "auto"
}

def set_oscillatory_config(config_dict: dict):
    """Set the oscillatory integral configuration globally. Only logs actual changes."""
    changes = {}
    for k, v in config_dict.items():
        if k in OSCILLATORY_CONFIG and OSCILLATORY_CONFIG[k] != v:
            changes[k] = (OSCILLATORY_CONFIG[k], v)
    
    OSCILLATORY_CONFIG.update(config_dict)
    
    if changes:
        change_str = ", ".join(f"{k}: {old}→{new}" for k, (old, new) in changes.items())
        logger.info("Config changed: %s", change_str)

def set_oscillatory_method(method: OscillatoryMethod):
    """Set only the oscillatory integral method globally (legacy support)."""
    OSCILLATORY_CONFIG["method"] = method
    logger.info("Oscillatory method set to: %s", method)


def get_worker_count() -> int:
    """
    Get number of CPU workers from config, with auto-detection.
    
    Returns
    -------
    int
        Number of workers: 
        - If n_workers is "auto" or 0: min(cpu_count, 8) for balanced performance
        - If n_workers > 0: exact value (capped at cpu_count)
    """
    n_workers_raw = OSCILLATORY_CONFIG.get("n_workers", "auto")
    cpu_count = os.cpu_count() or 4
    
    # Handle "auto" string or 0 as auto-detect
    if n_workers_raw == "auto" or n_workers_raw == 0:
        return min(cpu_count, 8)
    
    # Explicit int value
    try:
        n_workers = int(n_workers_raw)
        if n_workers > 0:
            return min(n_workers, cpu_count)
        else:
            return min(cpu_count, 8)
    except (ValueError, TypeError):
        return min(cpu_count, 8)


@dataclass(frozen=True)
class ExcitationChannelSpec:
    """
    Specification of a particular excitation channel.
    """
    l_i: int
    l_f: int
    n_index_i: int
    n_index_f: int
    N_equiv: int
    L_max_integrals: int   # For multipole expansion of 1/r12
    L_target_i: int
    L_target_f: int
    
    # New parameter for Partial Wave loop
    L_max_projectile: int = 5

@dataclass(frozen=True)
class DWBAResult:
    ok_open_channel: bool
    E_incident_eV: float
    E_excitation_eV: float
    sigma_total_au: float
    sigma_total_cm2: float


    k_i_au: float
    k_f_au: float

    # Differential Cross Section Data
    theta_deg: Optional[np.ndarray] = None
    dcs_au: Optional[np.ndarray] = None
    
    # Detailed Breakdown
    partial_waves: Dict[str, float] = None 


def _worker_partial_wave(
    l_i: int,
    E_incident_eV: float,
    E_final_eV: float,
    z_ion: float,
    U_i: DistortingPotential,
    U_f: DistortingPotential,
    grid: RadialGrid,
    V_core: np.ndarray,
    orb_i: BoundOrbital,
    orb_f: BoundOrbital,
    chan: ExcitationChannelSpec,
    theta_grid: np.ndarray,
    k_i_au: float,
    k_f_au: float
) -> Tuple[int, Dict[Tuple[int, int], Amplitudes], float]:
    """
    Worker function for a single projectile partial wave l_i.
    Must be at module level for pickling on Windows.
    Returns: (l_i, dict_of_amplitudes_for_this_li, sigma_li_contribution)
    """
    
    # Check Parity Rule Early
    Li = chan.L_target_i
    Lf = chan.L_target_f
    target_parity_change = (Li + Lf) % 2
    
    # Solve chi_i
    try:
        chi_i = solve_continuum_wave(grid, U_i, l_i, E_incident_eV, z_ion) 
    except:
        chi_i = None

    if chi_i is None:
        return l_i, {}, 0.0

    # Local storage
    local_amplitudes = {}
    # Cover full projectile-final range up to configured projectile cap
    lf_min = 0
    lf_max = max(chan.L_max_projectile, l_i + chan.L_max_integrals)
    
    for l_f in range(lf_min, lf_max + 1):
        if (l_i + l_f) % 2 != target_parity_change: continue
            
        try:
            chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
        except:
             chi_f = None
        if chi_f is None: continue

        # Integrals
        integrals = radial_ME_all_L(
            grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals,
            use_oscillatory_quadrature=True,
            oscillatory_method=OSCILLATORY_CONFIG["method"],
            CC_nodes=OSCILLATORY_CONFIG["CC_nodes"],
            phase_increment=OSCILLATORY_CONFIG["phase_increment"],
            min_grid_fraction=OSCILLATORY_CONFIG["min_grid_fraction"],
            k_threshold=OSCILLATORY_CONFIG["k_threshold"]
        )
        
        # Distribute
        for Mi in range(-Li, Li+1):
            for Mf in range(-Lf, Lf+1):
                amps = calculate_amplitude_contribution(
                    theta_grid, 
                    integrals.I_L_direct, 
                    integrals.I_L_exchange,
                    l_i, l_f, k_i_au, k_f_au,
                    Li, Lf, Mi, Mf
                )
                
                key = (Mi, Mf)
                if key not in local_amplitudes:
                    local_amplitudes[key] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )
                
                local_amplitudes[key].f_theta += amps.f_theta
                local_amplitudes[key].g_theta += amps.g_theta

    # Calculate Sigma Contribution (integrate local DCS for convergence diagnostics)
    sigma_li_total_cm2 = 0.0
    if len(local_amplitudes) > 0:
        dcs_local = np.zeros_like(theta_grid, dtype=float)
        for (Mi, Mf), amps in local_amplitudes.items():
            dcs_local += dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, k_i_au, k_f_au, Li, chan.N_equiv)
        sigma_li_total_au = integrate_dcs_over_angles(theta_grid, dcs_local)
        sigma_li_total_cm2 = sigma_au_to_cm2(sigma_li_total_au)
    
    return l_i, local_amplitudes, sigma_li_total_cm2
        
def _worker_solve_wave(
    l: int,
    E_eV: float,
    z_ion: float,
    U: DistortingPotential,
    grid: RadialGrid
) -> Tuple[int, Optional[ContinuumWave]]:
    """Worker for parallel wave solving."""
    try:
        chi = solve_continuum_wave(grid, U, l, E_eV, z_ion)
        return l, chi
    except:
        return l, None

def precompute_continuum_waves(
    L_max: int,
    E_eV: float,
    z_ion: float,
    U: DistortingPotential,
    grid: RadialGrid
) -> Dict[int, ContinuumWave]:
    """
    Pre-compute continuum waves for L=0..L_max in parallel.
    Returns dictionary mapping l -> ContinuumWave.
    """
    waves = {}
    import os
    import concurrent.futures

    max_workers = os.cpu_count()
    if max_workers is None: max_workers = 1
    
    # We only precompute up to L_max.
    # Note: solve_continuum_wave is purely CPU.
    
    tasks = []
    # Create valid inputs
    # DistortingPotential is picklable (dataclass with arrays).
    
    # Batch submission
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_solve_wave, l, E_eV, z_ion, U, grid): l for l in range(L_max + 1)}
        
        for future in concurrent.futures.as_completed(futures):
            l = futures[future]
            try:
                rl, chi = future.result()
                if chi is not None:
                    waves[rl] = chi
            except Exception as e:
                # If one fails, we just don't have it in cache.
                # Loop will try to re-solve or skip.
                pass
                
    return waves



def _pick_bound_orbital(orbs: Tuple[BoundOrbital, ...], n_index_wanted: int) -> BoundOrbital:
    for o in orbs:
        if o.n_index == n_index_wanted:
            return o
    raise ValueError(f"Requested bound state n_index={n_index_wanted} not found.")

def compute_total_excitation_cs(
    E_incident_eV: float,
    chan: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 3000,
    match_high_energy_eV: float = 1000.0,
    n_theta: int = 200,
    use_polarization_potential: bool = False,
    # Optimization Injection
    _precalc_grid: Optional[RadialGrid] = None,
    _precalc_V_core: Optional[np.ndarray] = None,
    _precalc_orb_i: Optional[BoundOrbital] = None,
    _precalc_orb_f: Optional[BoundOrbital] = None
) -> DWBAResult:
    """
    Main high-level function to compute Total Excitation Cross Section (TECS).
    
    Uses STATIC DWBA formulation as described in the article:
    - Distorting potentials: U_j = V_core + V_Hartree (no exchange in potentials)
    - Exchange treated perturbatively via T-matrix amplitude g
    
    Optionally adds polarization potential for enhanced accuracy.
    """

    t0 = time.perf_counter()
    if use_polarization_potential:
        logger.warning(
            "Excitation: polarization potential is heuristic and not part of the article DWBA."
        )
    
    # 1. Grid & Core
    if _precalc_grid is not None and _precalc_V_core is not None:
        grid = _precalc_grid
        V_core = _precalc_V_core
    else:
        grid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
        V_core = V_core_on_grid(grid, core_params)
        
    # 2. Bound States
    # Initial
    if _precalc_orb_i is not None:
        orb_i = _precalc_orb_i
    else:
        states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+1)
        orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
        
    # Final
    if _precalc_orb_f is not None:
        orb_f = _precalc_orb_f
    else:
        states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=chan.n_index_f+1)
        orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
        
    epsilon_exc = orb_f.energy_au
    dE_target_au = orb_f.energy_au - orb_i.energy_au
    dE_target_eV = dE_target_au / ev_to_au(1.0)
    E_final_eV = E_incident_eV - dE_target_eV

    k_i_au = float(k_from_E_eV(E_incident_eV))
    k_f_au = float(k_from_E_eV(E_final_eV))
    z_ion = core_params.Zc - 1.0

    # 4. Distorting Potentials
    # Article Eq. 456-463: Static potentials U_j = V_core + V_Hartree
    U_i, U_f = build_distorting_potentials(
        grid, V_core, orb_i, orb_f, 
        k_i_au=k_i_au, 
        k_f_au=k_f_au,
        use_exchange=False,  # Article uses static potentials only
        use_polarization=use_polarization_potential
    )
    
    # 5. Partial Wave Loop
    theta_grid = np.linspace(0.0, np.pi, n_theta) # Grid for angular integration
    
    # Amplitudes f_{Mf, Mi}(theta)
    # We store them in a dict keyed by (Mi, Mf)
    # For H 1s->2s: Mi=0, Mf=0 (singlet-singlet effectively for orbital part)
    # For H 1s->2p: Mi=0, Mf=-1,0,1
    
    # L_target definitions
    Li = chan.L_target_i
    Lf = chan.L_target_f
    
    # Initialize total amplitudes for each magnetic channel
    total_amplitudes: Dict[Tuple[int, int], Amplitudes] = {}
    
    for Mi in range(-Li, Li+1):
        for Mf in range(-Lf, Lf+1):
            total_amplitudes[(Mi, Mf)] = Amplitudes(
                np.zeros_like(theta_grid, dtype=complex),
                np.zeros_like(theta_grid, dtype=complex)
            )

    # Optimization: Parallel execution relies on worker solving chi_i locally.
    # Cache removed as it is not easily shareable across processes without overhead.

    # Loop over projectile l_i
    # Dynamic L_max estimation to ensure convergence at high energies.
    # Semi-classical argument: L ~ k*R. For effective range R~10-15 a.u. (n=2), 
    # and k_i up to ~6 a.u. (500 eV), we need L ~ 60-90.
    
    L_requested = chan.L_max_projectile
    
    # ==========================================================================
    # CLASSICAL TURNING POINT CRITERION
    # ==========================================================================
    # The centrifugal barrier creates a classical turning point at:
    #     r_t(L) = (L + 0.5) / k
    # 
    # For accurate asymptotic fitting, we require r_max >= C × r_t(L_max).
    # Therefore: L_max <= k × (r_max / C) - 0.5
    #
    # Using grid r_max (from function params) and safety_factor C=2.5
    # ==========================================================================
    
    # Get actual r_max from grid (passed via function params, default 200)
    r_max_actual = grid.r[-1] if grid is not None else 200.0
    
    # Compute maximum safe L based on classical turning point physics
    L_turning_point = compute_safe_L_max(k_i_au, r_max_actual, safety_factor=2.5)
    
    # Dynamic estimate for convergence (how many L needed for given k and orbital size)
    # R_eff ~ 8 a.u. typical for low excited states, buffer +5
    L_dynamic = int(k_i_au * 8.0) + 5
    
    # Use the minimum of:
    # 1. User requested (chan.L_max_projectile)
    # 2. Dynamic estimate (convergence requirement)
    # 3. Turning point limit (physical constraint from r_max)
    L_max_proj = max(L_requested, L_dynamic)  # At least fulfill request/convergence
    L_max_proj = min(L_max_proj, L_turning_point)  # But never exceed turning point
    
    # Hard cap for extreme cases (prevents runaway computation)
    L_max_proj = min(L_max_proj, 100)
    
    logger.info("Auto-L: E=%.1f eV (k=%.2f, r_max=%.0f) -> L_max_proj=%d (turning_pt=%d)", 
                E_incident_eV, k_i_au, r_max_actual, L_max_proj, L_turning_point)
    
    if L_max_proj < L_dynamic:
        logger.warning("L_max=%d limited by turning point (need r_max~%.0f for L=%d). "
                       "Consider increasing r_max or accepting fewer partial waves.",
                       L_max_proj, compute_required_r_max(k_i_au, L_dynamic), L_dynamic)

    
    # --- Execution Strategy Selection ---
    USE_GPU = False
    if HAS_CUPY:
        # Perform a runtime check (e.g. NVRTC availability)
        if check_cupy_runtime():
            USE_GPU = True
        else:
             logger.warning("CuPy detected but runtime check failed (missing NVRTC?). Fallback to CPU.")
    
    partial_waves_dict = {} # Initialize for both paths

    if USE_GPU:
        logger.info("GPU Accelerated: Summing Partial Waves l_i=0..%d on GPU (Single Process)...", L_max_proj)
        
        # --- Pre-calculate Waves (Hybrid Mode) ---
        logger.debug("Pre-calc: Solving continuum waves in parallel (CPU)...")
        # We need chi_i up to L_max_proj
        # We need chi_f up to L_max_proj + 10 (coupling range)
        
        t_pre = time.perf_counter()
        chi_i_cache = precompute_continuum_waves(L_max_proj, E_incident_eV, z_ion, U_i, grid)
        chi_f_cache = precompute_continuum_waves(L_max_proj + 15, E_final_eV, z_ion, U_f, grid)
        logger.debug("Pre-calc: Done in %.3f s. (Cached %d i-waves, %d f-waves)", time.perf_counter() - t_pre, len(chi_i_cache), len(chi_f_cache))

        # Sequential Loop, but with fast GPU integrals
        
        sigma_accumulated = 0.0
        consecutive_small_changes = 0
        convergence_threshold = 1e-5
        nonmono_count = 0  # Track non-monotonic decay for instability detection
        
        t0_sum = time.perf_counter()
        for l_i in range(L_max_proj + 1):
            if l_i % 10 == 0 and l_i > 0:
                elapsed = time.perf_counter() - t0_sum
                eta = (elapsed / l_i) * (L_max_proj - l_i) if l_i > 0 else 0
                logger.info("  Summing: l_i=%d/%d (Elapsed: %.1fs, ETA: %.1fs)", l_i, L_max_proj, elapsed, eta)
             # Logic similar to worker but sequential and utilizing GPU integrals where possible
             # To avoid code duplication, we could call a GPU-specific worker or inline here.
             # Inline is safer for accessing GPU context.
             
            # Check Parity (Early)
            target_parity_change = (Li + Lf) % 2
            
            # Solve chi_i (Use Cache)
            if l_i in chi_i_cache:
                chi_i = chi_i_cache[l_i]
            else:
                # Fallback if precalc missed it (rare)
                chi_i = solve_continuum_wave(grid, U_i, l_i, E_incident_eV, z_ion) 
            
            if chi_i is None: break
            
            lf_min = 0
            lf_max = max(L_max_proj, l_i + chan.L_max_integrals)
            
            # Local amplitude for this l_i
            li_amplitudes = {}
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    li_amplitudes[(Mi, Mf)] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )

            any_valid_lf = False
            for l_f in range(lf_min, lf_max + 1):
                if (l_i + l_f) % 2 != target_parity_change: continue
                
                # Use Cache for chi_f
                if l_f in chi_f_cache:
                    chi_f = chi_f_cache[l_f]
                else:
                    try:
                        chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
                    except:
                        chi_f = None
                        
                if chi_f is None: continue
                
                any_valid_lf = True
                
                # --- GPU INTEGRALS ---
                integrals = radial_ME_all_L_gpu(
                    grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals,
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
                
                # Distribute (CPU - fast)
                for Mi in range(-Li, Li+1):
                    for Mf in range(-Lf, Lf+1):
                        amps = calculate_amplitude_contribution(
                            theta_grid, 
                            integrals.I_L_direct, 
                            integrals.I_L_exchange,
                            l_i, l_f, k_i_au, k_f_au,
                            Li, Lf, Mi, Mf
                        )
                        tgt = li_amplitudes[(Mi, Mf)]
                        tgt.f_theta += amps.f_theta
                        tgt.g_theta += amps.g_theta

            # Accumulate into Total
            # Compute contribution of this l_i to total cross section
            sigma_li_contribution = 0.0
            if any_valid_lf:
                # Sum DCS over all magnetic sublevels for this l_i
                dcs_li_sum = np.zeros_like(theta_grid, dtype=float)
                for (Mi, Mf), amp in li_amplitudes.items():
                    dct_term = dcs_dwba(theta_grid, amp.f_theta, amp.g_theta, k_i_au, k_f_au, Li, chan.N_equiv)
                    dcs_li_sum += dct_term
                
                sigma_li_contribution = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, dcs_li_sum))
                partial_waves_dict[f"L{l_i}"] = sigma_li_contribution
            
            # === EARLY STOPPING CHECKS ===
            should_add = True
            stop_reason = None
            pending_topup = None
            
            # CHECK 1: Negligible contribution (optimization)
            # If this partial wave contributes less than 0.01% of accumulated total, stop
            if l_i > 5 and sigma_li_contribution > 0 and sigma_accumulated > 0:
                relative_contrib = sigma_li_contribution / sigma_accumulated
                if relative_contrib < 1e-6:
                    stop_reason = f"Negligible contribution: L{l_i} adds only {relative_contrib:.1e} of total"
                    should_add = True  # Still add this one, but stop after
            
            # CHECK 2: Upturn detection (numerical instability)
            # More sensitive: >1.1x after L>8 triggers stop and uses top-up from last monotonic trio
            if l_i >= 2 and stop_reason is None:
                prev_key = f"L{l_i-1}"
                curr_key = f"L{l_i}"
                if prev_key in partial_waves_dict and curr_key in partial_waves_dict:
                    prev_contrib = partial_waves_dict[prev_key]
                    curr_contrib = partial_waves_dict[curr_key]
                    
                    if curr_contrib > prev_contrib * 1.1 and prev_contrib > 1e-30 and l_i > 8:
                        nonmono_count += 1
                        if nonmono_count >= 1:  # Single increase is enough
                            logger.warning("Upturn at L=%d: %.2e > %.2e (prev). Stopping sum and using tail from last monotonic trio.", 
                                          l_i, curr_contrib, prev_contrib)
                            should_add = False
                            # Mark as excluded
                            partial_waves_dict[f"{curr_key}_excluded"] = partial_waves_dict.pop(curr_key)
                            stop_reason = "Numerical instability (upturn)"
                            # Prepare a conservative top-up from last 3 decreasing terms if available
                            try:
                                last_keys = [f"L{l_i-1}", f"L{l_i-2}", f"L{l_i-3}"]
                                vals = [partial_waves_dict[k] for k in last_keys]
                                if vals[0] < vals[1] and vals[1] < vals[2]:
                                    q = vals[0] / vals[1]
                                    q_prev = vals[1] / vals[2]
                                    if 0.0 < q < 0.8 and abs(q - q_prev) < 0.2:
                                        tail = vals[0] * q / (1.0 - q)
                                        pending_topup = (tail, l_i-1, q)
                                        logger.debug("Prepared tail from monotonic trio: tail=%.2e, q=%.3f (L>%d)", tail, q, l_i-1)
                            except Exception:
                                pending_topup = None
                    else:
                        nonmono_count = 0  # Reset on good decay
            
            # === ACCUMULATE OR STOP ===
            if should_add:
                for k_amp, v_amp in li_amplitudes.items():
                    if k_amp not in total_amplitudes:
                         total_amplitudes[k_amp] = Amplitudes(np.zeros_like(theta_grid, dtype=complex), np.zeros_like(theta_grid, dtype=complex))
                    
                    total_amplitudes[k_amp].f_theta += v_amp.f_theta
                    total_amplitudes[k_amp].g_theta += v_amp.g_theta
            
            if stop_reason:
                logger.info("Stopping partial wave sum at L=%d: %s", l_i, stop_reason)
                if pending_topup:
                    tail, l_ref, q = pending_topup
                    partial_waves_dict["born_topup"] = tail
                    logger.info("Applied conservative top-up tail=%.2e from L>%d (q=%.3f)", tail, l_ref, q)
                break
                
            # Compute Total CS snapshot
            snap_dcs = np.zeros_like(theta_grid, dtype=float)
            for (Mi, Mf), amps in total_amplitudes.items():
                chan_dcs = dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, k_i_au, k_f_au, Li, chan.N_equiv)
                snap_dcs += chan_dcs
            snap_sigma = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, snap_dcs))
            
            delta_sigma = abs(snap_sigma - sigma_accumulated)
            rel_change = delta_sigma / (snap_sigma + 1e-30)
            sigma_accumulated = snap_sigma
            
            # Progress log
            if l_i % 5 == 0:
                 logger.debug("l_i=%d done. Sigma=%.3e (dL/L=%.1e)", l_i, snap_sigma, rel_change)

            # CHECK 3: Convergence by relative change (might be implemented as a fallback)
            #if l_i > 10:
            #    if rel_change < convergence_threshold:
            #        consecutive_small_changes += 1
            #    else:
            #        consecutive_small_changes = 0
            #else:
            #    consecutive_small_changes = 0
            
            if consecutive_small_changes >= 4:
                logger.info("Convergence: Auto-stop at l_i=%d (Change < %.0e for 4 steps)", l_i, convergence_threshold)
                break

    else:
        # Fallback to CPU Parallel
        logger.info("CPU Parallel: Summing Partial Waves l_i (Auto-Convergence, Max=%d)...", L_max_proj)
        logger.debug("Note: Parallel execution makes sequential convergence check harder. Using batching.")
        
        # Strategy: Submit batches of 10 l_i. Check after each batch.
        import multiprocessing
        # Use configured worker count (auto-detected or explicit)
        max_workers = get_worker_count()
        logger.debug("Using %d CPU workers", max_workers)
        
        sigma_accumulated = 0.0
        consecutive_small_changes = 0
        convergence_threshold = 1e-5
        
        # Batch size proportional to workers
        batch_size = max(max_workers * 2, 10)
        
        current_l = 0
        pool_running = True
        
        try:
            with multiprocessing.Pool(processes=max_workers) as pool:
                while current_l <= L_max_proj and pool_running:
                    l_end = min(current_l + batch_size, L_max_proj + 1)
                    if l_end <= current_l: break
                    
                    batch_tasks = []
                    for l_i in range(current_l, l_end):
                         batch_tasks.append((
                            l_i, E_incident_eV, E_final_eV, z_ion, U_i, U_f,
                            grid, V_core, orb_i, orb_f, chan, theta_grid, k_i_au, k_f_au
                        ))
                    
                    results = pool.starmap(_worker_partial_wave, batch_tasks)
                    
                    # Process batch
                    for l_done, partial_amps, sigma_li_contrib in results:
                        if sigma_li_contrib > 0:
                            partial_waves_dict[f"L{l_done}"] = sigma_li_contrib

                        # Accumulate amplitudes
                        if partial_amps is not None:
                            for key, part_amp in partial_amps.items():
                                if key not in total_amplitudes:
                                    total_amplitudes[key] = Amplitudes(
                                        np.zeros_like(theta_grid, dtype=complex),
                                        np.zeros_like(theta_grid, dtype=complex)
                                    )
                                total_amplitudes[key].f_theta += part_amp.f_theta
                                total_amplitudes[key].g_theta += part_amp.g_theta
                    
                    # Check Convergence after batch
                    snap_dcs = np.zeros_like(theta_grid, dtype=float)
                    for (Mi, Mf), amps in total_amplitudes.items():
                        chan_dcs = dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, k_i_au, k_f_au, Li, chan.N_equiv)
                        snap_dcs += chan_dcs
                    snap_sigma = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, snap_dcs))
                    
                    delta_sigma = abs(snap_sigma - sigma_accumulated)
                    rel_change = delta_sigma / (snap_sigma + 1e-30)
                    sigma_accumulated = snap_sigma
                    
                    logger.debug("Batch %d-%d done. Sigma=%.3e (rel_change=%.1e)", current_l, l_end-1, snap_sigma, rel_change)

                    if snap_sigma > 1e-40 and rel_change < convergence_threshold:
                         # Need to be sure it's not jsut a lucky flat spot.
                         # Since batch is ~10-20 waves, if change over 20 waves is tiny, we are done.
                         logger.info("Convergence: Auto-stop at batch end l=%d", l_end-1)
                         pool_running = False
                         
                    current_l = l_end

        except KeyboardInterrupt:
            logger.warning("User Interrupt: Terminating worker processes...")
            pool.terminate()
            pool.join()
            raise
        except Exception as e:
             logger.error("Pool Execution Failed: %s", e)
             raise

    logger.info("Summation complete in %.3f s", time.perf_counter() - t0)


    # 6. Calc Cross Sections
    # Sum over M_f, Average over M_i (unpolarized target)
    # Average factor = 1 / (2Li + 1)
    
    total_dcs = np.zeros_like(theta_grid, dtype=float)
    
    # The normalization follows Eq. 216 via dcs_dwba, which applies:
    # (2pi)^4, k_f/k_i, and N_equiv/(2L_i+1).
    # calculate_amplitude_contribution already includes the (2/pi)*(1/ki*kf)
    # factors from the continuum normalization.
    
    spin_singlet_weight = 1.0/4.0
    spin_triplet_weight = 3.0/4.0
    
    for (Mi, Mf), amps in total_amplitudes.items():
        f = amps.f_theta
        g = amps.g_theta
        
        # Calculate DCS for this specific magnetic channel
        # dcs_dwba includes the (2pi)^4 factor and kinematic/statistical factors.
        # It applies (N_equiv / (2Li+1)) to the result.
        # Summing these yields the correct total averaged DCS.
        chan_dcs = dcs_dwba(
             theta_grid, f, g, 
             k_i_au, k_f_au, 
             Li, chan.N_equiv
        )
        total_dcs += chan_dcs
        
    # Helpers for integration
    # NO extra factors needed (they are in dcs_dwba)
    
    # Integrate for TCS
    sigma_total_au = integrate_dcs_over_angles(theta_grid, total_dcs)
    sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)
    sigma_total_au_base = sigma_total_au
    sigma_total_cm2_base = sigma_total_cm2
    
        # 7. Calibration
    # Moved to calibration.py and driver integration in main
    
    # --- Born Top-Up (Excitation) ---
    # Extrapolate higher partial waves using geometric series if applicable
    
    born_topup_added = False
    if partial_waves_dict:
        try:
            # Extract L indices that were computed (skip _excluded keys)
            l_indices = sorted([int(k[1:]) for k in partial_waves_dict.keys() 
                               if k.startswith("L") and not k.endswith("_excluded") and k[1:].isdigit()])
            
            # Search backwards to find a truly monotonically decaying segment
            if len(l_indices) >= 3:
                for i in range(len(l_indices) - 1, 1, -1):  # Start from end, go backwards
                    L_try = l_indices[i]
                    L_m1 = l_indices[i-1]
                    L_m2 = l_indices[i-2]
                    
                    val_L   = partial_waves_dict.get(f"L{L_try}", 0)
                    val_Lm1 = partial_waves_dict.get(f"L{L_m1}", 0)
                    val_Lm2 = partial_waves_dict.get(f"L{L_m2}", 0)
                    
                    # Check for strict monotonic decay
                    if val_L > 0 and val_Lm1 > val_L and val_Lm2 > val_Lm1:
                        q = val_L / val_Lm1
                        q_prev = val_Lm1 / val_Lm2
                        
                        # Robustness check
                        if q < 0.95 and abs(q - q_prev) < 0.3:
                            tail_cm2 = val_L * q / (1.0 - q)
                            
                            if tail_cm2 > 1e-50:
                                sigma_total_cm2 += tail_cm2
                                sigma_total_au = sigma_cm2_to_au(sigma_total_cm2)
                                partial_waves_dict["born_topup"] = tail_cm2
                                logger.debug("Top-Up: Added %.2e cm^2 (L>%d, q=%.3f)", tail_cm2, L_try, q)
                                born_topup_added = True
                                break  # Found valid segment, stop searching
                    
        except Exception as e:
            # Non-critical enhancement
            logger.debug("Top-Up: Skipped: %s", e)

    if born_topup_added and sigma_total_cm2_base > 0.0:
        tail_frac = (sigma_total_cm2 - sigma_total_cm2_base) / sigma_total_cm2_base
        if tail_frac < 0.1:
            scale = sigma_total_cm2 / sigma_total_cm2_base
            if np.isfinite(scale) and scale > 0.0:
                total_dcs *= scale
                sigma_total_au = sigma_total_au_base * scale
        else:
            logger.warning("Top-up fraction %.2f >= 0.1; skipping DCS scaling.", tail_frac)

    return DWBAResult(
        True, E_incident_eV, dE_target_eV,
        sigma_total_au, sigma_total_cm2,
        k_i_au, k_f_au,
        theta_grid * 180.0 / np.pi,
        total_dcs,

        partial_waves_dict
    )

# --- Optimized Pre-calculation Interface ---

@dataclass
class PreparedTarget:
    grid: RadialGrid
    V_core: np.ndarray
    orb_i: BoundOrbital
    orb_f: BoundOrbital
    core_params: CorePotentialParams
    chan: ExcitationChannelSpec
    dE_target_eV: float
    # Static Configuration
    use_polarization: bool

def prepare_target(
    chan: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 3000,
    use_polarization: bool = False
) -> PreparedTarget:
    """
    Pre-computes static properties for a given transition.

    The resulting target always uses static distorting potentials
    (exchange is handled in the T-matrix, not in U_j).
    """
    
    grid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    V_core = V_core_on_grid(grid, core_params)
    
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+1)
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=chan.n_index_f+1)
    
    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
    
    dE = (orb_f.energy_au - orb_i.energy_au) / ev_to_au(1.0)
    
    # We do NOT compute U_i, U_f here to keep prepare_target lightweight.
    
    return PreparedTarget(
        grid, V_core, orb_i, orb_f, core_params, chan, dE,
        use_polarization
    )

def compute_excitation_cs_precalc(
    E_incident_eV: float,
    prep: PreparedTarget,
    n_theta: int = 200
) -> DWBAResult:
    """Efficient runner using pre-computed target data."""
    
    E_final_eV = E_incident_eV - prep.dE_target_eV
    if E_final_eV <= 0.0:
        theta_grid = np.linspace(0.0, np.pi, n_theta)
        zero_dcs = np.zeros_like(theta_grid)
        return DWBAResult(
            False,
            E_incident_eV,
            prep.dE_target_eV,
            0.0,
            0.0,
            0.0,
            0.0,
            theta_grid * 180.0 / np.pi,
            zero_dcs,
            {}
        )

    k_i_au = float(k_from_E_eV(E_incident_eV))
    k_f_au = float(k_from_E_eV(E_final_eV))
    
    # We delegate back to the main runner which now supports 
    # injected pre-calculated objects.
    # This ensures we use the full robust logic (GPU/Parallel).
    
    return compute_total_excitation_cs(
        E_incident_eV, prep.chan, prep.core_params,
        n_points=len(prep.grid.r),
        n_theta=n_theta,
        use_polarization_potential=prep.use_polarization,
        _precalc_grid=prep.grid,
        _precalc_V_core=prep.V_core,
        _precalc_orb_i=prep.orb_i,
        _precalc_orb_f=prep.orb_f
    )
