# driver.py
"""
DWBA Excitation Cross Section Calculator
=========================================

High-level orchestration for electron-impact excitation cross section calculations
using the Distorted Wave Born Approximation (DWBA) method.

Pipeline
--------
1. Calculate bound states (initial & final) using SAE model potential
2. Construct distorting potentials (static/exchange/polarization)
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
)
from sigma_total import (
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
    dcs_dwba
)
from dwba_coupling import (
    calculate_amplitude_contribution,
    Amplitudes
)
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)



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
    lf_min = max(0, l_i - 10) 
    lf_max = l_i + 10
    
    for l_f in range(lf_min, lf_max + 1):
        if (l_i + l_f) % 2 != target_parity_change: continue
            
        try:
            chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
        except:
             chi_f = None
        if chi_f is None: continue

        # Integrals
        integrals = radial_ME_all_L(
            grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals
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

    # Calculate Sigma Contribution
    sigma_li_total = 0.0
    if len(local_amplitudes) > 0:
        val, _ = dcs_dwba(theta_grid, local_amplitudes, Li, chan.N_equiv)
        sigma_li_total = val
        
    return l_i, local_amplitudes, sigma_li_total
        
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
    exchange_method: str = 'fumc',
    use_polarization_potential: bool = False,
    # Optimization Injection
    _precalc_grid: Optional[RadialGrid] = None,
    _precalc_V_core: Optional[np.ndarray] = None,
    _precalc_orb_i: Optional[BoundOrbital] = None,
    _precalc_orb_f: Optional[BoundOrbital] = None
) -> DWBAResult:
    """
    Main high-level function to compute Total Excitation Cross Section (TECS).
    Now supports pre-calculated static properties for optimization.
    """
    
    # Derive flags
    use_exchange = (exchange_method is not None and exchange_method.lower() != 'none')

    t0 = time.perf_counter()
    
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
    U_i, U_f = build_distorting_potentials(
        grid, V_core, orb_i, orb_f, 
        k_i_au=k_i_au, 
        k_f_au=k_f_au,
        use_exchange=use_exchange,
        use_polarization=use_polarization_potential,
        exchange_method=exchange_method
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
    # We restore L_max_projectile default to 25 in datastruct (via caller or explicit set)
    # Dynamic L_max estimation to ensure convergence at high energies.
    # Semi-classical argument: L ~ k*R. For effective range R~10-15 a.u. (n=2), 
    # and k_i up to ~6 a.u. (500 eV), we need L ~ 60-90.
    
    L_requested = chan.L_max_projectile
    # Heuristic: L_dynamic = k_i_au * R_eff + buffer.
    # Using R_eff = 10.0 (generous for low n) + safety 10.
    L_dynamic = int(k_i_au * 12.0) + 15
    
    # We take the maximum of requested (user spec) and dynamic (convergence safety)
    # But we also clamp it to avoid excessive runtime if E is huge, 
    # though 1000 eV requires ~80-100.
    L_max_proj = max(L_requested, L_dynamic)

    
    # Cap global max to reasonable value (e.g. 150) to prevent infinite/too long runs
    L_max_proj = min(L_max_proj, 150)
    
    # If user explicitly set a very high L in chan, respect it if < 150 (already handled by max)
    # If user set small L (e.g. 5 default), this dynamic logic overrides it properly.
    
    logger.info("Auto-L: E=%.1f eV (k=%.2f) -> L_max_proj=%d", E_incident_eV, k_i_au, L_max_proj) 
    
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
        
        for l_i in range(L_max_proj + 1):
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
            
            lf_min = max(0, l_i - 10) 
            lf_max = l_i + 10
            
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
                    grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals
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

            for k_amp, v_amp in li_amplitudes.items():
                if k_amp not in total_amplitudes:
                     total_amplitudes[k_amp] = Amplitudes(np.zeros_like(theta_grid, dtype=complex), np.zeros_like(theta_grid, dtype=complex))
                
                # Add to total
                total_amplitudes[k_amp].f_theta += v_amp.f_theta
                total_amplitudes[k_amp].g_theta += v_amp.g_theta
                
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

            # Check convergence
            if l_i > 10 and rel_change < convergence_threshold:
                consecutive_small_changes += 1
            else:
                consecutive_small_changes = 0
            
            if consecutive_small_changes >= 4:
                logger.info("Convergence: Auto-stop at l_i=%d (Change < %.0e for 4 steps)", l_i, convergence_threshold)
                break

    else:
        # Fallback to CPU Parallel
        logger.info("CPU Parallel: Summing Partial Waves l_i (Auto-Convergence, Max=%d)...", L_max_proj)
        logger.debug("Note: Parallel execution makes sequential convergence check harder. Using batching.")
        
        # Strategy: Submit batches of 10 l_i. Check after each batch.
        import multiprocessing
        max_workers = os.cpu_count()
        if max_workers is None: max_workers = 1
        
        sigma_accumulated = 0.0
        consecutive_small_changes = 0
        convergence_threshold = 1e-5
        
        # Batch size
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
    
    prefac_kinematics = (k_f_au / k_i_au)
    # N_equiv factor? Eq 216 says: N * (k_f/k_i) * 1/(2Li+1) ...
    # Our amplitudes already include geometric factors?
    # Eq 412 has "2/pi".
    # Cross section formula dsigma/dOmega = ... |f|^2 ...
    # Wait, Eq 216 puts factors EXPLICITLY.
    # But our calculate_amplitude_contribution includes (2/pi) * 1/(ki kf).
    # Check normalization. 
    # Standard: dSigma/dOmega = (kf/ki) |f_scattering|^2.
    # Our f is T-matrix-like.
    # Usually f_scattering = -(2pi)^2 ... T.
    # Let's trust Eq 412 produces f_scattering directly (it has asymptotic form).
    # Yes, Eq 403-410 shows f = ... integral.
    # So dcs = (k_f / k_i) * Sum |f|^2.
    
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
    
        # 7. Calibration
    # Moved to calibration.py and driver integration in main
    
    # --- Born Top-Up (Excitation) ---
    # Extrapolate higher partial waves using geometric series if applicable
    
    if partial_waves_dict:
        try:
            # Extract L indices that were computed
            l_indices = sorted([int(k[1:]) for k in partial_waves_dict.keys() if k.startswith("L")])
            if len(l_indices) >= 3:
                L_last = l_indices[-1]
                val_L   = partial_waves_dict[f"L{L_last}"]
                val_Lm1 = partial_waves_dict[f"L{L_last-1}"]
                val_Lm2 = partial_waves_dict[f"L{L_last-2}"]
                
                # Check for monotonic decay
                if (val_L > 0 and val_Lm1 > val_L and val_Lm2 > val_Lm1):
                    q = val_L / val_Lm1
                    q_prev = val_Lm1 / val_Lm2
                    
                    # Robustness check
                    if q < 0.95 and abs(q - q_prev) < 0.2:
                        tail_au = val_L * q / (1.0 - q)
                        
                        if tail_au > 1e-50: # Avoid noise
                            # Add to totals
                            sigma_total_au += tail_au
                            sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)
                            partial_waves_dict["born_topup"] = tail_au
                            logger.debug("Top-Up: Added %.2e cm^2 (L>%d, q=%.3f)", sigma_au_to_cm2(tail_au), L_last, q)
        except Exception as e:
            # Non-critical enhancement
            logger.debug("Top-Up: Skipped: %s", e)

    return DWBAResult(
        True, E_incident_eV, dE_target_eV,
        sigma_total_au, sigma_total_cm2,
        k_i_au, k_f_au,
        theta_grid * 180.0 / np.pi,
        total_dcs, # dcs_vals variable name was likely wrong in previous view, check context. But wait, local var was 'total_dcs'?
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
    use_exchange: bool
    use_polarization: bool
    exchange_method: str

def prepare_target(
    chan: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 3000,
    use_exchange: bool = False,
    use_polarization: bool = False,
    exchange_method: str = 'fumc'
) -> PreparedTarget:
    """Pre-computes static properties for a given transition."""
    
    grid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    V_core = V_core_on_grid(grid, core_params)
    
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+1)
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=chan.n_index_f+1)
    
    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
    
    dE = (orb_f.energy_au - orb_i.energy_au) / ev_to_au(1.0)
    
    # We do NOT compute U_i, U_f here because Exchange depends on E.
    
    return PreparedTarget(
        grid, V_core, orb_i, orb_f, core_params, chan, dE,
        use_exchange, use_polarization, exchange_method
    )

def compute_excitation_cs_precalc(
    E_incident_eV: float,
    prep: PreparedTarget,
    n_theta: int = 200
) -> DWBAResult:
    """Efficient runner using pre-computed target data."""
    
    E_final_eV = E_incident_eV - prep.dE_target_eV
    if E_final_eV <= 0.0:
        return DWBAResult(False, E_incident_eV, prep.dE_target_eV, 0.0, 0.0, 0.0, 0.0, None)

    k_i_au = float(k_from_E_eV(E_incident_eV))
    k_f_au = float(k_from_E_eV(E_final_eV))
    
    # We delegate back to the main runner which now supports 
    # injected pre-calculated objects.
    # This ensures we use the full robust logic (GPU/Parallel).
    
    return compute_total_excitation_cs(
        E_incident_eV, prep.chan, prep.core_params,
        n_points=len(prep.grid.r),
        n_theta=n_theta,
        exchange_method=prep.exchange_method,
        use_polarization_potential=prep.use_polarization,
        _precalc_grid=prep.grid,
        _precalc_V_core=prep.V_core,
        _precalc_orb_i=prep.orb_i,
        _precalc_orb_f=prep.orb_f
    )
