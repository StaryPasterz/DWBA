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
from collections import deque
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
    "gpu_memory_threshold": 0.8,
    # CPU worker count: "auto" = auto-detect (min(cpu_count, 8)), int > 0 = explicit count
    "n_workers": "auto",
    # v2.13+: ODE solver: "auto" (recommended), "rk45", "johnson", "numerov"
    "solver": "rk45"
}

# =============================================================================
# SCAN-LEVEL LOGGING CONTROL
# =============================================================================
# Prevents repetitive log messages for each energy point in a scan.
# Reset via reset_scan_logging() at start of new scan.
# =============================================================================
_SCAN_LOGGED = False  # True after first energy point logs hardware info


def _is_hotpath_debug_enabled() -> bool:
    """
    Enable very verbose per-(l_i,l_f) debug only on explicit request.
    """
    return os.environ.get("DWBA_HOTPATH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

def reset_scan_logging():
    """Reset scan-level logging flags. Call at start of new energy scan."""
    global _SCAN_LOGGED
    _SCAN_LOGGED = False
    logger.debug("Scan logging flags reset")

def set_oscillatory_config(config_dict: dict) -> None:
    """Set the oscillatory integral configuration globally. Only logs actual changes."""
    def normalize(v):
        # Treat 0 and "auto" as semantically identical for block size and workers
        if v == 0 or v == "0": return "auto"
        return v

    changes = {}
    for k, v in config_dict.items():
        if k in OSCILLATORY_CONFIG:
            old_norm = normalize(OSCILLATORY_CONFIG[k])
            new_norm = normalize(v)
            if old_norm != new_norm:
                changes[k] = (OSCILLATORY_CONFIG[k], v)
    
    OSCILLATORY_CONFIG.update(config_dict)
    
    if changes:
        change_str = ", ".join(f"{k}: {old}→{new}" for k, (old, new) in changes.items())
        logger.info("Config updated: %s", change_str)

def set_oscillatory_method(method: OscillatoryMethod) -> None:
    """Set only the oscillatory integral method globally (legacy support)."""
    if OSCILLATORY_CONFIG.get("method") != method:
        OSCILLATORY_CONFIG["method"] = method
        logger.info("Oscillatory method set to: %s", method)


def get_worker_count(silent: bool = False) -> int:
    """
    Get number of CPU workers from config, with auto-detection.
    
    Parameters
    ----------
    silent : bool
        If True, suppresses the info log.
    """
    n_workers_raw = OSCILLATORY_CONFIG.get("n_workers", "auto")
    cpu_count = os.cpu_count() or 4
    
    selected_count = 1
    mode_desc = ""
    
    if n_workers_raw == "auto" or n_workers_raw == 0 or n_workers_raw == "0":
        selected_count = min(cpu_count, 8)
        mode_desc = f"auto (balanced, capping {cpu_count} to 8)" if cpu_count > 8 else "auto"
    elif n_workers_raw == "max":
        selected_count = cpu_count
        mode_desc = "max (all cores)"
    else:
        try:
            val = int(n_workers_raw)
            if val > 0:
                selected_count = min(val, cpu_count)
                mode_desc = "manual" if val <= cpu_count else f"manual (capped {val} to {cpu_count})"
            else:
                selected_count = min(cpu_count, 8)
                mode_desc = "auto (fallback)"
        except (ValueError, TypeError):
            selected_count = min(cpu_count, 8)
            mode_desc = "auto (fallback)"
            
    # Log removed (v2.12+) - redundant with "Numerical Config | CPU Workers" log
    return selected_count


def log_calculation_params(
    mode: str, 
    L_max_proj: int,
    actual_gpu_mode: Optional[str] = None,
    actual_block_size: Optional[int] = None,
    E_eV: Optional[float] = None,
    k_au: Optional[float] = None,
    r_max: Optional[float] = None,
    n_points: Optional[int] = None,
    force_log: bool = False
) -> None:
    """
    Log a consistent summary of calculation parameters.
    
    Only logs full hardware info once per scan (controlled by _SCAN_LOGGED).
    Subsequent calls only log per-energy info if provided.
    
    Parameters
    ----------
    mode : str
        "GPU" or "CPU Parallel"
    L_max_proj : int
        Actual L_max for projectile partial waves.
    actual_gpu_mode : str, optional
        Actual GPU memory mode used (e.g., "full", "block").
    actual_block_size : int, optional
        Actual block size used (0 = full matrix).
    E_eV : float, optional
        Incident energy in eV (for per-energy logging).
    k_au : float, optional
        Wave number in a.u.
    r_max : float, optional
        Grid maximum radius in a.u.
    n_points : int, optional
        Number of grid points.
    force_log : bool
        If True, log hardware info even if already logged.
    """
    global _SCAN_LOGGED
    from dwba_matrix_elements import HAS_CUPY
    
    method = OSCILLATORY_CONFIG.get("method", "advanced")
    workers = get_worker_count(silent=True)
    
    logger.info("Calculation Start | Mode: %s | L_max_proj: %d", mode, L_max_proj)
    logger.info("Numerical Config  | Method: %s | CPU Workers: %d", method, workers)
    
    if HAS_CUPY and mode == "GPU":
        # Show actual vs configured GPU parameters
        gpu_mode_cfg = OSCILLATORY_CONFIG.get("gpu_memory_mode", "auto")
        block_cfg = OSCILLATORY_CONFIG.get("gpu_block_size", "auto")
        
        # Use actual values if provided, else config values
        gpu_mode_used = actual_gpu_mode or gpu_mode_cfg
        block_used = actual_block_size if actual_block_size is not None else block_cfg
        
        # Format block display
        if block_used == 0 or block_used == "auto":
            block_disp = "full-matrix"
        else:
            block_disp = f"{block_used}"
        
        logger.info("Hardware          | Platform: GPU (CuPy) | Memory: %s | Block: %s", 
                   gpu_mode_used, block_disp)
    else:
        logger.info("Hardware          | Platform: CPU (NumPy)")

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
    except Exception as e:
        logger.debug("chi_i solve failed for l_i=%d: %s", l_i, e)
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
        except Exception as e:
            logger.debug("chi_f solve failed for l_f=%d: %s", l_f, e)
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
            k_threshold=OSCILLATORY_CONFIG["k_threshold"],
            U_f_array=U_f.U_of_r  # Bug #2 fix: check both potentials
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
    grid: RadialGrid,
    phase_method: str = "hybrid",
    solver: str = "auto"
) -> Tuple[int, Optional[ContinuumWave]]:
    """Worker for parallel wave solving."""
    try:
        chi = solve_continuum_wave(
            grid, U, l, E_eV, z_ion,
            phase_extraction_method=phase_method,
            solver=solver
        )
        return l, chi
    except Exception as e:
        logger.debug("Worker wave solve failed for l=%d: %s", l, e)
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

    max_workers = get_worker_count(silent=True)  # Silent: avoid repeated log per energy
    
    # Retrieve config explicitly to pass to workers (crucial for Windows/spawn)
    phase_method = OSCILLATORY_CONFIG.get("phase_extraction", "hybrid")
    solver = OSCILLATORY_CONFIG.get("solver", "auto")
    
    # We only precompute up to L_max.
    # Note: solve_continuum_wave is purely CPU.
    
    tasks = []
    # Create valid inputs
    # DistortingPotential is picklable (dataclass with arrays).
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_l = {
            executor.submit(
                _worker_solve_wave, l, E_eV, z_ion, U, grid, 
                phase_method=phase_method, solver=solver
            ): l
            for l in range(L_max + 1)
        }
        
        # Collect results with progress logging
        completed = 0
        total = L_max + 1
        
        for future in concurrent.futures.as_completed(future_to_l):
            l = future_to_l[future]
            try:
                rl, chi = future.result()
                if chi is not None:
                    waves[rl] = chi
            except Exception as e:
                # If one fails, we just don't have it in cache.
                # Loop will try to re-solve or skip.
                pass
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.debug(f"Precomputing waves: {completed}/{total} ({(completed/total)*100:.0f}%)")
    
    # Log solver usage statistics (DEBUG level to avoid clutter)
    solver_stats = {}
    for w in waves.values():
        # Fallback for waves created before v2.12 (safety)
        method = getattr(w, 'solver_method', 'Unknown')
        solver_stats[method] = solver_stats.get(method, 0) + 1
    
    if solver_stats:
        stats_str = ", ".join(f"{s}: {n}" for s, n in solver_stats.items())
        logger.info(f"Continuum waves computed (E={E_eV:.2f} eV): {stats_str}")

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
    n_points: int = 1000,
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
    # 1. User requested (chan.L_max_projectile) - can be explicit ceiling for pilot
    # 2. Dynamic estimate (convergence requirement)  
    # 3. Turning point limit (physical constraint from r_max)
    #
    # Note: If user explicitly set a higher L_max, use that (for production).
    # If user explicitly set a lower L_max (like pilot mode), respect that as ceiling.
    if L_requested >= L_dynamic:
        # User wants MORE than dynamic estimate - trust user (production)
        L_max_proj = min(L_requested, L_turning_point)
    else:
        # User set lower explicit value (e.g., pilot speed) - respect ceiling
        # But warn if convergence might be affected
        L_max_proj = L_requested
        if L_dynamic > L_requested + 10:
            logger.debug("L_max=%d (user) may be insufficient for convergence (L_dynamic=%d). "
                        "Increase if results unstable.", L_requested, L_dynamic)
    
    # Hard cap for extreme cases (prevents runaway computation)
    L_max_proj = min(L_max_proj, 100)
    
    logger.debug("Auto-L: E=%.1f eV (k=%.2f, r_max=%.0f) -> L_max_proj=%d (turning_pt=%d)", 
                E_incident_eV, k_i_au, r_max_actual, L_max_proj, L_turning_point)
    
    if L_max_proj < L_dynamic:
        logger.debug("L_max=%d limited by turning point (need r_max~%.0f for L=%d).",
                       L_max_proj, compute_required_r_max(k_i_au, L_dynamic), L_dynamic)

    
    # --- Execution Strategy Selection ---
    USE_GPU = False
    if HAS_CUPY:
        # Perform a runtime check (e.g. NVRTC availability)
        if check_cupy_runtime():
            USE_GPU = True
        else:
             logger.warning("CuPy detected but runtime check failed (missing NVRTC?). Fallback to CPU.")
    
    log_calculation_params("GPU" if USE_GPU else "CPU Parallel", L_max_proj)

    partial_waves_dict = {} # Initialize for both paths

    if USE_GPU:
        # Sequential Loop, but with fast GPU integrals
        # logger.info("GPU Accelerated: Summing Partial Waves l_i=0..%d on GPU (Single Process)...", L_max_proj)
        
        # --- Pre-calculate Waves (Hybrid Mode) ---
        logger.debug("Pre-calc: Solving continuum waves in parallel (CPU)...")
        # We need chi_i up to L_max_proj
        # We need chi_f up to L_max_proj + 10 (coupling range)
        
        t_pre = time.perf_counter()
        chi_i_cache = precompute_continuum_waves(L_max_proj, E_incident_eV, z_ion, U_i, grid)
        l_f_precompute_max = L_max_proj + max(0, int(chan.L_max_integrals))
        chi_f_cache = precompute_continuum_waves(l_f_precompute_max, E_final_eV, z_ion, U_f, grid)
        logger.debug(
            "Pre-calc: Done in %.3f s. (Cached %d i-waves [0..%d], %d f-waves [0..%d])",
            time.perf_counter() - t_pre,
            len(chi_i_cache), L_max_proj,
            len(chi_f_cache), l_f_precompute_max
        )

        # === PHASE 3: Create GPU cache for energy-level reuse ===
        from dwba_matrix_elements import GPUCache
        max_chi = OSCILLATORY_CONFIG.get("max_chi_cached", 20)  # v2.5: configurable
        gpu_cache = GPUCache.from_grid(grid, max_chi_cached=max_chi)
        logger.debug("GPU: Created energy-level cache (max_chi_cached=%d)", max_chi)

        # Sequential Loop, but with fast GPU integrals
        
        sigma_accumulated = 0.0  # For progress logging
        dcs_history = deque(maxlen=4)   # v2.15+: Track recent (l_i, DCS_array) for per-angle stability convergence
        
        t0_sum = time.perf_counter()
        for l_i in range(L_max_proj + 1):
            if l_i % 10 == 0 and l_i > 0:
                elapsed = time.perf_counter() - t0_sum
                eta = (elapsed / l_i) * (L_max_proj - l_i) if l_i > 0 else 0
                logger.info("Summing: l_i=%d/%d (Elapsed: %.1fs, ETA: %.1fs)", l_i, L_max_proj, elapsed, eta)
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
            li_iter_start = time.perf_counter()
            last_li_heartbeat = li_iter_start
            heartbeat_every_s = float(OSCILLATORY_CONFIG.get("gpu_li_heartbeat_s", 120.0))
            slow_pair_warn_s = float(OSCILLATORY_CONFIG.get("gpu_pair_warn_s", 20.0))
            lf_parity_total = sum(
                1 for lf in range(lf_min, lf_max + 1)
                if (l_i + lf) % 2 == target_parity_change
            )
            lf_done = 0
            
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
                lf_done += 1

                now = time.perf_counter()
                if heartbeat_every_s > 0 and (now - last_li_heartbeat) >= heartbeat_every_s:
                    logger.info(
                        "Summing heartbeat: l_i=%d/%d, l_f=%d/%d (current l_f=%d, elapsed %.1fs)",
                        l_i, L_max_proj, lf_done, max(1, lf_parity_total), l_f, now - li_iter_start
                    )
                    last_li_heartbeat = now
                
                # Use Cache for chi_f
                if l_f in chi_f_cache:
                    chi_f = chi_f_cache[l_f]
                else:
                    try:
                        chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
                    except Exception as e:
                        logger.debug("GPU path chi_f solve failed for l_f=%d: %s", l_f, e)
                        chi_f = None
                        
                if chi_f is None: continue
                
                any_valid_lf = True
                
                # Very verbose hot-path debug can significantly slow long runs.
                if _is_hotpath_debug_enabled():
                    logger.debug(
                        "GPU INTEGRALS DEBUG | l_i=%d, l_f=%d | "
                        "grid.r=%d, V_core=%d, U_i=%d, U_f=%d | "
                        "orb_i.u=%d, orb_f.u=%d | "
                        "chi_i.chi=%d (idx_m=%d), chi_f.chi=%d (idx_m=%d)",
                        l_i, l_f,
                        len(grid.r), len(V_core), len(U_i.U_of_r), len(U_f.U_of_r),
                        len(orb_i.u_of_r), len(orb_f.u_of_r),
                        len(chi_i.chi_of_r), chi_i.idx_match,
                        len(chi_f.chi_of_r), chi_f.idx_match
                    )
                
                # --- GPU INTEGRALS ---
                t_pair = time.perf_counter()
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
                    gpu_memory_threshold=OSCILLATORY_CONFIG["gpu_memory_threshold"],
                    gpu_cache=gpu_cache,  # Phase 3: pass energy-level cache
                    U_f_array=U_f.U_of_r  # Bug #2 fix: check both potentials
                )
                pair_elapsed = time.perf_counter() - t_pair
                if slow_pair_warn_s > 0 and pair_elapsed >= slow_pair_warn_s:
                    logger.warning(
                        "Slow GPU pair: l_i=%d, l_f=%d took %.1fs (idx_i=%d, idx_f=%d, L_int=%d)",
                        l_i, l_f, pair_elapsed,
                        getattr(chi_i, "idx_match", -1),
                        getattr(chi_f, "idx_match", -1),
                        chan.L_max_integrals
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
            
            # === EARLY STOPPING CHECKS (v2.15 per literature) ===
            # Literature insight: Non-monotonicity of individual l_i is NORMAL in DWBA
            # due to interference in |f±g|². Correct test: stability of TOTAL DCS.
            should_add = True
            stop_reason = None

            # CHECK 0: Numerical safety on CURRENT l_i contribution.
            for (Mi, Mf), amps in li_amplitudes.items():
                if not np.all(np.isfinite(amps.f_theta)) or not np.all(np.isfinite(amps.g_theta)):
                    stop_reason = "Numerical failure: NaN/Inf in l_i amplitudes"
                    logger.error("Numerical failure at L=%d: non-finite l_i amplitudes for (Mi=%d, Mf=%d)", l_i, Mi, Mf)
                    should_add = False
                    break

            # CHECK 1: Numerical safety on accumulated state.
            if total_amplitudes and stop_reason is None:
                for (Mi, Mf), amps in total_amplitudes.items():
                    if not np.all(np.isfinite(amps.f_theta)) or not np.all(np.isfinite(amps.g_theta)):
                        stop_reason = "Numerical failure: NaN/Inf in accumulated amplitudes"
                        logger.error("Numerical failure at L=%d: non-finite accumulated amplitudes", l_i)
                        should_add = False
                        break

            # === ACCUMULATE OR STOP ===
            if should_add:
                for k_amp, v_amp in li_amplitudes.items():
                    if k_amp not in total_amplitudes:
                         total_amplitudes[k_amp] = Amplitudes(np.zeros_like(theta_grid, dtype=complex), np.zeros_like(theta_grid, dtype=complex))

                    total_amplitudes[k_amp].f_theta += v_amp.f_theta
                    total_amplitudes[k_amp].g_theta += v_amp.g_theta

            if stop_reason:
                logger.info("Stopping partial wave sum at L=%d: %s", l_i, stop_reason)
                break

            # Compute TOTAL DCS snapshot AFTER adding current l_i.
            snap_dcs = np.zeros_like(theta_grid, dtype=float)
            for (Mi, Mf), amps in total_amplitudes.items():
                chan_dcs = dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, k_i_au, k_f_au, Li, chan.N_equiv)
                snap_dcs += chan_dcs
            snap_sigma = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, snap_dcs))

            # CHECK 2: Per-angle DCS stability (post-add state only).
            # This prevents false convergence decisions based on the pre-add state.
            if l_i >= 5 and total_amplitudes:
                dcs_history.append((l_i, snap_dcs.copy()))
                if len(dcs_history) >= 4:
                    old_dcs = dcs_history[-4][1]  # DCS from 3 L's ago
                    with np.errstate(divide='ignore', invalid='ignore'):
                        angle_changes = np.abs(snap_dcs - old_dcs) / (snap_dcs + 1e-50)
                        angle_changes[~np.isfinite(angle_changes)] = 0.0
                    max_change = np.max(angle_changes)
                    avg_change = np.mean(angle_changes)
                    if max_change < 0.01 and l_i > 15:
                        stop_reason = f"Converged: max Δ(DCS)/DCS = {max_change:.2e} over all θ (stable)"
                        logger.info(
                            "Partial wave sum converged at L=%d: max angle change = %.2e, avg = %.2e",
                            l_i, max_change, avg_change
                        )
            
            delta_sigma = abs(snap_sigma - sigma_accumulated)
            rel_change = delta_sigma / (snap_sigma + 1e-30)
            sigma_accumulated = snap_sigma
            
            # Progress log
            if l_i % 5 == 0:
                 logger.debug("l_i=%d done. Sigma=%.3e (dL/L=%.1e)", l_i, snap_sigma, rel_change)

            if stop_reason:
                logger.info("Stopping partial wave sum at L=%d: %s", l_i, stop_reason)
                break

        # === END OF GPU BLOCK: Cleanup cache ===
        gpu_cache.clear()
        # Full pool cleanup between energy points (safe: runs once per energy)
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        logger.debug("GPU: Cache cleared at end of energy point")

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
    
    # --- Top-Up Section ---
    # Strategy:
    # - Forbidden transitions (|ΔL| ≠ 1): Use Born geometric extrapolation (fast convergence)
    # - Dipole-allowed (E1) transitions (|ΔL| = 1): Use Coulomb-Bethe formula (slow 1/L convergence)
    
    delta_L = abs(Lf - Li)
    is_dipole_allowed = (delta_L == 1)
    
    topup_applied = False
    topup_type = None
    topup_value = 0.0
    
    if is_dipole_allowed:
        # --- Coulomb-Bethe Top-Up for E1 Transitions ---
        # For dipole transitions, σ_l ∝ 1/L for large L (slow convergence)
        # Only apply if the last few partial waves show proper ~1/L decay
        if partial_waves_dict and len([k for k in partial_waves_dict.keys() if k.startswith("L") and k[1:].isdigit()]) >= 3:
            try:
                l_indices = sorted([int(k[1:]) for k in partial_waves_dict.keys() 
                                   if k.startswith("L") and not k.endswith("_excluded") and k[1:].isdigit()])
                
                if len(l_indices) >= 3:
                    L_max = l_indices[-1]
                    L_prev = l_indices[-2]
                    L_prev2 = l_indices[-3]
                    
                    val_L = partial_waves_dict.get(f"L{L_max}", 0)
                    val_Lm1 = partial_waves_dict.get(f"L{L_prev}", 0)
                    val_Lm2 = partial_waves_dict.get(f"L{L_prev2}", 0)
                    
                    # Check for proper convergence: values should be decreasing
                    if val_L > 0 and val_Lm1 > val_L and val_Lm2 > val_Lm1 and L_max >= 3:
                        q1 = val_L / val_Lm1  # Current decay rate
                        q2 = val_Lm1 / val_Lm2  # Previous decay rate
                        
                        # Only apply if decay is relatively stable and not too fast
                        # (if q < 0.5, convergence is already fast enough)
                        if 0.5 < q1 < 0.98 and abs(q1 - q2) < 0.2:
                            # Bethe logarithmic formula for 1/L decay
                            bethe_factor = np.log(2.0 * L_max + 1) / (L_max + 1)
                            tail_cm2 = val_L * bethe_factor * 2.0  # Factor 2 is empirical safety margin
                            
                            if tail_cm2 > 1e-50 and tail_cm2 < sigma_total_cm2_base * 0.5:
                                sigma_total_cm2 += tail_cm2
                                sigma_total_au = sigma_cm2_to_au(sigma_total_cm2)
                                partial_waves_dict["coulomb_bethe_topup"] = tail_cm2
                                topup_applied = True
                                topup_type = "Coulomb-Bethe"
                                topup_value = tail_cm2
                            
            except Exception as e:
                logger.debug("Coulomb-Bethe Top-Up: Skipped: %s", e)
    else:
        # --- Born Top-Up for Forbidden Transitions ---
        # Geometric series extrapolation (fast exponential decay expected)
        if partial_waves_dict:
            try:
                l_indices = sorted([int(k[1:]) for k in partial_waves_dict.keys() 
                                   if k.startswith("L") and not k.endswith("_excluded") and k[1:].isdigit()])
                
                if len(l_indices) >= 3:
                    for i in range(len(l_indices) - 1, 1, -1):
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
                                    topup_applied = True
                                    topup_type = "Born"
                                    topup_value = tail_cm2
                                    break
                        
            except Exception as e:
                logger.debug("Born Top-Up: Skipped: %s", e)

    # Apply scaling and log result
    if topup_applied and sigma_total_cm2_base > 0.0:
        tail_frac = (sigma_total_cm2 - sigma_total_cm2_base) / sigma_total_cm2_base
        if tail_frac < 0.2:  # More permissive for Coulomb-Bethe
            scale = sigma_total_cm2 / sigma_total_cm2_base
            if np.isfinite(scale) and scale > 0.0:
                total_dcs *= scale
                sigma_total_au = sigma_total_au_base * scale
                logger.info("Top-Up          | %s applied (tail=%.2e cm², +%.1f%%)", 
                           topup_type, topup_value, tail_frac * 100)
        else:
            logger.warning("Top-up fraction %.2f >= 0.2; skipping DCS scaling.", tail_frac)
    else:
        # Log transition type and why no top-up
        trans_type = "E1 (dipole)" if is_dipole_allowed else "Forbidden"
        logger.info("Top-Up          | Not applied (%s transition, no suitable decay)", trans_type)

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
    n_points: int = 1000,
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
    n_theta: int = 200,
    # Pilot light mode overrides (v2.5+)
    L_max_integrals_override: Optional[int] = None,
    L_max_projectile_override: Optional[int] = None,
) -> DWBAResult:
    """
    Efficient runner using pre-computed target data.
    
    Parameters
    ----------
    E_incident_eV : float
        Incident electron energy in eV.
    prep : PreparedTarget
        Pre-computed target data from prepare_target().
    n_theta : int
        Number of theta points for DCS (can be reduced for pilot).
    L_max_integrals_override : int, optional
        Override L_max for radial integrals (pilot light mode).
    L_max_projectile_override : int, optional
        Override L_max for projectile partial waves (pilot light mode).
    """
    
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
    
    # Use channel from prep, optionally with L_max overrides for pilot mode
    chan = prep.chan
    if L_max_integrals_override is not None or L_max_projectile_override is not None:
        # Create modified ChannelSpec for pilot light mode
        from dataclasses import replace
        chan = replace(
            prep.chan,
            L_max_integrals=L_max_integrals_override or prep.chan.L_max_integrals,
            L_max_projectile=L_max_projectile_override or prep.chan.L_max_projectile
        )
        logger.debug("Pilot light mode: L_max_integrals=%d, L_max_projectile=%d",
                    chan.L_max_integrals, chan.L_max_projectile)
    
    # We delegate back to the main runner which now supports 
    # injected pre-calculated objects.
    # This ensures we use the full robust logic (GPU/Parallel).
    
    return compute_total_excitation_cs(
        E_incident_eV, chan, prep.core_params,
        n_points=len(prep.grid.r),
        n_theta=n_theta,
        use_polarization_potential=prep.use_polarization,
        _precalc_grid=prep.grid,
        _precalc_V_core=prep.V_core,
        _precalc_orb_i=prep.orb_i,
        _precalc_orb_f=prep.orb_f
    )
