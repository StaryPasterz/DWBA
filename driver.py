# driver.py
#
# High-level orchestration for DWBA excitation calculations.
#
# DESCRIPTION
# -----------
# This module acts as the computation engine for electron-impact excitation cross sections.
# It coordinates the entire pipeline:
# 1. Calculation of Target Bound States (Initial & Final).
# 2. Construction of Distorted Potentials (Static-Only by default).
# 3. Partial Wave Loop:
#    - Iterate l_i (projectile in)
#    - Iterate l_f (projectile out)
#    - Iterate M_i, M_f (magnetic sublevels)
#    - Compute T-matrix radial integrals.
#    - Accumulate amplitudes f(theta), g(theta).
# 4. Cross Section Evaluation (Integration over angles).
# 5. Application of M-Tong Calibration (Empirical Formula).
#
# UNITS
# -----
# - Internal Physics: Hartree Atomic Units (energy in Ha, length in a0).
# - Input/Output API: Energy in eV, Cross Sections in cm^2 (and a0^2).
#

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
    k_i_au: float
    k_f_au: float

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
) -> Tuple[int, Dict[Tuple[int, int], Amplitudes]]:
    """
    Worker function for a single projectile partial wave l_i.
    Must be at module level for pickling on Windows.
    Returns: (l_i, dict_of_amplitudes_for_this_li)
    """
    
    # Check Parity Rule Early (Optimization)
    Li = chan.L_target_i
    Lf = chan.L_target_f
    target_parity_change = (Li + Lf) % 2
    
    # Solvers
    # Optimization: We re-solve chi_i here. 
    # In multiprocessing, sharing cache across processes is hard without Manager.
    # Re-solving is safer and likely negligible overhead vs radial integrals.
    chi_i = solve_continuum_wave(grid, U_i, l_i, E_incident_eV, z_ion) 
    if chi_i is None:
        return l_i, {}

    # Local storage for Amplitudes contributed by this l_i
    # Key: (Mi, Mf)
    local_amplitudes = {}

    lf_min = max(0, l_i - 10) 
    lf_max = l_i + 10
    
    for l_f in range(lf_min, lf_max + 1):
        # Check Parity Rule
        if (l_i + l_f) % 2 != target_parity_change:
            continue
            
        # Solve chi_f
        try:
            chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
        except:
             chi_f = None
             
        if chi_f is None:
            continue

        # Compute Radial Integrals
        integrals = radial_ME_all_L(
            grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals
        )
        
        # Distribute contribution
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
                
    return l_i, local_amplitudes

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
    n_theta: int = 200
) -> DWBAResult:
    """
    Main high-level function.
    Calculates excitation cross section in DWBA.
    Implements Partial Wave Summation over projectile l_i, l_f.
    """

    t0 = time.perf_counter()
    
    # 1. Grid
    grid: RadialGrid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    
    # 2. V_core
    V_core = V_core_on_grid(grid, core_params)
    
    # 3. Bound States
    # We solve for enough states to catch the requested index
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+1)
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=chan.n_index_f+1)

    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
    
    epsilon_exc = orb_f.energy_au
    dE_target_au = orb_f.energy_au - orb_i.energy_au
    dE_target_eV = dE_target_au / ev_to_au(1.0)
    E_final_eV = E_incident_eV - dE_target_eV
    
    if E_final_eV <= 0.0:
        return DWBAResult(False, E_incident_eV, dE_target_eV, 0.0, 0.0, 0.0, 0.0)

    k_i_au = float(k_from_E_eV(E_incident_eV))
    k_f_au = float(k_from_E_eV(E_final_eV))
    z_ion = core_params.Zc - 1.0

    # 4. Distorting Potentials (Static Only - no exchange in U)
    U_i, U_f = build_distorting_potentials(
        grid=grid,
        V_core_array=V_core,
        orbital_initial=orb_i,
        orbital_final=orb_f,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
        use_exchange=False # STRICTLY STATIC per article
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
    
    # print(f"  [Auto-L] E={E_incident_eV:.1f} eV (k={k_i_au:.2f}) -> L_max_proj={L_max_proj}") 
    
    # --- Execution Strategy Selection ---
    USE_GPU = False
    if HAS_CUPY:
        # Perform a runtime check (e.g. NVRTC availability)
        if check_cupy_runtime():
            USE_GPU = True
        else:
             print("  [GPU Info] CuPy detected but runtime check failed (missing NVRTC?). Fallback to CPU.")
    
    if USE_GPU:
        print(f"  [GPU Accelerated] Summing Partial Waves l_i=0..{L_max_proj} on GPU (Single Process)...")
        # Sequential Loop, but with fast GPU integrals
        
        for l_i in range(L_max_proj + 1):
             # Logic similar to worker but sequential and utilizing GPU integrals where possible
             # To avoid code duplication, we could call a GPU-specific worker or inline here.
             # Inline is safer for accessing GPU context.
             
            # Check Parity (Early)
            target_parity_change = (Li + Lf) % 2
            
            # Solve chi_i (CPU - small overhead)
            # Re-solve is fine
            chi_i = solve_continuum_wave(grid, U_i, l_i, E_incident_eV, z_ion) 
            if chi_i is None: break
            
            lf_min = max(0, l_i - 10) 
            lf_max = l_i + 10
            
            for l_f in range(lf_min, lf_max + 1):
                if (l_i + l_f) % 2 != target_parity_change: continue
                
                try:
                    chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion)
                except:
                    chi_f = None
                if chi_f is None: continue
                
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
                        tgt = total_amplitudes[(Mi, Mf)]
                        tgt.f_theta += amps.f_theta
                        tgt.g_theta += amps.g_theta
            
            # Progress log for GPU path
            if l_i % 5 == 0:
                 print(f"    l_i={l_i} done.")

    else:
        # Fallback to CPU Parallel
        print(f"  [CPU Parallel] Summing Partial Waves l_i=0..{L_max_proj} (multiprocessing pool)...")
        
        # Parallel Execution using multiprocessing.Pool for better Ctrl+C handling
        import multiprocessing
        max_workers = os.cpu_count()
        if max_workers is None: max_workers = 1
        
        # Prepare arguments for starmap
        tasks = []
        for l_i in range(L_max_proj + 1):
            tasks.append((
                l_i, E_incident_eV, E_final_eV, z_ion, U_i, U_f,
                grid, V_core, orb_i, orb_f, chan, theta_grid, k_i_au, k_f_au
            ))
            
        try:
            with multiprocessing.Pool(processes=max_workers) as pool:
                # Use imap_unordered to process results as they come in
                # We need a wrapper to unpack arguments if using starmap logic, 
                # but starmap matches arguments directly.
                
                # pool.starmap blocks, but we want to catch KeyboardInterrupt.
                # pool.map_async (or starmap_async) allows waiting with timeout or checking.
                # Alternatively, just use starmap inside try/except.
                
                results_iter = pool.starmap_async(_worker_partial_wave, tasks)
                
                # Wait for result with a loop to allow signal catching? 
                # Actually .get(timeout) allows catching signals.
                # But simplest is to just let it block and catch the interrupt.
                
                results = results_iter.get(timeout=None)
                
                for l_done, partial_amps in results:
                     for key, part_amp in partial_amps.items():
                        if key not in total_amplitudes:
                            total_amplitudes[key] = Amplitudes(
                                np.zeros_like(theta_grid, dtype=complex),
                                np.zeros_like(theta_grid, dtype=complex)
                            )
                        total_amplitudes[key].f_theta += part_amp.f_theta
                        total_amplitudes[key].g_theta += part_amp.g_theta

        except KeyboardInterrupt:
            print("\n[User Interrupt] Terminating worker processes...")
            pool.terminate()
            pool.join()
            print("[User Interrupt] Workers terminated.")
            raise
        except Exception as e:
             print(f"Pool Execution Failed: {e}")
             raise

    print(f"  Summation complete in {time.perf_counter() - t0:.3f} s")


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
    
    return DWBAResult(
        True, E_incident_eV, dE_target_eV,
        sigma_total_au, sigma_total_cm2,
        k_i_au, k_f_au
    )
