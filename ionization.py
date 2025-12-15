# ionization.py
#
# DWBA electron-impact ionization calculator.

from __future__ import annotations
import numpy as np
import time
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

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
    
    # Projectile partial wave limit (same as in driver.py)
    L_max_projectile: int = 25

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
    Worker to compute SDCS using Partial Wave analysis.
    Returns (sdcs_value_au, partial_waves_dict_au).
    """
    E_scatt_eV = E_total_final_eV - E_eject_eV
    
    k_i_au = k_from_E_eV(E_incident_eV)
    k_scatt_au = k_from_E_eV(E_scatt_eV)
    k_eject_au = k_from_E_eV(E_eject_eV)
    
    z_ion_inc = core_params.Zc - 1.0
    z_ion_final = core_params.Zc
    
    # U_inc_obj can be reconstructed from arrays if passed, or passed as object
    # In parallel call, we pass arrays. Reconstruct:
    U_inc_obj = DistortingPotential(U_of_r=U_inc_vals, V_hartree_of_r=V_hartree_inc)
    
    U_ion_vals = V_core
    U_ion_obj = DistortingPotential(U_of_r=U_ion_vals, V_hartree_of_r=np.zeros_like(V_core))
    
    sigma_energy_au = 0.0
    partial_L_contribs = {}
    theta_grid = np.linspace(0, np.pi, 200)
    
    for l_ej in range(chan.l_eject_max + 1):
        chi_eject = solve_continuum_wave(grid, U_ion_obj, l_ej, E_eject_eV, z_ion=z_ion_final)
        if chi_eject is None: continue
        
        Li = chan.l_i
        Lf = l_ej 
        L_max_proj = chan.L_max_projectile
        
        for l_i_proj in range(L_max_proj + 1):
            # --- Solve Incident Wave ---
            # Try cache first
            if chi_i_cache is not None and l_i_proj in chi_i_cache:
                chi_i = chi_i_cache[l_i_proj]
            else:
                chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
            
            if chi_i is None: break 
            
            sigma_this_Li = 0.0
            
            # Compatible l_f_proj
            lf_min = max(0, l_i_proj - 10)
            lf_max = l_i_proj + 10
            target_parity_change = (Li + Lf) % 2
            
            amps_shell = {}
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    amps_shell[(Mi, Mf)] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )
            
            # Cache for scattered waves within this energy step
            chi_scatt_cache: Dict[int, Optional[ContinuumWave]] = {}

            for l_f_proj in range(lf_min, lf_max + 1):
                if (l_i_proj + l_f_proj) % 2 != target_parity_change: continue
                
                # Check cache for chi_scatt
                if l_f_proj in chi_scatt_cache:
                    chi_scatt_wave = chi_scatt_cache[l_f_proj]
                else:
                    try:
                        chi_scatt_wave = solve_continuum_wave(grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final)
                    except: 
                        chi_scatt_wave = None
                    chi_scatt_cache[l_f_proj] = chi_scatt_wave
                
                if chi_scatt_wave is None: continue
                
                # Integrals
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

            # Now compute DCS for this l_i_proj and l_eject
            dcs_shell = np.zeros_like(theta_grid, dtype=float)
            for (Mi, Mf), amp in amps_shell.items():
                chan_dcs = dcs_dwba(
                    theta_grid, amp.f_theta, amp.g_theta, 
                    k_i_au, k_scatt_au, Li, chan.N_equiv
                )
                dcs_shell += chan_dcs
            
            dos_factor = 2.0 / (np.pi * k_eject_au)
            dcs_shell *= dos_factor
            
            sigma_shell_au = integrate_dcs_over_angles(theta_grid, dcs_shell)
            sigma_energy_au += sigma_shell_au
            key = f"L{l_i_proj}"
            partial_L_contribs[key] = partial_L_contribs.get(key, 0.0) + sigma_shell_au
            
    return sigma_energy_au, partial_L_contribs

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
    exchange_method: str = 'fumc'
) -> IonizationResult:
    """
    Calculate Total Ionization Cross Section (TICS) via Partial Wave DWBA.
    """

    # 1. Setup Grid & Potential
    grid = make_r_grid(r_min, r_max, n_points)
    V_core = V_core_on_grid(grid, core_params)

    # 2. Initial Bound State
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
    
    # 3. Energy Integration Grid (E_eject integration)
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    # Hard clamp to minimal safe energy for continuum solver
    safe_min_eV = 0.1
    # Ensure no step is below safe_min
    steps[steps < safe_min_eV] = safe_min_eV
    
    # If checking strict spacing, we might want to ensure they don't collapse,
    # but the integral is robust enough to handle duplicates or we can unique-ify.
    # For now, just clamping the start is enough as linspace is monotonic.
    # Actually, let's just shift the start:
    if steps[0] < safe_min_eV:
         steps[0] = safe_min_eV
         
    # Ensure monotonicity if step 1 was < 0.1?
    # If total range is very small (e.g. 0 to 0.5), steps might be [0.0, 0.05, 0.10...]
    # If we clamp steps[0]=0.1, we might have [0.1, 0.05...] -> invalid.
    # Better:
    steps = np.maximum(steps, safe_min_eV)
    steps = np.unique(steps) # Remove duplicates if multiple clamped to 0.1
    if len(steps) < 2:
         # Recover if we collapsed everything
         steps = np.linspace(safe_min_eV, max(safe_min_eV+0.5, E_total_final_eV/2.0), n_energy_steps+1) 

    sdcs_values_au = [] 
    
    # 4. Potentials
    k_i_au_in = float(k_from_E_eV(E_incident_eV))
    U_inc_obj, _ = build_distorting_potentials(
        grid, V_core, orb_i, orb_i, 
        k_i_au=k_i_au_in, 
        use_exchange=use_exchange,
        use_polarization=use_polarization,
        exchange_method=exchange_method
    )
    
    # 5. Energy Integration Implementation
    USE_GPU = False
    if HAS_CUPY and check_cupy_runtime():
        USE_GPU = True
        
    print(f"  Ionization Scan E_inc={E_incident_eV:.1f} eV: Integrating dSigma/dE...")
    
    if USE_GPU:
        print("    [Mode] GPU Acceleration (Sequential Energy Scan)")
        
        # Precompute incident waves (chi_i) once for GPU path too!
        print(f"    [Pre-calc] Caching incident waves up to L={chan.L_max_projectile}...")
        t_pre = time.perf_counter()
        chi_i_cache = precompute_continuum_waves(
            chan.L_max_projectile, E_incident_eV, 
            core_params.Zc - 1.0, U_inc_obj, grid
        )
        print(f"    [Pre-calc] Done in {time.perf_counter()-t_pre:.3f}s. Cached {len(chi_i_cache)} waves.")

        for idx_E, E_eject_eV in enumerate(steps):
             print(f"    Step {idx_E+1}/{len(steps)}: E_ej={E_eject_eV:.2f} eV...", end=" ", flush=True)
             val, partials = _compute_sdcs_at_energy(
                 E_eject_eV, E_incident_eV, E_total_final_eV,
                 chan, core_params, grid, V_core, orb_i,
                 U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                 True, # use_gpu
                 chi_i_cache # Pass the cache!
             )
             sdcs_values_au.append(val)
             print(f"SDCS={val:.2e}")
             
    else:
        import multiprocessing
        max_workers = os.cpu_count()
        if max_workers is None: max_workers = 1
        
        print(f"    [Mode] CPU Parallel (Workers={max_workers})")
        
        # Precompute incident waves (chi_i) once!
        print(f"    [Pre-calc] Caching incident waves up to L={chan.L_max_projectile}...")
        t_pre = time.perf_counter()
        chi_i_cache = precompute_continuum_waves(
            chan.L_max_projectile, E_incident_eV, 
            core_params.Zc - 1.0, U_inc_obj, grid
        )
        print(f"    [Pre-calc] Done in {time.perf_counter()-t_pre:.3f}s. Cached {len(chi_i_cache)} waves.")
        
        # Prepare Tasks
        tasks = []
        for E_eject_eV in steps:
            tasks.append((
                 E_eject_eV, E_incident_eV, E_total_final_eV,
                 chan, core_params, grid, V_core, orb_i,
                 U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                 False, # use_gpu
                 chi_i_cache 
            ))
            
        # Use imap for live progress
        # Wrapper to unpack args because imap takes 1 arg
        
        try:
            with multiprocessing.Pool(processes=max_workers) as pool:
                # Helper: _wrapper_sdcs_helper takes a SINGLE tuple argument
                sdcs_values_au = []
                total_steps = len(tasks)
                
                # Use imap
                # imap returns an iterator. We iterate it.
                # If KeyboardInterrupt happens during iteration, we must catch it and terminate pool.
                iterator = pool.imap(_wrapper_sdcs_helper, tasks)
                
                for i_step, result in enumerate(iterator):
                    val = result[0] # result is (val, partials)
                    sdcs_values_au.append(val)
                    print(f"    Step {i_step+1}/{total_steps}: E_ej={steps[i_step]:.2f} eV -> SDCS={val:.2e}")
                    
        except KeyboardInterrupt:
            print("\n    [Ionization] Interrupt detected! Terminating workers...")
            # pool context manager handles terminate on exit if exception occurs? 
            # Actually no, 'with' calls close/join usually unless exception.
            # If we are inside 'with', exception triggers __exit__. 
            # Pool.__exit__ calls terminate(). 
            # So just re-raising should be enough IF the signal reaches here.
            raise



    # 6. Integrate SDCS
    steps_au = [ev_to_au(e) for e in steps]
    total_sigma_au = np.trapz(sdcs_values_au, steps_au)
    total_sigma_cm2 = sigma_au_to_cm2(total_sigma_au)
    
    print(f"  Total Ionization Sigma = {total_sigma_cm2:.2e} cm^2")
    
    sdcs_data_list = list(zip(steps, [sigma_au_to_cm2(s*ev_to_au(1.0)) for s in sdcs_values_au]))

    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au,
        total_sigma_cm2,
        sdcs_data_list,
        {} # partials not aggregated fully in result here for brevity
    )

def _wrapper_sdcs_helper(args):
    """Helper for imap unpacking."""
    return _compute_sdcs_at_energy(*args)
