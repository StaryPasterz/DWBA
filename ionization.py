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
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential,
    U_distorting # needed for manual manual U_ion construction
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
    # Detailed SDCS dSigma/dE can be added here if needed
    sdcs_data: Optional[List[Tuple[float, float]]] = None # (E_eject_eV, dcs_cm2/eV)





def compute_ionization_cs(
    E_incident_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-4,
    r_max: float = 200.0,
    n_points: int = 4000,
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
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0)

    E_total_final_eV = E_incident_eV - IP_eV
    
    # 3. Energy Integration Grid (E_eject integration)
    # We integrate from ~0 to E_total/2 (symmetric).
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    if steps[0] == 0.0:
        steps[0] = 0.5 * (steps[1] - steps[0]) * 0.1 # avoid exactly 0 energy

    sdcs_values_au = [] # dSigma/dE in a.u. (area/energy)

    # 4. Potentials
    # U_inc: Target=Neutral. Includes V_Hartree(orb_i).
    # We apply the selected model (Static / DWSE / SEP) to the Incident Channel.
    k_i_au_in = float(k_from_E_eV(E_incident_eV))
    U_inc_obj, _ = build_distorting_potentials(
        grid, V_core, orb_i, orb_i, 
        k_i_au=k_i_au_in, 
        use_exchange=use_exchange,
        use_polarization=use_polarization,
        exchange_method=exchange_method
    )
    
def _compute_sdcs_at_energy(
    E_eject_eV: float,
    E_incident_eV: float,
    E_total_final_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    grid: RadialGrid,
    V_core: np.ndarray,
    orb_i: BoundOrbital,
    U_inc_vals: np.ndarray, # Passed as array to be picklable/light
    V_hartree_inc: np.ndarray,
    use_gpu: bool
) -> float:
    """
    Worker to compute SDCS (dSigma/dE) at a specific ejected energy.
    Can be run in parallel (CPU) or sequentially (GPU).
    """
    E_scatt_eV = E_total_final_eV - E_eject_eV
    
    k_i_au = k_from_E_eV(E_incident_eV)
    k_scatt_au = k_from_E_eV(E_scatt_eV)
    k_eject_au = k_from_E_eV(E_eject_eV)
    
    z_ion_inc = core_params.Zc - 1.0
    z_ion_final = core_params.Zc
    
    # Reconstruct Potentials
    # U_inc
    U_inc_obj = DistortingPotential(U_of_r=U_inc_vals, V_hartree_of_r=V_hartree_inc)
    
    # U_ion: Pure core
    U_ion_vals = V_core
    U_ion_obj = DistortingPotential(U_of_r=U_ion_vals, V_hartree_of_r=np.zeros_like(V_core))
    
    sigma_energy_au = 0.0
    
    # Pre-calc chi_eject if possible? No, it depends on l_ej.
    
    theta_grid = np.linspace(0, np.pi, 200)

    for l_ej in range(chan.l_eject_max + 1):
        # Solve ejected wave
        chi_eject = solve_continuum_wave(grid, U_ion_obj, l_ej, E_eject_eV, z_ion=z_ion_final)
        if chi_eject is None: continue
        
        Li = chan.l_i
        Lf = l_ej 
        
        # Accumulate amplitudes
        total_amplitudes: Dict[Tuple[int, int], Amplitudes] = {}
        for Mi in range(-Li, Li+1):
            for Mf in range(-Lf, Lf+1):
                total_amplitudes[(Mi, Mf)] = Amplitudes(
                    np.zeros_like(theta_grid, dtype=complex),
                    np.zeros_like(theta_grid, dtype=complex)
                )

        L_max_proj = chan.L_max_projectile
        
        # Inner Projectile Loop
        # If we are in this worker on CPU, we run sequentially here 
        # (because we parallelized the outer loop).
        # If use_gpu is True, we run GPU logic here.
        
        if use_gpu and HAS_CUPY and check_cupy_runtime():
             # GPU PATH (Sequential inside worker, but worker might be sequential too)
             for l_i_proj in range(L_max_proj + 1):
                chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
                if chi_i is None: break
                
                lf_min = max(0, l_i_proj - 10)
                lf_max = l_i_proj + 10
                target_parity_change = (Li + Lf) % 2
                
                for l_f_proj in range(lf_min, lf_max + 1):
                    if (l_i_proj + l_f_proj) % 2 != target_parity_change: continue
                    try:
                        chi_scatt_wave = solve_continuum_wave(grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final)
                    except: chi_scatt_wave = None
                    if chi_scatt_wave is None: continue
                    
                    # GPU Integrals
                    integrals = radial_ME_all_L_gpu(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i,
                        bound_f=chi_eject, 
                        cont_i=chi_i,
                        cont_f=chi_scatt_wave,
                        L_max=chan.L_max
                    )
                    
                    for Mi in range(-Li, Li+1):
                        for Mf in range(-Lf, Lf+1):
                            amps = calculate_amplitude_contribution(
                                theta_grid, 
                                integrals.I_L_direct, 
                                integrals.I_L_exchange,
                                l_i_proj, l_f_proj, k_i_au, k_scatt_au,
                                Li, Lf, Mi, Mf
                            )
                            tgt = total_amplitudes[(Mi, Mf)]
                            tgt.f_theta += amps.f_theta
                            tgt.g_theta += amps.g_theta

        else:
            # CPU PATH (Sequential inner loop, Parallel outer loop)
            # No inner multiprocessing here to avoid nesting.
             for l_i_proj in range(L_max_proj + 1):
                chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
                if chi_i is None: break # Higher L won't work either usually
                
                lf_min = max(0, l_i_proj - 10)
                lf_max = l_i_proj + 10
                target_parity_change = (Li + Lf) % 2
    
                for l_f_proj in range(lf_min, lf_max + 1):
                    if (l_i_proj + l_f_proj) % 2 != target_parity_change: continue
                    
                    try:
                        chi_scatt_wave = solve_continuum_wave(grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final)
                    except: chi_scatt_wave = None
                    if chi_scatt_wave is None: continue
                        
                    # CPU Integrals
                    integrals = radial_ME_all_L(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i,
                        bound_f=chi_eject, 
                        cont_i=chi_i,
                        cont_f=chi_scatt_wave,
                        L_max=chan.L_max
                    )
                    
                    for Mi in range(-Li, Li+1):
                        for Mf in range(-Lf, Lf+1):
                            amps = calculate_amplitude_contribution(
                                theta_grid, 
                                integrals.I_L_direct, 
                                integrals.I_L_exchange,
                                l_i_proj, l_f_proj, k_i_au, k_scatt_au,
                                Li, Lf, Mi, Mf
                            )
                            tgt = total_amplitudes[(Mi, Mf)]
                            tgt.f_theta += amps.f_theta
                            tgt.g_theta += amps.g_theta
        
        # Calculate DCS for this l_eject
        total_dcs = np.zeros_like(theta_grid, dtype=float)
        for (Mi, Mf), amps in total_amplitudes.items():
            f = amps.f_theta
            g = amps.g_theta
            chan_dcs = dcs_dwba(
                theta_grid, f, g, 
                k_i_au, k_scatt_au, 
                Li, chan.N_equiv
            )
            total_dcs += chan_dcs
        
        dos_factor = 2.0 / (np.pi * k_eject_au)
        total_dcs *= dos_factor
        
        sigma_l_eject_au = integrate_dcs_over_angles(theta_grid, total_dcs)
        sigma_energy_au += sigma_l_eject_au
        
    return sigma_energy_au

def compute_ionization_cs(
    E_incident_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-4,
    r_max: float = 200.0,
    n_points: int = 4000,
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
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0)

    E_total_final_eV = E_incident_eV - IP_eV
    
    # 3. Energy Integration Grid (E_eject integration)
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    if steps[0] == 0.0:
        steps[0] = 0.5 * (steps[1] - steps[0]) * 0.1 

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
    # Check GPU availability
    USE_GPU = False
    if HAS_CUPY and check_cupy_runtime():
        USE_GPU = True
        
    print(f"  Ionization Scan E_inc={E_incident_eV:.1f} eV: Integrating dSigma/dE...")
    
    if USE_GPU:
        # GPU PATH: Run sequentially to avoid context conflicts
        # The inner worker uses GPU.
        print("    [Mode] GPU Acceleration (Sequential Energy Scan)")
        for idx_E, E_eject_eV in enumerate(steps):
             print(f"    Step {idx_E+1}/{len(steps)}: E_ej={E_eject_eV:.2f} eV...", end=" ", flush=True)
             val = _compute_sdcs_at_energy(
                 E_eject_eV, E_incident_eV, E_total_final_eV,
                 chan, core_params, grid, V_core, orb_i,
                 U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                 True # use_gpu
             )
             sdcs_values_au.append(val)
             print(f"SDCS={val:.2e}")
             
    else:
        # CPU PATH: Run Parallel on Energies (Coarse-Grained)
        import multiprocessing
        max_workers = os.cpu_count()
        if max_workers is None: max_workers = 1
        
        print(f"    [Mode] CPU Parallel (Workers={max_workers})")
        
        # Prepare valid static args for pickling
        # Note: DistortingPotential involves arrays, better pass as np.ndarray
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            # Create tasks
            tasks = []
            for E_eject_eV in steps:
                tasks.append((
                     E_eject_eV, E_incident_eV, E_total_final_eV,
                     chan, core_params, grid, V_core, orb_i,
                     U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                     False # use_gpu
                ))
            
            # Map
            results = pool.starmap(_compute_sdcs_at_energy, tasks)
            sdcs_values_au = results
            
            # Print brief summary
            for i, val in enumerate(sdcs_values_au):
                 print(f"    Step {i+1}: E_ej={steps[i]:.2f} eV -> SDCS={val:.2e}")

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
        sdcs_data_list
    )
