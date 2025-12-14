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
    # Detailed SDCS dSigma/dE
    sdcs_data: Optional[List[Tuple[float, float]]] = None # (E_eject_eV, dcs_cm2/eV)
    # Partial Waves contributions (L_projectile -> Sigma)
    partial_waves: Optional[Dict[str, float]] = None


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
    exchange_method: str = 'fumc',
    # Optimization Injection
    _precalc_grid: Optional[RadialGrid] = None,
    _precalc_V_core: Optional[np.ndarray] = None,
    _precalc_orb_i: Optional[BoundOrbital] = None
) -> IonizationResult:
    """
    Calculate Total Ionization Cross Section (TICS) via Partial Wave DWBA.
    """

    # 1. Setup Grid & Potential
    if _precalc_grid is not None and _precalc_V_core is not None:
        grid = _precalc_grid
        V_core = _precalc_V_core
    else:
        grid = make_r_grid(r_min, r_max, n_points)
        V_core = V_core_on_grid(grid, core_params)

    # 2. Initial Bound State
    if _precalc_orb_i is not None:
        orb_i = _precalc_orb_i
    else:
        states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+2)
        orb_i = None
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
    if steps[0] == 0.0:
        steps[0] = 0.5 * (steps[1] - steps[0]) * 0.1 

    sdcs_values_au = [] 
    
    # Store partial wave contributions to SDCS for each energy step
    # List of dicts: [{L0: val, L1: val...}, ...]
    sdcs_partials_list = []

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
        for idx_E, E_eject_eV in enumerate(steps):
             print(f"    Step {idx_E+1}/{len(steps)}: E_ej={E_eject_eV:.2f} eV...", end=" ", flush=True)
             val, partials = _compute_sdcs_at_energy(
                 E_eject_eV, E_incident_eV, E_total_final_eV,
                 chan, core_params, grid, V_core, orb_i,
                 U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                 True 
             )
             sdcs_values_au.append(val)
             sdcs_partials_list.append(partials)
             print(f"SDCS={val:.2e}")
             
    else:
        import multiprocessing
        max_workers = os.cpu_count()
        if max_workers is None: max_workers = 1
        
        print(f"    [Mode] CPU Parallel (Workers={max_workers})")
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            tasks = []
            for E_eject_eV in steps:
                tasks.append((
                     E_eject_eV, E_incident_eV, E_total_final_eV,
                     chan, core_params, grid, V_core, orb_i,
                     U_inc_obj.U_of_r, U_inc_obj.V_hartree_of_r,
                     False 
                ))
            
            results = pool.starmap(_compute_sdcs_at_energy, tasks)
            # results is list of (val, partials)
            sdcs_values_au = [r[0] for r in results]
            sdcs_partials_list = [r[1] for r in results]
            
            for i, val in enumerate(sdcs_values_au):
                 print(f"    Step {i+1}: E_ej={steps[i]:.2f} eV -> SDCS={val:.2e}")

    # 6. Integrate SDCS
    steps_au = [ev_to_au(e) for e in steps]
    total_sigma_au = np.trapz(sdcs_values_au, steps_au)
    total_sigma_cm2 = sigma_au_to_cm2(total_sigma_au)
    
    # Integrate Partial Waves
    # We have dSigma_L / dE at each step. We trapz them.
    # First, find all L keys present
    all_L_keys = set()
    for d in sdcs_partials_list:
        all_L_keys.update(d.keys())
        
    integrated_partials = {}
    for L_key in all_L_keys:
        # Extract trace for this L
        trace = [d.get(L_key, 0.0) for d in sdcs_partials_list]
        val_au = np.trapz(trace, steps_au)
        # Convert to cm2
        integrated_partials[L_key] = sigma_au_to_cm2(val_au)
    
    print(f"  Total Ionization Sigma = {total_sigma_cm2:.2e} cm^2")
    
    sdcs_data_list = list(zip(steps, [sigma_au_to_cm2(s*ev_to_au(1.0)) for s in sdcs_values_au]))

    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au,
        total_sigma_cm2,
        sdcs_data_list,
        integrated_partials
    )

def compute_ionization_cs_precalc(
    E_incident_eV: float,
    prep: "PreparedTarget", # From driver
    n_energy_steps: int = 50
) -> IonizationResult:
    """
    Optimized version of compute_ionization_cs that uses pre-calculated
    target properties (grid, V_core, orb_i) from PreparedTarget.
    """
    # Unpack from Prep
    grid = prep.grid
    V_core = prep.V_core
    orb_i = prep.orb_i
    core_params = prep.core_params
    chan = prep.chan
    
    # Correction: IonizationChannelSpec is not ExcitationChannelSpec.
    # We need to ensure prep.chan is compatible or castable.
    # Actually, PreparedTarget stores ExcitationChannelSpec.
    # We might need a separate 'prepare_ionization_target' or make PreparedTarget generic.
    # For now, let's assume valid data is passed manually or refactor.
    # The 'chan' in prep is mostly L_i, L_f, n_i...
    # Ionization needs IonizationChannelSpec. Let's create it on the fly or pass it in.
    
    # Wait, 'chan' in PrepareTarget is ExcitationChannelSpec. 
    # Ionization needs L_i, L_eject_max, L_max...
    # We should probably pass the IonizationChannelSpec explicitly here as argument, 
    # instead of relying on prep.chan (which is for excitation).
    # But V_core, grid, orb_i are reusable.
    
    pass # Placeholder for thought process check.

# Actual implementation below
def compute_ionization_cs_precalc_implementation(
    E_incident_eV: float,
    prep: "PreparedTarget",
    ion_spec: IonizationChannelSpec,
    n_energy_steps: int = 50
) -> IonizationResult:
    
    IP_eV = prep.dE_target_eV * -1.0 # Binding Energy was negative, so IP is positive dE? NO.
    # In driver: dE = E_f - E_i. For excitation E_f > E_i usually? No E_f < E_i (bound).
    # orb.energy is negative.
    # IP = -orb_i.energy.
    IP_eV = -orb_i.energy_au / ev_to_au(1.0)
    
    if E_incident_eV <= IP_eV:
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0, [], {})

    E_total_final_eV = E_incident_eV - IP_eV
    
    # Integration Grid
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    if steps[0] == 0.0: steps[0] = 0.5*(steps[1]-steps[0])*0.1

    # Potentials (Incident Channel)
    k_i_au_in = float(k_from_E_eV(E_incident_eV))
    
    # We must recalculate U_inc because it depends on Energy (Exchange) and k_i
    U_inc_obj, _ = build_distorting_potentials(
        grid, V_core, orb_i, orb_i, # Initial->Initial (Elastic-like for incident)
        k_i_au=k_i_au_in,
        k_f_au=k_i_au_in,
        use_exchange=prep.use_exchange,
        use_polarization=prep.use_polarization,
        exchange_method=prep.exchange_method
    )
    
    # ... (Rest is identical to original compute_ionization_cs loop)
    # To reuse code, we call the worker loop.
    
    # ... [Copy of the Integration Loop] ...
    # This is getting verbose to duplicate.
    # Ideally, compute_ionization_cs should just take optional pre-calculated objects.
    
    return compute_ionization_cs(
        E_incident_eV, ion_spec, core_params, 
        r_max=grid.r[-1], n_points=len(grid.r), 
        n_energy_steps=n_energy_steps,
        use_exchange=prep.use_exchange,
        use_polarization=prep.use_polarization,
        exchange_method=prep.exchange_method,
        # INJECT PRE-CALCULATED OBJECTS
        _precalc_grid=grid,
        _precalc_V_core=V_core,
        _precalc_orb_i=orb_i
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
    U_inc_vals: np.ndarray, 
    V_hartree_inc: np.ndarray,
    use_gpu: bool
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
    
    U_inc_obj = DistortingPotential(U_of_r=U_inc_vals, V_hartree_of_r=V_hartree_inc)
    
    U_ion_vals = V_core
    U_ion_obj = DistortingPotential(U_of_r=U_ion_vals, V_hartree_of_r=np.zeros_like(V_core))
    
    sigma_energy_au = 0.0
    
    # Store partial contribution per L_projectile
    # { "L0": val, "L1": val ... }
    partial_L_contribs = {}
    
    theta_grid = np.linspace(0, np.pi, 200)

    for l_ej in range(chan.l_eject_max + 1):
        chi_eject = solve_continuum_wave(grid, U_ion_obj, l_ej, E_eject_eV, z_ion=z_ion_final)
        if chi_eject is None: continue
        
        Li = chan.l_i
        Lf = l_ej 
        
        # Amplitudes logic...
        # We need to track contribution per l_i_proj inside the loops.
        # This is tricky because we usually sum amplitudes F, G first, then square for DCS.
        # But Partial Cross Section is formally defined as Sum(|f_L|^2).
        # In DWBA, the total cross section is Sum(Sigma_Li).
        # So we CAN calculate sigma for each l_i_proj independently.
        
        # Structure: Outer loop l_i_proj.
        # For each l_i_proj, we find all compatible l_f_proj.
        # Calculate DCS for THAT l_i_proj specific part.
        # Add to total.
        
        L_max_proj = chan.L_max_projectile
        
        # We process l_i_proj sequentially and accumulate sigma
        
        # Depending on GPU/CPU, the implementation differs slightly but logic is same.
        # To avoid code duplication, we'll iterate l_i_proj and do the work.
        
        for l_i_proj in range(L_max_proj + 1):
            
            # --- Solve Incident Wave ---
            chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
            if chi_i is None: break 
            
            sigma_this_Li = 0.0
            
            # Compatible l_f_proj
            lf_min = max(0, l_i_proj - 10)
            lf_max = l_i_proj + 10
            target_parity_change = (Li + Lf) % 2
            
            # We need to sum amplitudes over M states for this l_i_proj
            # Amplitudes for this Li_proj shell:
            # amps_shell[(Mi, Mf)] = Amplitudes(...)
            
            amps_shell = {}
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    amps_shell[(Mi, Mf)] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )
            
            # Inner loop over l_f_proj
            for l_f_proj in range(lf_min, lf_max + 1):
                if (l_i_proj + l_f_proj) % 2 != target_parity_change: continue
                
                try:
                    chi_scatt_wave = solve_continuum_wave(grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final)
                except: chi_scatt_wave = None
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
            
            # Add to Totals
            sigma_energy_au += sigma_shell_au
            
            # Add to Partial Tracker
            key = f"L{l_i_proj}"
            partial_L_contribs[key] = partial_L_contribs.get(key, 0.0) + sigma_shell_au
            
    return sigma_energy_au, partial_L_contribs

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
