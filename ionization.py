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
    RadialDWBAIntegrals,
)
from dwba_coupling import (
    calculate_amplitude_contribution,
    Amplitudes
)
from sigma_total import (
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
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
    n_energy_steps: int = 10
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
    # We allow exchange=False (Static) consistent with driver.py default.
    k_i_au_in = float(k_from_E_eV(E_incident_eV))
    U_inc_obj, _ = build_distorting_potentials(grid, V_core, orb_i, orb_i, k_i_au=k_i_au_in, use_exchange=False)
    
    # U_ion: Target=Ion. Pure core potential. Ejected & Scattered electrons see this.
    # No hartree part.
    U_ion_vals = V_core # Assuming only core
    U_ion_obj = DistortingPotential(U_of_r=U_ion_vals, V_hartree_of_r=np.zeros_like(V_core))

    # Caching continuum waves for this energy step is useful for l_eject loop
    # but since E_eject changes, we can't cache across energy steps much.
    
    # Loop over Ejected Energy
    print(f"  Ionization Scan E_inc={E_incident_eV:.1f} eV: Integrating dSigma/dE...")
    
    for idx_E, E_eject_eV in enumerate(steps):
        E_scatt_eV = E_total_final_eV - E_eject_eV
        
        k_i_au = k_from_E_eV(E_incident_eV)
        k_scatt_au = k_from_E_eV(E_scatt_eV)
        k_eject_au = k_from_E_eV(E_eject_eV)
        
        # Effective Z for Coulomb tails
        z_ion_inc = core_params.Zc - 1.0   # e + Neutral
        z_ion_final = core_params.Zc       # e + Ion+
        
        # Accumulate cross section for this energy partition
        sigma_energy_au = 0.0

        # Loop over Ejected Angular Momentum l_ej
        for l_ej in range(chan.l_eject_max + 1):
            
            # Solve ejected wave (acts as 'final bound state')
            # Normalized to unit amplitude -> Need to apply DoS factor later.
            try:
                chi_eject = solve_continuum_wave(grid, U_ion_obj, l_ej, E_eject_eV, z_ion=z_ion_final)
            except:
                continue

            # Now we run the Standard DWBA Partial Wave Loop 
            # treating chi_eject as 'orb_f' with L_target_f = l_ej.
            
            Li = chan.l_i
            Lf = l_ej # Final "target" angular momentum is just the ejected electron L
            
            theta_grid = np.linspace(0, np.pi, 200)
            
            # Dictionary for amplitudes: (Mi, Mf) -> Amplitudes
            # Mi: -Li..Li
            # Mf: -Lf..Lf (represents m_eject)
            total_amplitudes: Dict[Tuple[int, int], Amplitudes] = {}
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    total_amplitudes[(Mi, Mf)] = Amplitudes(
                        np.zeros_like(theta_grid, dtype=complex),
                        np.zeros_like(theta_grid, dtype=complex)
                    )

            # --- Projectile Partial Wave Loop ---
            # Adapted from driver.py
            L_max_proj = chan.L_max_projectile
            
            for l_i_proj in range(L_max_proj + 1):
                # chi_i (incident)
                try:
                    chi_i = solve_continuum_wave(grid, U_inc_obj, l_i_proj, E_incident_eV, z_ion=z_ion_inc)
                except:
                    break
                
                # l_f range
                lf_min = max(0, l_i_proj - 10)
                lf_max = l_i_proj + 10
                target_parity_change = (Li + Lf) % 2
                
                for l_f_proj in range(lf_min, lf_max + 1):
                    if (l_i_proj + l_f_proj) % 2 != target_parity_change:
                        continue
                        
                    try:
                        chi_scatt_wave = solve_continuum_wave(grid, U_ion_obj, l_f_proj, E_scatt_eV, z_ion=z_ion_final)
                    except:
                        continue
                        
                    # Matrix Elements
                    # bound_f is chi_eject
                    integrals = radial_ME_all_L(
                        grid, V_core, U_inc_obj.U_of_r,
                        bound_i=orb_i,
                        bound_f=chi_eject, 
                        cont_i=chi_i,
                        cont_f=chi_scatt_wave,
                        L_max=chan.L_max
                    )
                    
                    # Accumulate contribution to all magnetic sublevels
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
            
            # --- End Projectile Loop ---
            
            # Calculate DCS for this l_eject
            total_dcs = np.zeros_like(theta_grid, dtype=float)
            spin_weight = 1.0 # Simple sum? No, need spin stats.
            # Ionization from singlet ground state:
            # We assume unpolarized.
            # Same formulas apply: 1/4 S + 3/4 T.
            
            for (Mi, Mf), amps in total_amplitudes.items():
                f = amps.f_theta
                g = amps.g_theta
                term = (1.0/4.0)*np.abs(f+g)**2 + (3.0/4.0)*np.abs(f-g)**2
                total_dcs += term
            
            # Kinematic Prefactor (k_scatt / k_i)
            # AND Density of States for ejected electron: 1 / (pi * k_eject)
            prefac = (k_scatt_au / k_i_au) * (chan.N_equiv / (2*Li+1))
            dos_factor = 1.0 / (np.pi * k_eject_au)
            
            total_dcs *= (prefac * dos_factor)
            
            # Integrated over scattering angles
            sigma_l_eject_au = integrate_dcs_over_angles(theta_grid, total_dcs)
            
            sigma_energy_au += sigma_l_eject_au
            
        # Store SDCS (sigma per energy)
        sdcs_values_au.append(sigma_energy_au)
        print(f"    E_eject={E_eject_eV:.1f} eV: dSigma/dE = {sigma_energy_au:.2e} a.u.")

    # 5. Integrate SDCS over E_eject
    # Int [0, Etot/2] dSigma/dE dE
    # dE in Hartree needed for consistent units.
    
    steps_au = [ev_to_au(e) for e in steps]
    total_sigma_au = np.trapz(sdcs_values_au, steps_au)
    total_sigma_cm2 = sigma_au_to_cm2(total_sigma_au)
    
    # Store detailed data
    sdcs_data_list = list(zip(steps, [sigma_au_to_cm2(s*ev_to_au(1.0)) for s in sdcs_values_au]))
    # Note: sdcs_values_au is dS/dE_au. To get dS/dE_eV, we multiply by dE_au/dE_eV = 1/27.21 ?
    # Dimensionally: Sigma = Integrate[ (dS/dE)_au * dE_au ].
    # If we want plot vs eV: (dS/dE)_eV = (dS/dE)_au * (dE_au/dE_eV).
    
    print(f"  Total Ionization Sigma = {total_sigma_cm2:.2e} cm^2")

    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au,
        total_sigma_cm2,
        sdcs_data_list
    )
