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
    RadialDWBAIntegrals,
)
from sigma_total import (
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
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
    k_f_au: float

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
    match_high_energy_eV: float = 1000.0
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
    theta_grid = np.linspace(0.0, np.pi, 200) # Grid for angular integration
    
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

    # Optimization: Cache continuum waves to avoid re-solving ODEs
    # Key: (l, energy_eV, z_ion) -> ContinuumWave
    continuum_cache: Dict[Tuple[int, float, float], ContinuumWave] = {}

    def get_continuum_wave(l_idx: int, E_val: float, z_val: float, U_pot: DistortingPotential) -> Optional[ContinuumWave]:
        key = (l_idx, E_val, z_val)
        if key in continuum_cache:
            return continuum_cache[key]
        try:
            cw = solve_continuum_wave(grid, U_pot, l_idx, E_val, z_val)
            continuum_cache[key] = cw
            return cw
        except Exception:
            # If high l fails (centrifugal barrier), return None
            return None

    # Loop over projectile l_i
    # We restore L_max_projectile default to 25 in datastruct (via caller or explicit set)
    # The user wanted optimization, and caching makes L=25 feasible.
    L_max_proj = chan.L_max_projectile if chan.L_max_projectile > 5 else 25 
    
    print(f"  Summing Partial Waves l_i=0..{L_max_proj} ...")
    
    for l_i in range(L_max_proj + 1):
        t_li_start = time.perf_counter()
        # Solve chi_i
        chi_i = get_continuum_wave(l_i, E_incident_eV, z_ion, U_i)
        if chi_i is None:
            break

        # Determine allowed l_f range
        # Triangle rule with L_target transfer?
        # Actually Eq 216 sums over all M which implies all coupled L.
        # General conservation of parity: l_i + Li = l_f + Lf (parity) ?
        # Parity: (-1)^(li + Li) = (-1)^(lf + Lf).
        
        target_parity_change = (Li + Lf) % 2
        projectile_parity_must_change = target_parity_change
        
        # l_f usually close to l_i.
        lf_min = max(0, l_i - 10) 
        lf_max = l_i + 10
        
        for l_f in range(lf_min, lf_max + 1):
            # Check Parity Rule
            if (l_i + l_f) % 2 != target_parity_change:
                continue
                
            # Solve chi_f
            chi_f = get_continuum_wave(l_f, E_final_eV, z_ion, U_f)
            if chi_f is None:
                continue

            # Compute Radial Integrals
            # This is expensive.
            integrals = radial_ME_all_L(
                grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, chan.L_max_integrals
            )
            
            # Now distribute this (l_i, l_f) contribution to all (Mi, Mf) channels
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    # Compute Amplitude Contrib
                    amps = calculate_amplitude_contribution(
                        theta_grid, 
                        integrals.I_L_direct, 
                        integrals.I_L_exchange,
                        l_i, l_f, k_i_au, k_f_au,
                        Li, Lf, Mi, Mf
                    )
                    
                    # Accumulate
                    tgt = total_amplitudes[(Mi, Mf)]
                    tgt.f_theta += amps.f_theta
                    tgt.g_theta += amps.g_theta
        
        t_li_end = time.perf_counter()
        dt_li = t_li_end - t_li_start
        print(f"    l_i={l_i} done in {dt_li:.3f} s")


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
        
        # Unpolarized electrons: 1/4 |f+g|^2 + 3/4 |f-g|^2
        term_singlet = np.abs(f + g)**2
        term_triplet = np.abs(f - g)**2
        
        dcs_channel = spin_singlet_weight * term_singlet + spin_triplet_weight * term_triplet
        total_dcs += dcs_channel
        
    total_dcs *= prefac_kinematics
    total_dcs *= (chan.N_equiv / (2*Li + 1))
    
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
