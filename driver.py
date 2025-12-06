# driver.py
#
# High-level orchestration for DWBA excitation calculations.
#
# Units:
#   - Internally everything is atomic units (hartree for energies, bohr for length).
#   - User-facing energy is in eV.
#   - Cross sections reported in both a0^2 and cm^2.
#

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
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
    DWBAAngularCoeffs,
    f_theta_from_coeffs,
    dcs_dwba,
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
)
from dwba_coupling import (
    ChannelAngularInfo,
    build_angular_coeffs_for_channel,
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
    L_max: int
    L_i_total: int


@dataclass(frozen=True)
class DWBAResult:
    """
    Final result bundle for one excitation channel.
    """
    ok_open_channel: bool
    E_incident_eV: float
    E_excitation_eV: float
    sigma_total_au: float
    sigma_total_cm2: float
    sigma_mtong_au: float
    sigma_mtong_cm2: float
    k_i_au: float
    k_f_au: float


def _pick_bound_orbital(orbs: Tuple[BoundOrbital, ...], n_index_wanted: int) -> BoundOrbital:
    for o in orbs:
        if o.n_index == n_index_wanted:
            return o
    raise ValueError(f"Requested bound state n_index={n_index_wanted} not found.")


def _compute_excitation_energy_au(orb_i: BoundOrbital, orb_f: BoundOrbital) -> float:
    return orb_f.energy_au - orb_i.energy_au


def _apply_mtong_scaling(
    sigma_total_au: float,
    E_incident_eV: float,
    E_excitation_eV: float
) -> float:
    """
    Apply M-Tong scaling (or BE-scaling) to the DWBA cross section.
    Formula: sigma_scaled(E) = [ E / (E + E_exc) ] * sigma_DWBA(E)
    """
    if E_incident_eV <= 0.0:
        return 0.0
    
    scaling_factor = E_incident_eV / (E_incident_eV + E_excitation_eV)
    return sigma_total_au * scaling_factor


def compute_total_excitation_cs(
    E_incident_eV: float,
    chan: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 4000,
) -> DWBAResult:
    """
    Main high-level function.
    Calculates excitation cross section in DWBA including M-Tong scaling.
    """

    t0 = time.perf_counter()
    
    # 1. Grid
    grid: RadialGrid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    t1 = time.perf_counter()

    # 2. Core potential
    V_core = V_core_on_grid(grid, core_params)
    t2 = time.perf_counter()

    # 3. Bound states
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=max(chan.n_index_i, chan.n_index_i+2))
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=max(chan.n_index_f, chan.n_index_f+2))

    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
    
    dE_target_au = _compute_excitation_energy_au(orb_i, orb_f)
    dE_target_eV = dE_target_au / ev_to_au(1.0)

    t3 = time.perf_counter()

    # 4. Distorting potentials
    U_i, U_f = build_distorting_potentials(
        grid=grid,
        V_core_array=V_core,
        orbital_initial=orb_i,
        orbital_final=orb_f
    )
    t4 = time.perf_counter()

    # 5. Continuum waves
    # Charge of the target seen by the scattering electron:
    # Target = Core + ActiveElectron.
    # Charge = Z_core - 1.
    z_ion = core_params.Zc - 1.0

    E_final_eV = E_incident_eV - dE_target_eV

    k_i_au = float(k_from_E_eV(E_incident_eV)) if E_incident_eV > 0.0 else 0.0
    k_f_au = float(k_from_E_eV(E_final_eV))    if E_final_eV    > 0.0 else 0.0

    channel_open = (E_final_eV > 0.0) and (k_i_au > 0.0) and (k_f_au > 0.0)

    if not channel_open:
        return DWBAResult(
            ok_open_channel=False,
            E_incident_eV=E_incident_eV,
            E_excitation_eV=dE_target_eV,
            sigma_total_au=0.0,
            sigma_total_cm2=0.0,
            sigma_mtong_au=0.0,
            sigma_mtong_cm2=0.0,
            k_i_au=k_i_au,
            k_f_au=k_f_au,
        )

    chi_i: ContinuumWave = solve_continuum_wave(
        grid=grid,
        U_channel=U_i,
        l=chan.l_i,
        E_eV=E_incident_eV,
        z_ion=z_ion   # Pass ionic charge
    )

    chi_f: ContinuumWave = solve_continuum_wave(
        grid=grid,
        U_channel=U_f,
        l=chan.l_f, 
        E_eV=E_final_eV,
        z_ion=z_ion   # Pass ionic charge
    )

    t5 = time.perf_counter()

    # 6. Radial integrals
    radial_ints: RadialDWBAIntegrals = radial_ME_all_L(
        grid=grid,
        V_core_array=V_core,
        U_i_array=U_i.U_of_r,
        bound_i=orb_i,
        bound_f=orb_f,
        cont_i=chi_i,
        cont_f=chi_f,
        L_max=chan.L_max
    )
    I_L_dict: Dict[int, float] = radial_ints.I_L

    t6 = time.perf_counter()

    # 7. Angular coefficients
    chan_info = ChannelAngularInfo(
        l_i=chan.l_i,
        l_f=chan.l_f,
        S_i=float(0.5),
        S_f=float(0.5),
        L_i=float(chan.L_i_total),
        L_f=float(chan.l_f),
        J_i=float(chan.L_i_total),
        J_f=float(chan.l_f),
        k_i_au=k_i_au,
        k_f_au=k_f_au,
    )
    coeffs = build_angular_coeffs_for_channel(I_L_dict, chan_info)

    # 8. Amplitudes and Cross Sections
    theta_grid = np.linspace(0.0, np.pi, 400)
    cos_theta = np.cos(theta_grid)

    f_theta, g_theta = f_theta_from_coeffs(cos_theta, coeffs)

    dcs_theta = dcs_dwba(
        theta_grid=theta_grid,
        f_theta=f_theta,
        g_theta=g_theta,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
        L_i=chan.L_i_total,
        N_equiv=chan.N_equiv
    )

    sigma_total_au = integrate_dcs_over_angles(theta_grid, dcs_theta)
    sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)

    # 9. M-Tong scaling
    sigma_mtong_au = _apply_mtong_scaling(
        sigma_total_au=sigma_total_au,
        E_incident_eV=E_incident_eV,
        E_excitation_eV=dE_target_eV
    )
    sigma_mtong_cm2 = sigma_au_to_cm2(sigma_mtong_au)

    print("Timing diagnostic:")
    print(f"  grid            {t1-t0:.3f} s")
    print(f"  V_core          {t2-t1:.3f} s")
    print(f"  bound states    {t3-t2:.3f} s")
    print(f"  distorting pots {t4-t3:.3f} s")
    print(f"  continuum waves {t5-t4:.3f} s")
    print(f"  DWBA radial     {t6-t5:.3f} s")

    return DWBAResult(
        ok_open_channel=True,
        E_incident_eV=E_incident_eV,
        E_excitation_eV=dE_target_eV,
        sigma_total_au=sigma_total_au,
        sigma_total_cm2=sigma_total_cm2,
        sigma_mtong_au=sigma_mtong_au,
        sigma_mtong_cm2=sigma_mtong_cm2,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
    )
