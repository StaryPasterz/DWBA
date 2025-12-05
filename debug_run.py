from grid import make_r_grid, k_from_E_eV, ev_to_au
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from driver import _pick_bound_orbital, ExcitationChannelSpec
from distorting_potential import build_distorting_potentials
from continuum import solve_continuum_wave
from dwba_matrix_elements import radial_ME_all_L
from dwba_coupling import ChannelAngularInfo, build_angular_coeffs_for_channel
from sigma_total import f_theta_from_coeffs, dcs_dwba, integrate_dcs_over_angles, sigma_au_to_cm2

import numpy as np

def main():
    # --- parametry testowe ---
    core_params = CorePotentialParams(
        Zc = 2.0,
        a1 = 8.043,
        a2 = 2.715,
        a3 = 0.506,
        a4 = 0.982,
        a5 = -0.043,
        a6 = 0.401,
    )

    chan = ExcitationChannelSpec(
        l_i=1,
        l_f=0,
        n_index_i=1,
        n_index_f=2,
        N_equiv=5,
        L_max=2,
        L_i_total=1,
    )

    E_incident_eV = 200.0

    # mała siatka diagnostyczna
    r_min = 0.1
    r_max = 50.0
    n_points = 200


    print("checkpoint 1: building grid")
    grid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)

    print("checkpoint 2: core potential")
    V_core = V_core_on_grid(grid, core_params)

    print("checkpoint 3: bound states solve (initial l_i)")
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=max(chan.n_index_i, chan.n_index_i+2))

    print("checkpoint 4: bound states solve (final l_f)")
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=max(chan.n_index_f, chan.n_index_f+2))

    print("checkpoint 5: pick bound orbitals")
    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)

    # energia wzbudzenia (w eV)
    dE_target_au = (orb_f.energy_au - orb_i.energy_au)
    dE_target_eV = dE_target_au / ev_to_au(1.0)
    print(f"excitation threshold ~ {dE_target_eV:.4f} eV")

    E_final_eV = E_incident_eV - dE_target_eV
    if E_final_eV <= 0:
        print("channel closed at 200 eV -> stop here (to jest ok fizycznie)")
        return

    k_i_au = float(k_from_E_eV(E_incident_eV))
    k_f_au = float(k_from_E_eV(E_final_eV))

    print(f"k_i={k_i_au:.4f} a.u., k_f={k_f_au:.4f} a.u.")

    print("checkpoint 6: build distorting potentials")
    U_i, U_f = build_distorting_potentials(
        grid=grid,
        V_core_array=V_core,
        orbital_initial=orb_i,
        orbital_final=orb_f
    )

    print("checkpoint 7: solve continuum incoming")
    chi_i = solve_continuum_wave(
        grid=grid,
        U_channel=U_i,
        l=chan.l_i,
        E_eV=E_incident_eV
    )

    print("checkpoint 8: solve continuum outgoing")
    chi_f = solve_continuum_wave(
        grid=grid,
        U_channel=U_f,
        l=chan.l_f,
        E_eV=E_final_eV
    )

    print("checkpoint 9: DWBA radial integrals")
    radial_ints = radial_ME_all_L(
        grid=grid,
        V_core_array=V_core,
        U_i_array=U_i.U_of_r,
        bound_i=orb_i,
        bound_f=orb_f,
        cont_i=chi_i,
        cont_f=chi_f,
        L_max=chan.L_max
    )
    I_L_dict = radial_ints.I_L
    print("I_L keys:", list(I_L_dict.keys()))

    print("checkpoint 10: angular coefficients F_L,G_L")
    chan_info = ChannelAngularInfo(
        l_i=chan.l_i,
        l_f=chan.l_f,
        S_i=0.5,  # placeholder spin
        S_f=0.5,
        L_i=float(chan.L_i_total),
        L_f=float(chan.l_f),
        J_i=float(chan.L_i_total),
        J_f=float(chan.l_f),
    )
    coeffs = build_angular_coeffs_for_channel(I_L_dict, chan_info)

    print("checkpoint 11: build amplitudes f(theta), g(theta)")
    theta_grid = np.linspace(0.0, np.pi, 200)
    cos_theta = np.cos(theta_grid)

    f_theta, g_theta = f_theta_from_coeffs(cos_theta, coeffs)

    print("checkpoint 12: dσ/dΩ(θ)")
    dcs_theta = dcs_dwba(
        theta_grid=theta_grid,
        f_theta=f_theta,
        g_theta=g_theta,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
        L_i=chan.L_i_total,
        N_equiv=chan.N_equiv
    )

    print("checkpoint 13: integrate over angle")
    sigma_total_au = integrate_dcs_over_angles(theta_grid, dcs_theta)
    sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)

    print("DONE.")
    print("sigma_total_au :", sigma_total_au)
    print("sigma_total_cm2:", sigma_total_cm2)

if __name__ == "__main__":
    main()
