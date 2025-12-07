
import numpy as np
from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials
from continuum import solve_continuum_wave
from dwba_matrix_elements import radial_ME_all_L
from dwba_coupling import calculate_amplitude_contribution
from driver import ExcitationChannelSpec

def check_magnitude():
    # 1. Setup
    E_inc_eV = 12.0 # Near threshold (10.2)
    Z = 1.0
    r_max = 100.0
    
    grid = make_r_grid(r_min=1e-4, r_max=r_max, n_points=2000)
    # For H, parameters a1-a6 are 0.
    core = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core)
    
    # 2. Bound States 1s, 2s
    print("Solving Bound States...")
    states_0 = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    # Solve bound states 1s (l=0)
    states_0 = solve_bound_states(grid, V_core, l=0, n_states_max=1)
    orb_1s = states_0[0]

    # Solve bound states 2p (l=1)
    states_1 = solve_bound_states(grid, V_core, l=1, n_states_max=1)
    orb_2p = states_1[0]
    
    orb_f = orb_2p
    name_f = "2p"
    
    print(f"1s E={orb_1s.energy_au:.4f}")
    print(f"2p E={orb_2p.energy_au:.4f} (Exp -0.125)")
    
    dE = orb_f.energy_au - orb_1s.energy_au
    dE_eV = dE * 27.211
    E_final_eV = E_inc_eV - dE_eV
    
    print(f"E_inc={E_inc_eV} eV, dE={dE_eV:.2f} eV, E_final={E_final_eV:.2f} eV")
    
    k_i = k_from_E_eV(E_inc_eV)
    k_f = k_from_E_eV(E_final_eV)
    print(f"k_i={k_i:.4f}, k_f={k_f:.4f} a.u.")
    
    # 3. Continuum
    print("Building Distorting Potentials...")
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_f, k_i, k_f, use_exchange=False)
    
    print("Solving Continuum Waves l=0...")
    chi_i = solve_continuum_wave(grid, U_i, 0, E_inc_eV)
    chi_f = solve_continuum_wave(grid, U_f, 0, E_final_eV)
    
    print(f"Chi_i norm check: max={np.max(np.abs(chi_i.chi_of_r)):.4f}")
    
    # 4. Radial Integrals
    print("Computing Radial Integrals (L_max=5)...")
    integrals = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_1s, orb_f, chi_i, chi_f, L_max=5)
    
    print("Direct Integrals:")
    for L, val in integrals.I_L_direct.items():
        print(f"  L={L}: {val:.4e}")
        
    print("Exchange Integrals:")
    for L, val in integrals.I_L_exchange.items():
        print(f"  L={L}: {val:.4e}")
        
    # 5. Amplitude (l_i=0 -> l_f=1)
    # Target 1s->2p (Li=0 -> Lf=1)
    
    theta = np.linspace(0, np.pi, 2)
    
    print("Computing Amplitude Contribution (l=0->1 channel)...")
    amps = calculate_amplitude_contribution(
        theta, 
        integrals.I_L_direct, 
        integrals.I_L_exchange,
        l_i=0, l_f=1,
        ki=k_i, kf=k_f,
        L_target_i=0, L_target_f=1,
        M_target_i=0, M_target_f=0
    )
    
    f_mag = np.abs(amps.f_theta)
    g_mag = np.abs(amps.g_theta)
    print(f"|f(0)| = {f_mag[0]:.4e}")
    print(f"|g(0)| = {g_mag[0]:.4e}")
    
    # Cross section Eq 216: N * (2pi)^4 * (kf/ki) * ...
    # N=1 for H.
    dcs_val = (k_f / k_i) * (0.25 * np.abs(amps.f_theta + amps.g_theta)**2 + 0.75 * np.abs(amps.f_theta - amps.g_theta)**2)
    dcs_val *= (2*np.pi)**4
    
    print(f"DCS(0) with (2pi)^4 = {dcs_val[0]:.4e} a.u.")
    
    # Estimate TCS
    sigma_est = 4 * np.pi * dcs_val[0] 
    print(f"Est. Partial TCS (l=0) = {sigma_est:.4e} a.u.")
    
    # Convert to cm2
    print(f"Est. Partial TCS = {sigma_est * 2.8e-17:.4e} cm2")

if __name__ == "__main__":
    check_magnitude()
