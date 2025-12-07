# debug.py
#
# Diagnostic script to trace DWBA calculation steps and verify intermediate values using the V2 codebase.
#
# USAGE:
#   python debug.py
#
# Generates plots:
#   debug_bound_states.png
#   debug_potentials.png
#   debug_continuum.png
#   debug_integrands.png

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Add current directory to path if needed
sys.path.append(os.getcwd())

from grid import make_r_grid, ev_to_au, k_from_E_eV, integrate_trapz
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials
from continuum import solve_continuum_wave
from dwba_matrix_elements import radial_ME_all_L
from dwba_coupling import calculate_amplitude_contribution

def main():
    print("=== DWBA DEBUG DIAGNOSTICS ===")
    
    # --- 1. CONFIGURATION ---
    Z = 1.0  # Hydrogen
    E_inc_eV = 50.0
    transition = "1s -> 2s" # or "1s -> 2p"
    
    ni, li = 1, 0
    nf, lf = 2, 1 # 2p
    
    print(f"Test Case: Z={Z}, {ni}s -> {nf}{'s' if lf==0 else 'p'} @ {E_inc_eV} eV")

    # --- 2. GRID SETUP ---
    r_max = 100.0
    n_points = 3000
    grid = make_r_grid(r_min=1e-4, r_max=r_max, n_points=n_points)
    print(f"Grid: {n_points} points, r_min={grid.r[0]:.2e}, r_max={grid.r[-1]:.1f}")
    
    # --- 3. BOUND STATES ---
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core_params)
    
    # Initial State
    states_i = solve_bound_states(grid, V_core, l=li, n_states_max=ni+1)
    # n_index = n - l
    n_idx_i_val = ni - li
    orb_i = [s for s in states_i if s.n_index == n_idx_i_val][0]
    
    # Final State
    states_f = solve_bound_states(grid, V_core, l=lf, n_states_max=nf+1)
    n_idx_f_val = nf - lf
    orb_f = [s for s in states_f if s.n_index == n_idx_f_val][0]
    
    print("\n--- Bound States ---")
    print(f"Initial ({ni}l{li}): E={orb_i.energy_au:.6f} Ha (Expected: {-0.5/ni**2:.6f})")
    print(f"Final   ({nf}l{lf}): E={orb_f.energy_au:.6f} Ha (Expected: {-0.5/nf**2:.6f})")
    
    # Check Normalization
    norm_i = integrate_trapz(np.abs(orb_i.u_of_r)**2, grid)
    norm_f = integrate_trapz(np.abs(orb_f.u_of_r)**2, grid)
    print(f"Norm i: {norm_i:.6f}")
    print(f"Norm f: {norm_f:.6f}")
    
    if abs(norm_i - 1.0) > 1e-4 or abs(norm_f - 1.0) > 1e-4:
        print("WARNING: Bound state normalization error!")

    # Plot Bound States
    plt.figure(figsize=(10, 6))
    plt.plot(grid.r, orb_i.u_of_r, label=f'Initial n={ni} l={li}')
    plt.plot(grid.r, orb_f.u_of_r, label=f'Final n={nf} l={lf}')
    plt.xlim(0, 20)
    plt.title("Bound State Wavefunctions u(r)")
    plt.xlabel("r (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_bound_states.png")
    print("Saved debug_bound_states.png")

    # --- 4. DISTORTING POTENTIALS ---
    dE = orb_f.energy_au - orb_i.energy_au # Excitation energy (negative if excitation? No, E_f > E_i usually implies absorption)
    # Wait, Bound energy: E_1s = -0.5, E_2s = -0.125. dE = -0.125 - (-0.5) = +0.375
    E_final_eV = E_inc_eV - (dE * 27.211)
    
    k_i = k_from_E_eV(E_inc_eV)
    k_f = k_from_E_eV(E_final_eV)
    
    print(f"\n--- Kinematics ---")
    print(f"E_inc = {E_inc_eV:.2f} eV, k_i = {k_i:.4f} a.u.")
    print(f"E_fin = {E_final_eV:.2f} eV, k_f = {k_f:.4f} a.u.")
    
    # Build Potentials
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_i, orb_f, k_i, k_f, use_exchange=False)
    
    print("\n--- Distorting Potentials ---")
    print(f"U_i(r_min) = {U_i.U_of_r[0]:.2f}, U_i(r_max) = {U_i.U_of_r[-1]:.2e}")
    print(f"U_f(r_min) = {U_f.U_of_r[0]:.2f}, U_f(r_max) = {U_f.U_of_r[-1]:.2e}")
    
    # Plot Potentials
    plt.figure(figsize=(10, 6))
    plt.plot(grid.r, U_i.U_of_r, label='U_i (Entrance)')
    plt.plot(grid.r, U_f.U_of_r, label='U_f (Exit)')
    plt.plot(grid.r, V_core, '--', color='gray', label='V_core', alpha=0.5)
    plt.xlim(0, 10) # Zoom in near core
    plt.ylim(-Z*2, 0.5)
    plt.title("Distorting Potentials U(r)")
    plt.xlabel("r (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_potentials.png")
    print("Saved debug_potentials.png")
    
    # --- 5. CONTINUUM WAVES ---
    print("\n--- Continuum Waves Check ---")
    l_test = 0
    chi_i = solve_continuum_wave(grid, U_i, l=l_test, E_eV=E_inc_eV)
    chi_f = solve_continuum_wave(grid, U_f, l=l_test, E_eV=E_final_eV)
    
    print(f"L={l_test}")
    print(f"Chi_i Phase Shift: {chi_i.phase_shift:.4f} rad")
    print(f"Chi_f Phase Shift: {chi_f.phase_shift:.4f} rad")
    
    # Check Asymptotic Amplitude
    # Should be ~1.0
    tail_amp_i = np.max(np.abs(chi_i.chi_of_r[-500:]))
    tail_amp_f = np.max(np.abs(chi_f.chi_of_r[-500:]))
    print(f"Asymp Amp i: ~{tail_amp_i:.4f}")
    print(f"Asymp Amp f: ~{tail_amp_f:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(grid.r, chi_i.chi_of_r, label=f'Chi_i L={l_test}')
    plt.plot(grid.r, chi_f.chi_of_r, label=f'Chi_f L={l_test}')
    plt.xlim(0, 50)
    plt.title(f"Continuum Waves L={l_test} (normalized to unit amp)")
    plt.xlabel("r (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_continuum.png")
    print("Saved debug_continuum.png")

    # --- 6. RADIAL INTEGRALS ---
    print("\n--- Radial Integrals (First few partial waves) ---")
    
    L_Range = 5
    integrals_log = []
    
    for l_p in range(L_Range):
        try:
             c_i = solve_continuum_wave(grid, U_i, l=l_p, E_eV=E_inc_eV)
             # Let's assume l_f = l_i for simplicity in this quick scan, or adjacent
             # For s->s, monopole term L_T=0 connects l -> l.
             c_f = solve_continuum_wave(grid, U_f, l=l_p, E_eV=E_final_eV)
             
             ints = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_i, orb_f, c_i, c_f, L_max=2)
             
             val_dir = ints.I_L_direct.get(0, 0.0) # L_T=0
             val_exc = ints.I_L_exchange.get(0, 0.0)
             
             print(f"l_i={l_p}, l_f={l_p} | R_dir(L=0)={val_dir:.4e}, R_exc(L=0)={val_exc:.4e}")
             
             if abs(val_dir) > 100.0:
                 print("  >>> WARNING: Huge integral value!")
                 
        except Exception as e:
            print(f"l={l_p} failed: {e}")

    # --- 7. PARTIAL WAVE SUMMATION (1s -> 2p) ---
    print("\n--- Full Cross Section Summation (1s -> 2p) ---")
    
    # Grid for angular integration
    theta_grid = np.linspace(0.0, np.pi, 200)
    
    # Storage for amplitudes
    # Key: (Mi, Mf) -> Amplitudes(f, g)
    from dwba_coupling import Amplitudes
    
    # H 1s -> 2p: Li=0, Lf=1. 
    # Mi=0. Mf can be -1, 0, 1.
    total_amplitudes = {}
    Li, Lf = li, lf
    
    for Mi in range(-Li, Li+1):
        for Mf in range(-Lf, Lf+1):
             total_amplitudes[(Mi, Mf)] = Amplitudes(
                np.zeros_like(theta_grid, dtype=complex),
                np.zeros_like(theta_grid, dtype=complex)
            )

    L_max_proj = 20
    print(f"Summing partial waves up to l_i={L_max_proj} for 1s->2p...")
    
    for l_i in range(L_max_proj + 1):
        # Solve chi_i
        chi_i = solve_continuum_wave(grid, U_i, l=l_i, E_eV=E_inc_eV)
        
        # Parity: Li=0, Lf=1. Target Parity Change = Odd.
        # Projectile: l_i -> l_f must change parity.
        # So l_f = l_i +/- 1, etc.
        parity_target = (Li + Lf) % 2 # 1
        
        lf_min = max(0, l_i - 5)
        lf_max = l_i + 5
        
        for l_f in range(lf_min, lf_max + 1):
            if (l_i + l_f) % 2 != parity_target:
                continue
                
            chi_f = solve_continuum_wave(grid, U_f, l=l_f, E_eV=E_final_eV)
            
            # Integrals
            ints = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f, L_max=8)
            
            # Add to amplitudes
            for Mi in range(-Li, Li+1):
                for Mf in range(-Lf, Lf+1):
                    amps = calculate_amplitude_contribution(
                        theta_grid,
                        ints.I_L_direct,
                        ints.I_L_exchange,
                        l_i, l_f, k_i, k_f,
                        Li, Lf, Mi, Mf
                    )
                    tgt = total_amplitudes[(Mi, Mf)]
                    tgt.f_theta += amps.f_theta
                    tgt.g_theta += amps.g_theta

        if l_i % 5 == 0:
            print(f"  l_i={l_i} done.")


    
    # Expected for H 1s-2s at 50eV is approx 1.0 * pi * a0^2 ???
    # Actually checking literature: Callaway et al gives around 0.1 - 0.2 pi a0^2.
    # So 0.6 a.u. area?
    
    # Integration
    total_dcs = np.zeros_like(theta_grid, dtype=float)
    prefac_kinematics = (k_f / k_i)
    
    spin_singlet = 0.25
    spin_triplet = 0.75
    
    for (Mi, Mf), amps in total_amplitudes.items():
        f = amps.f_theta
        g = amps.g_theta
        
        term_singlet = np.abs(f + g)**2
        term_triplet = np.abs(f - g)**2
        
        dcs_channel = spin_singlet * term_singlet + spin_triplet * term_triplet
        total_dcs += dcs_channel
    
    total_dcs *= prefac_kinematics
    # Average over initial states (2Li+1)
    total_dcs *= (1.0 / (2*Li + 1))
    
    # Manual trapz:
    dcs_sin = total_dcs * np.sin(theta_grid)
    sigma_au = 2 * np.pi * np.trapz(dcs_sin, theta_grid)
    
    from grid import HARTREE_TO_EV
    # 1 a.u. length = 5.29177e-11 m = 5.29e-9 cm
    # 1 a.u. area = 2.80028e-17 cm^2
    au2cm2 = 2.80028e-17
    
    sigma_cm2 = sigma_au * au2cm2
    
    print(f"\n--- Final Results ---")
    print(f"Sigma (a.u.): {sigma_au:.4e}")
    print(f"Sigma (cm^2): {sigma_cm2:.4e}")
    
    print(f"DONE.")

if __name__ == "__main__":
    main()
