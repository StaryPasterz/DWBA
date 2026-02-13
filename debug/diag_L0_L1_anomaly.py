#!/usr/bin/env python
"""
Deep Diagnostic: Investigate L0 vs L1 anomaly at critical energies
Specifically checks why L1 > L0 at 12.65 eV for s->s transition
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials

# Test energies - focus on anomaly region
TEST_ENERGIES = [11.0, 12.0, 12.65, 13.0, 14.0, 17.0, 22.0, 50.0]

def run_diagnostic():
    print("=" * 80)
    print("L0 vs L1 ANOMALY DIAGNOSTIC")
    print("=" * 80)
    
    # Create high-res grid
    n_points = 10000
    r_max = 200.0
    grid = make_r_grid(r_max, n_points)
    print(f"Grid: {n_points} points, r_max = {r_max}")
    
    # H potential
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core_params)
    
    # Get 1s and 2s states
    states_1s = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    orb_1s = [s for s in states_1s if s.n_index == 1][0]
    orb_2s = [s for s in states_1s if s.n_index == 2][0]
    
    E_1s = orb_1s.energy_au
    E_2s = orb_2s.energy_au
    threshold_eV = (E_2s - E_1s) * 27.211386
    print(f"Threshold: {threshold_eV:.2f} eV")
    
    print("\n" + "-" * 80)
    print(f"{'E(eV)':<8} | {'k_i':<8} | {'k_f':<8} | {'L':<3} | {'delta_l':<12} | {'chi_max':<10} | Notes")
    print("-" * 80)
    
    for E_inc in TEST_ENERGIES:
        if E_inc <= threshold_eV:
            print(f"{E_inc:<8.2f} | BELOW THRESHOLD")
            continue
        
        # Kinematics
        dE_au = E_2s - E_1s
        E_inc_au = E_inc / 27.211386
        E_f_au = E_inc_au - dE_au
        
        k_i = np.sqrt(2.0 * E_inc_au)
        k_f = np.sqrt(2.0 * E_f_au)
        
        # Build potentials for this energy
        U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
        
        for L in [0, 1, 2, 3]:
            try:
                wave_i = solve_continuum_wave(grid, U_i, L, E_inc, z_ion=0.0)
                wave_f = solve_continuum_wave(grid, U_f, L, E_f_au * 27.211386, z_ion=0.0)
                
                delta_i = wave_i.phase_shift
                delta_f = wave_f.phase_shift
                chi_i_max = np.max(np.abs(wave_i.chi_of_r))
                chi_f_max = np.max(np.abs(wave_f.chi_of_r))
                
                notes = ""
                if chi_i_max > 2.0:
                    notes += "LARGE_CHI "
                if abs(delta_i) > np.pi/2:
                    notes += "BIG_PHASE "
                if chi_i_max < 0.5:
                    notes += "SMALL_CHI "
                
                print(f"{E_inc:<8.2f} | {k_i:<8.3f} | {k_f:<8.3f} | {L:<3} | {delta_i:<12.6f} | {chi_i_max:<10.4f} | {notes}")
                
            except Exception as e:
                print(f"{E_inc:<8.2f} | {k_i:<8.3f} | {k_f:<8.3f} | {L:<3} | ERROR: {str(e)[:30]}")
    
    print("\n" + "=" * 80)
    print("CHECKING BOUND STATE OVERLAPS AT ANOMALY ENERGY")
    print("=" * 80)
    
    # Check at E=12.65 eV
    E_test = 12.65
    dE_au = E_2s - E_1s
    E_inc_au = E_test / 27.211386
    E_f_au = E_inc_au - dE_au
    
    k_i = np.sqrt(2.0 * E_inc_au)
    k_f = np.sqrt(2.0 * E_f_au)
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
    
    # Calculate radial overlaps
    r = grid.r
    dr = np.diff(r, prepend=r[0])
    dr[0] = r[0]
    
    print(f"\nE = {E_test} eV, k_i = {k_i:.3f}, k_f = {k_f:.3f}")
    print(f"Wavelength lambda_i = {2*np.pi/k_i:.2f} a.u.")
    
    for L in [0, 1]:
        wave_i = solve_continuum_wave(grid, U_i, L, E_test, z_ion=0.0)
        
        # Overlap <1s|chi_L_i>
        integrand_1s = orb_1s.u_of_r * wave_i.chi_of_r
        overlap_1s = np.sum(integrand_1s * dr)
        
        # Overlap <2s|chi_L_i>
        integrand_2s = orb_2s.u_of_r * wave_i.chi_of_r
        overlap_2s = np.sum(integrand_2s * dr)
        
        print(f"  L={L}: <1s|chi_L> = {overlap_1s:.4f}, <2s|chi_L> = {overlap_2s:.4f}, delta = {wave_i.phase_shift:.4f}")

if __name__ == "__main__":
    run_diagnostic()
