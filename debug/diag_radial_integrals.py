#!/usr/bin/env python
"""
Diagnostic: Investigate radial matrix elements for L0 vs L1 anomaly
Check if I_L integrals have unexpected behavior at anomaly energies
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from dwba_matrix_elements import radial_ME_all_L

# Focus on anomaly region
TEST_ENERGIES = [11.0, 12.65, 14.0, 22.0, 50.0]  

print("=" * 80)
print("RADIAL MATRIX ELEMENT DIAGNOSTIC")
print("=" * 80)

# Setup
grid = make_r_grid(r_max=200.0, n_points=6000)
core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V_core = V_core_on_grid(grid, core_params)

# Get 1s and 2s states
states_s = solve_bound_states(grid, V_core, l=0, n_states_max=10)
# Hydrogen energies are -0.5/n^2
# Find state closest to -0.5
orb_1s = min(states_s, key=lambda s: abs(s.energy_au + 0.5))
# Find state closest to -0.125
orb_2s = min(states_s, key=lambda s: abs(s.energy_au + 0.125))

E_1s = orb_1s.energy_au
E_2s = orb_2s.energy_au
threshold_eV = (E_2s - E_1s) * 27.211386
print(f"Target States:")
print(f"  1s: E={E_1s:.6f} Ha (expected -0.500000)")
print(f"  2s: E={E_2s:.6f} Ha (expected -0.125000)")
print(f"Threshold: {threshold_eV:.2f} eV")
print(f"Grid: {len(grid.r)} points\n")

for E_inc in TEST_ENERGIES:
    if E_inc <= threshold_eV:
        print(f"E={E_inc} eV: BELOW THRESHOLD\n")
        continue
    
    # Kinematics
    dE_au = E_2s - E_1s
    E_inc_au = E_inc / 27.211386
    E_f_au = E_inc_au - dE_au
    E_f_eV = E_f_au * 27.211386
    
    k_i = k_from_E_eV(E_inc)
    k_f = k_from_E_eV(E_f_eV)
    
    # Build potentials
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
    
    print(f"E = {E_inc:.2f} eV (k_i={k_i:.3f}, k_f={k_f:.3f})")
    print("-" * 100)
    print(f"{'l_i':>2} {'l_f':>2} | {'Direct I0':>10} | {'Exch I0':>10} | {'Phase_i':>8} | {'Phase_f':>8} | {'Sum(|ID|)':>10} | {'Sum(|IE|)':>10}")
    print("-" * 100)
    
    for l_i in [0, 1, 2]:
        for l_f in [0, 1, 2]:
            try:
                wave_i = solve_continuum_wave(grid, U_i, l_i, E_inc, z_ion=0.0)
                wave_f = solve_continuum_wave(grid, U_f, l_f, E_f_eV, z_ion=0.0)
                
                # Get radial integrals
                result = radial_ME_all_L(
                    grid, V_core, U_i.U_of_r, 
                    orb_1s, orb_2s, wave_i, wave_f, 
                    L_max=5
                )
                
                ID = result.I_L_direct
                IE = result.I_L_exchange
                
                I0_D = ID.get(0, 0.0)
                I0_E = IE.get(0, 0.0)
                
                phi_i = wave_i.phase_shift
                phi_f = wave_f.phase_shift
                
                sum_D = sum(abs(v) for v in ID.values())
                sum_E = sum(abs(v) for v in IE.values())
                
                print(f"{l_i:>2} {l_f:>2} | {I0_D:>10.3e} | {I0_E:>10.3e} | {phi_i:>8.4f} | {phi_f:>8.4f} | {sum_D:>10.3e} | {sum_E:>10.3e}")
                
            except Exception as e:
                print(f"{l_i:>2} {l_f:>2} | ERROR: {str(e)[:40]}")
    
    print()

print("=" * 80)
print("KEY INSIGHT: For s→s transition, l_i=0,l_f=0 should dominate.")
print("If l_i=0,l_f=1 or l_i=1,l_f=0 has larger I_L, that's the anomaly source.")
print("=" * 80)
