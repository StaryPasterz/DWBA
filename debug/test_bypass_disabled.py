#!/usr/bin/env python
"""
Quick Test: Compare partial wave behavior with Born bypass disabled
Run: python debug/test_bypass_disabled.py
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

# Focus on anomaly region
TEST_ENERGIES = [11.0, 12.0, 12.65, 13.0, 14.0, 15.0, 20.0]

print("=" * 70)
print("TEST: Born Bypass DISABLED")
print("=" * 70)

# Setup
grid = make_r_grid(200.0, 8000)
core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V_core = V_core_on_grid(grid, core_params)

# Get 1s state for distorting potential
states = solve_bound_states(grid, V_core, l=0, n_states_max=2)
orb_1s = [s for s in states if s.n_index == 1][0]

print(f"Grid: {len(grid.r)} points, r_max = {grid.r[-1]:.0f}")
print(f"\nE (eV)  | L | phase_shift  | chi_max | Notes")
print("-" * 70)

for E in TEST_ENERGIES:
    k = k_from_E_eV(E)
    # Build potential
    U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k, k, use_exchange=False)
    
    for L in [0, 1, 2]:
        try:
            wave = solve_continuum_wave(grid, U_i, L, E, z_ion=0.0)
            delta = wave.phase_shift
            chi_max = np.max(np.abs(wave.chi_of_r))
            
            notes = ""
            if chi_max > 1.5:
                notes += "LARGE "
            if abs(delta) > 1.5:
                notes += "BIG_PHASE "
                
            print(f"{E:>7.2f} | {L} | {delta:>12.6f} | {chi_max:>7.3f} | {notes}")
        except Exception as e:
            print(f"{E:>7.2f} | {L} | ERROR: {str(e)[:25]}")

print("\n" + "=" * 70)
print("If L0 phase_shift is near +/-1.57 (pi/2), that explains the anomaly")
print("=" * 70)
