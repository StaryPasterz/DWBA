#!/usr/bin/env python
"""Debug script to investigate phase instability for high L."""
import numpy as np
import sys
sys.path.insert(0, '.')

from grid import make_r_grid, k_from_E_eV
from continuum import (
    solve_continuum_wave, _find_match_point,
    _extract_phase_logderiv_neutral, _extract_phase_logderiv_coulomb,
    _derivative_5point
)
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials

# Setup
grid = make_r_grid(1e-5, 200.0, 3000)
r = grid.r

core = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V = V_core_on_grid(grid, core)
states = solve_bound_states(grid, V, l=0, n_states_max=2)
orb = states[0]

k_i = k_from_E_eV(100.0)  # 100 eV
U_i, _ = build_distorting_potentials(grid, V, orb, orb, k_i, k_i, use_exchange=False)

print("=" * 60)
print("INVESTIGATION: Phase instability for high L")
print("=" * 60)

for L in [10, 20, 30, 38, 40]:
    print(f"\n--- L = {L} ---")
    
    r_turn = np.sqrt(L * (L + 1)) / k_i
    r_turn_safe = 2.5 * r_turn
    
    print(f"  r_turn = {r_turn:.2f}, 2×r_turn = {r_turn_safe:.2f}")
    
    # Get match point
    idx_match, r_m = _find_match_point(r, U_i.U_of_r, k_i, L, threshold=1e-2)
    
    print(f"  Match point: idx={idx_match}, r_m={r_m:.2f}")
    print(f"  r_m > 2×r_turn? {r_m > r_turn_safe}")
    
    # Check alt point (used in diagnostic)
    idx_alt = idx_match - 5
    r_alt = r[idx_alt]
    
    print(f"  Alt point: idx={idx_alt}, r_alt={r_alt:.2f}")
    print(f"  r_alt > 2×r_turn? {r_alt > r_turn_safe}")
    
    # The problem: if r_alt < 2×r_turn, we're comparing phase in two different regimes!
    if r_alt < r_turn_safe:
        print(f"  ⚠️ ALT POINT IS BEFORE 2×r_turn! This causes phase comparison to fail!")
    
    # Also check wavelength sampling
    dr_local = r[idx_match] - r[idx_match-1]
    wavelength = 2 * np.pi / k_i
    pts_per_wavelength = wavelength / dr_local
    
    print(f"  Local dr = {dr_local:.4f}, wavelength = {wavelength:.2f}")
    print(f"  Points per wavelength at r_m: {pts_per_wavelength:.1f}")
    
    if pts_per_wavelength < 10:
        print(f"  ⚠️ UNDERSAMPLED: only {pts_per_wavelength:.1f} pts/wavelength!")
