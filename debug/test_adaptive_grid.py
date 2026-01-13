"""
Diagnostic for adaptive grid behavior.
Tests what parameters are calculated for different energies.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from grid import k_from_E_eV
from DW_main import calculate_optimal_grid_params

# Parameters from H2s.yaml
base_r_max = 200.0
base_n_points = 3000
scale_factor = 2.5
n_points_max = 15000
min_pts_per_wl = 15
L_max_proj = 5  # From H2s.yaml

print("=" * 70)
print(" ADAPTIVE GRID DIAGNOSTIC")
print("=" * 70)
print()
print(f"Base config: r_max={base_r_max}, n_points={base_n_points}")
print(f"L_max_projectile: {L_max_proj}")
print(f"scale_factor: {scale_factor}, n_points_max: {n_points_max}")
print(f"min_points_per_wavelength: {min_pts_per_wl}")
print()
print(f"{'E(eV)':<10} | {'k(a.u.)':<10} | {'r_needed':<12} | {'r_max':<10} | {'n_points':<10} | Notes")
print("-" * 80)

for E in [10.2, 10.7, 15, 20, 50, 100, 200, 300, 500, 1000]:
    k = k_from_E_eV(E)
    r_needed = scale_factor * (L_max_proj + 0.5) / k
    r_max, n_pts = calculate_optimal_grid_params(
        E, L_max_proj, base_r_max, base_n_points, 
        scale_factor, n_points_max, min_pts_per_wl
    )
    
    wavelength = 2 * np.pi / k if k > 0.01 else float('inf')
    
    notes = []
    if r_max > base_r_max:
        notes.append(f"r_max↑")
    if n_pts > base_n_points:
        notes.append(f"n_pts↑ (λ={wavelength:.1f})")
    
    print(f"{E:<10.1f} | {k:<10.4f} | {r_needed:<12.1f} | {r_max:<10.1f} | {n_pts:<10d} | {' '.join(notes)}")

print()
print("=" * 70)
print(" KEY FINDINGS")
print("=" * 70)
print("""
1. r_needed = scale_factor * (L+0.5) / k
   - For LOW energy: k is small, so r_needed is LARGER
   - For HIGH energy: k is large, so r_needed is SMALLER
   - r_max = max(base_r_max, r_needed) → only increases if r_needed > base

2. n_points scaling happens for HIGH energies (wavelength-based):
   - At high k, wavelength is short → need more points per unit length
   - n_wavelength_required = r_check * log_ratio / dr_needed

3. If n_points changes between energies but grid/prep isn't regenerated,
   you get index-out-of-bounds errors!
""")
