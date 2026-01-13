"""Test auto r_max implementation"""
import sys
sys.path.insert(0, '.')

from DW_main import calculate_optimal_grid_params

L = 5

print("r_max='auto':")
for E in [10, 50, 100, 500, 1000]:
    r, n = calculate_optimal_grid_params(E, L, 'auto', 3000)
    print(f"  E={E}eV: r_max={r:.1f}, n_pts={n}")

print()
print("r_max=200 (fixed):")
for E in [10, 50, 100, 500, 1000]:
    r, n = calculate_optimal_grid_params(E, L, 200, 3000)
    print(f"  E={E}eV: r_max={r:.1f}, n_pts={n}")
