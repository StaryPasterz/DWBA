#!/usr/bin/env python
"""Quick debug script to trace match point logic."""
import numpy as np
import sys
sys.path.insert(0, '.')

from grid import make_r_grid
from continuum import _find_match_point

grid = make_r_grid(1e-5, 200.0, 3000)
r = grid.r
U_arr = -1.0/r  # Simple Coulomb potential

k = 1.0
l = 5
k2 = k**2
threshold = 1e-2
N = len(r)

r_turn = np.sqrt(l * (l + 1)) / k
r_turn_safe = 2.0 * r_turn

print(f"L={l}, k={k}")
print(f"r_turn = {r_turn:.4f}")
print(f"2 * r_turn = {r_turn_safe:.4f}")

# Replicate the function logic
MIN_MARGIN = 50
search_start = max(MIN_MARGIN, N // 4)
idx_turn = np.searchsorted(r, r_turn)
search_start = max(search_start, idx_turn + 10)
search_end = int(0.9 * N)

print(f"search_start idx = {search_start}, r = {r[search_start]:.4f}")
print(f"search_end idx = {search_end}, r = {r[search_end]:.4f}")

# Search with new logic
found = False
for idx in range(search_start, search_end):
    ri = r[idx]
    Ui = U_arr[idx]
    
    # Stage 1: centrifugal safety
    if ri < r_turn_safe:
        continue
    
    # Stage 2: potential criterion
    if abs(2.0 * Ui) < threshold * k2:
        print(f"Found ideal point at idx={idx}, r={ri:.4f}, |2U|={abs(2*Ui):.6f}")
        found = True
        break

if not found:
    print("No ideal point found - fallback used")
    fallback_idx = max(search_start, int(0.7 * N))
    print(f"Fallback idx = {fallback_idx}, r = {r[fallback_idx]:.4f}")

# Now call actual function
idx, r_m = _find_match_point(r, U_arr, k, l, threshold=threshold)
print(f"\nActual function result: idx={idx}, r_m={r_m:.4f}")
print(f"Is r_m > 2*r_turn? {r_m > r_turn_safe}")
