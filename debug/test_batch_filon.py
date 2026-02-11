"""Final validation: batch vs per-element using SAME parameters (including dynamic node count)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
from oscillatory_integrals import (
    _filon_oscillatory_integral_multipole_batch,
    _filon_segment_complex,
    _FILON_MAX_SEGMENTS, _FILON_MAX_EFFECTIVE_DPHI,
)

L_values = np.arange(1, 9, dtype=int)
moments = np.array([1.2, 0.05, 0.003, 0.0002, 1e-5, 1e-7, 1e-9, 1e-11])
omega = 17.1
phase_offset = 0.5
r_start = 80.0
r_end = 480.0
delta_phi = np.pi / 4

# Compute effective n_nodes (same as batch function)
dr = delta_phi / abs(omega)
n_seg_raw = max(1, int(np.ceil((r_end - r_start) / dr)))
n_seg = min(n_seg_raw, _FILON_MAX_SEGMENTS)
eff_dphi = abs(omega) * (r_end - r_start) / max(1, n_seg)
n_nodes_eff = 5
if eff_dphi > _FILON_MAX_EFFECTIVE_DPHI:
    scale = int(np.ceil(eff_dphi / _FILON_MAX_EFFECTIVE_DPHI))
    n_nodes_eff = min(15, max(5, 5 + 2 * (scale - 1)))

print(f"n_segments={n_seg}, n_nodes_eff={n_nodes_eff}, eff_dphi={eff_dphi:.4f}, max_dphi={_FILON_MAX_EFFECTIVE_DPHI:.4f}")

# Batch
t0 = time.perf_counter()
result_batch = _filon_oscillatory_integral_multipole_batch(
    moments, L_values, omega, phase_offset, r_start, r_end,
    delta_phi=delta_phi, n_nodes_per_segment=5
)
t_batch = time.perf_counter() - t0

# Reference with MATCHING n_nodes_eff
bounds = np.linspace(r_start, r_end, n_seg + 1)
result_ref = np.zeros(len(L_values), dtype=np.complex128)
t2 = time.perf_counter()
for i_seg in range(n_seg):
    a, b = bounds[i_seg], bounds[i_seg + 1]
    r_nodes = np.linspace(a, b, n_nodes_eff)
    inv_r = 1.0 / np.maximum(r_nodes, 1e-30)
    pow_mat = inv_r[None, :] ** (L_values[:, None] + 1.0)
    f_mat = moments[:, None] * pow_mat
    for i in range(len(L_values)):
        result_ref[i] += _filon_segment_complex(f_mat[i], r_nodes, omega, phase_offset)
t_ref = time.perf_counter() - t2

print(f"Batch: {t_batch:.3f}s | Reference: {t_ref:.3f}s | Speedup: {t_ref/max(1e-9,t_batch):.1f}x")
print()
print(f"{'L':>2} | {'Batch real':>15} | {'Ref real':>15} | {'Abs diff':>10} | {'Rel diff':>10}")
print("-" * 75)
for i, L in enumerate(L_values):
    rb = result_batch[i].real
    rr = result_ref[i].real
    ad = abs(rb - rr)
    denom = max(abs(rr), 1e-30)
    rd = ad / denom
    print(f"{L:2d} | {rb:15.8e} | {rr:15.8e} | {ad:10.2e} | {rd:10.2e}")

max_rel = np.max(np.abs(result_batch.real - result_ref.real) / (np.abs(result_ref.real) + 1e-30))
print(f"\nMax relative difference: {max_rel:.2e}")
if max_rel < 1e-10:
    print("PASS")
else:
    print(f"WARN: max_rel={max_rel:.2e}")
