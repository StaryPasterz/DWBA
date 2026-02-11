"""Check GPU VRAM and compute matrix sizes for pilot calibration."""
import cupy as cp

free, total = cp.cuda.Device().mem_info
props = cp.cuda.runtime.getDeviceProperties(0)
name = props['name'].decode()
print(f"GPU: {name}")
print(f"Total VRAM: {total / 1e9:.2f} GB")
print(f"Free VRAM:  {free / 1e9:.2f} GB")

pool = cp.get_default_memory_pool()
print(f"Pool total: {pool.total_bytes() / 1e9:.2f} GB")
print(f"Pool used:  {pool.used_bytes() / 1e9:.2f} GB")

# Matrix sizes for pilot: n_points=5735
n = 5735
print(f"\nFor n_points={n}:")
print(f"  Standard matrix ({n}x{n}): {n*n*8/1e9:.2f} GB (x2 for inv_gtr + log_ratio = {2*n*n*8/1e9:.2f} GB)")
print(f"  Filon matrix ({n}x{n}):    {n*n*8/1e9:.2f} GB (x2 = {2*n*n*8/1e9:.2f} GB)")
print(f"  Build temps (ratio):       {n*n*8/1e9:.2f} GB")
print(f"  Ratio cache (if enabled):  {n*n*8/1e9:.2f} GB")
print(f"  Working kernel copy:       {n*n*8/1e9:.2f} GB")
print(f"  TOTAL for full Filon (+recursive): {7*n*n*8/1e9:.2f} GB")
print(f"  TOTAL for minimal (standard only): {3*n*n*8/1e9:.2f} GB")
