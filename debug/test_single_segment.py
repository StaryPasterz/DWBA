"""Debug: compare _filon_3pt_batch vs _filon_segment_complex on a single triplet."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from oscillatory_integrals import _filon_3pt_batch, _filon_segment_complex

omega = 17.1
phase_offset = 0.5
a, b = 80.0, 80.098  # one segment

r_nodes = np.linspace(a, b, 3)
r0, r1, r2 = r_nodes

# Single scalar f_vals
f_vals = np.array([1.0 / r0**2, 1.0 / r1**2, 1.0 / r2**2])
f_mat = f_vals.reshape(1, 3)

ref = _filon_segment_complex(f_vals, r_nodes, omega, phase_offset)
batch = _filon_3pt_batch(f_mat, r0, r1, r2, omega, phase_offset)

print(f"Reference: {ref}")
print(f"Batch[0]:  {batch[0]}")
print(f"Diff:      {abs(ref - batch[0]):.2e}")
print(f"h = {b-a:.6f}")
print(f"theta = {omega * (b-a) / 2:.6f}")
print(f"omega*h = {omega * (b-a):.6f}")

# Now test on a full 5-node segment: compare recursive _filon_segment_complex vs my batch
print("\n=== Full 5-node segment ===")
a5, b5 = 80.0, 80.098
r5 = np.linspace(a5, b5, 5)
f5 = 1.0 / r5**2

ref5 = _filon_segment_complex(f5, r5, omega, phase_offset)

# My batch approach: split into triplets [0:3] and [2:5]
f5_mat = f5.reshape(1, 5)
batch5 = np.complex128(0)
for j in range(0, 3, 2):  # j=0, j=2
    batch5 += _filon_3pt_batch(
        f5_mat[:, j:j+3],
        r5[j], r5[j+1], r5[j+2],
        omega, phase_offset
    )[0]

print(f"Reference (recursive): {ref5}")
print(f"Batch (triplets):      {batch5}")
print(f"Diff: {abs(ref5 - batch5):.2e}")
