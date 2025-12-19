"""
Diagnostic script to verify bound state norms and radial integrals for 1s->2s vs 1s->2p.
"""

import numpy as np
from grid import make_r_grid, integrate_trapz
from potential_core import V_core_on_grid, CorePotentialParams
from bound_states import solve_bound_states
from driver import ExcitationChannelSpec

print("=" * 60)
print("Bound State Diagnostics for H(1s), H(2s), H(2p)")
print("=" * 60)

# Hydrogen parameters (Zc=1, a1...a6 = 0 for pure Coulomb)
params = CorePotentialParams(Zc=1, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

# Create grid
grid = make_r_grid(r_min=1e-5, r_max=200.0, n_points=3000)
V_core = V_core_on_grid(grid, params)

print(f"\nGrid: {len(grid.r)} points, r = [{grid.r[0]:.2e}, {grid.r[-1]:.1f}] a.u.")
print(f"V_core at r=1: {V_core[np.argmin(np.abs(grid.r - 1.0))]:.4f} Ha (expected: -1.0)")

# Solve for s-states (l=0)
print("\n--- l=0 (s-states) ---")
states_l0 = solve_bound_states(grid, V_core, l=0, n_states_max=3)
for i, st in enumerate(states_l0):
    norm = integrate_trapz(st.u_of_r**2, grid)
    print(f"  n={st.n_index}, E = {st.energy_au:.6f} Ha, norm = {norm:.6f}")
    # Expected: E_1s = -0.5, E_2s = -0.125

# Solve for p-states (l=1)
print("\n--- l=1 (p-states) ---")
states_l1 = solve_bound_states(grid, V_core, l=1, n_states_max=3)
for i, st in enumerate(states_l1):
    norm = integrate_trapz(st.u_of_r**2, grid)
    print(f"  n={st.n_index}, E = {st.energy_au:.6f} Ha, norm = {norm:.6f}")
    # Expected: E_2p = -0.125

# Compare radial wavefunctions at specific points
print("\n--- Radial Wavefunction u(r) comparison at r=2 a.u. ---")
idx_r2 = np.argmin(np.abs(grid.r - 2.0))
r_val = grid.r[idx_r2]

u_1s = states_l0[0].u_of_r[idx_r2]
u_2s = states_l0[1].u_of_r[idx_r2] if len(states_l0) > 1 else 0
u_2p = states_l1[0].u_of_r[idx_r2] if len(states_l1) > 0 else 0

print(f"  r = {r_val:.2f} a.u.")
print(f"  u_1s(r) = {u_1s:.6f}")
print(f"  u_2s(r) = {u_2s:.6f}")
print(f"  u_2p(r) = {u_2p:.6f}")

# Compare overlap <1s|2s> and <1s|2p> (should be 0 for orthogonal states... but different l are auto-orthogonal)
overlap_1s_2s = integrate_trapz(states_l0[0].u_of_r * states_l0[1].u_of_r, grid) if len(states_l0) > 1 else 0
print(f"\n--- Orthogonality check ---")
print(f"  <1s|2s> (radial only) = {overlap_1s_2s:.6e} (should be ~0)")

# For 1s->2p, only l_T=1 contributes, so check the key radial integral form
# I_dir = Int Int (chi_f chi_i)(r1) * (r_</r_>^2) * (u_f u_i)(r2) dr1 dr2
# Simplified check: just compare the target density |u_i u_f|^2 norms

print("\n--- Target Density Integrals ---")
u_1s_arr = states_l0[0].u_of_r
u_2s_arr = states_l0[1].u_of_r if len(states_l0) > 1 else np.zeros_like(u_1s_arr)
u_2p_arr = states_l1[0].u_of_r if len(states_l1) > 0 else np.zeros_like(u_1s_arr)

rho_1s_2s = u_1s_arr * u_2s_arr
rho_1s_2p = u_1s_arr * u_2p_arr

int_rho_2s = np.sum(np.abs(rho_1s_2s) * grid.w_trapz)
int_rho_2p = np.sum(np.abs(rho_1s_2p) * grid.w_trapz)

print(f"  Int |u_1s * u_2s| dr = {int_rho_2s:.6f}")
print(f"  Int |u_1s * u_2p| dr = {int_rho_2p:.6f}")
print(f"  Ratio = {int_rho_2p / int_rho_2s:.3f}")

print("\n" + "=" * 60)
