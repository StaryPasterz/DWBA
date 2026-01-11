import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import make_r_grid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

grid = make_r_grid(100.0, 2000)
params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V = V_core_on_grid(grid, params)

print(f"V[0] at r={grid.r[0]:.4f} is {V[0]:.4f}")
print(f"V[-1] at r={grid.r[-1]:.4f} is {V[-1]:.4f}")

states = solve_bound_states(grid, V, l=0, n_states_max=5)
for s in states:
    print(f"n_index={s.n_index}, E={s.energy_au:.6f} Ha")
