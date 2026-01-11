import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import make_r_grid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

grid = make_r_grid(r_max=200.0, n_points=12000)
core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V_core = V_core_on_grid(grid, core_params)

states = solve_bound_states(grid, V_core, l=0, n_states_max=5)
orb_1s = min(states, key=lambda s: abs(s.energy_au + 0.5))
orb_2s = min(states, key=lambda s: abs(s.energy_au + 0.125))

overlap = np.sum(grid.w_simpson * orb_1s.u_of_r * orb_2s.u_of_r)
print(f"Overlap <1s|2s> (Simpson): {overlap:.2e}")

# Check if radial_ME_all_L uses w_simpson or w_trapz for L=0 correction
# (Viewing code shows it uses rho2_dir_w which is based on w_limited)
# w_limited comes from 'w' which is grid.w_simpson in radial_ME_all_L.

norm1s = np.sum(grid.w_simpson * orb_1s.u_of_r**2)
norm2s = np.sum(grid.w_simpson * orb_2s.u_of_r**2)
print(f"Norm <1s|1s>: {norm1s:.6f}")
print(f"Norm <2s|2s>: {norm2s:.6f}")
