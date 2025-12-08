
import numpy as np
from grid import make_r_grid
from potential_core import V_core_on_grid, CorePotentialParams
from bound_states import solve_bound_states

def check_bound_states():
    # H atom: Zc=1, all alphas=0
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    grid = make_r_grid(r_min=1e-5, r_max=200.0, n_points=4000)
    V = V_core_on_grid(grid, params)
    
    print("Solving H Bound States (l=0)...")
    states = solve_bound_states(grid, V, l=0, n_states_max=3)
    
    for s in states:
        n = s.n_index
        E = s.energy_au
        E_exact = -0.5 / (n**2)
        err = abs(E - E_exact)
        print(f"n={n} l=0 | E_calc={E:.6f} | E_exact={E_exact:.6f} | Err={err:.2e}")

if __name__ == "__main__":
    check_bound_states()
