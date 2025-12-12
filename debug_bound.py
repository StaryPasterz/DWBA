
import numpy as np
from grid import make_r_grid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

def test_bound():
    print("Testing Bound State Solver...")
    
    # 1. Grid
    grid = make_r_grid(r_min=0.05, r_max=100.0, n_points=2000)
    print(f"Grid: N={grid.r.size}, r_min={grid.r[0]:.4f}, r_max={grid.r[-1]:.1f}")
    
    # 2. Potential
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V = V_core_on_grid(grid, params)
    print(f"Potential at r_min: {V[0]:.4f} (Expected ~ -1/0.05 = -20)")
    
    # 3. Solve 1s (l=0)
    print("Solving l=0...")
    states = solve_bound_states(grid, V, l=0, n_states_max=2)
    print(f"Found {len(states)} states.")
    for s in states:
        print(f"  n_index={s.n_index} E={s.energy_au:.5f} au")
        
if __name__ == "__main__":
    test_bound()
