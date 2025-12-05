from grid import make_r_grid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

def main():
    core_params = CorePotentialParams(
        Zc = 1.0,
        a1 = 0.0, a2 = 0.0,
        a3 = 0.0, a4 = 0.0,
        a5 = 0.0, a6 = 0.0,
    )

    grid = make_r_grid(r_min=0.1, r_max=50.0, n_points=200)

    V_core = V_core_on_grid(grid, core_params)

    states_s = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    states_p = solve_bound_states(grid, V_core, l=1, n_states_max=3)

    print("=== l=0 (s-like) ===")
    for orb in states_s:
        print(f"n_index={orb.n_index}, E={orb.energy_au} Hartree")

    print("=== l=1 (p-like) ===")
    for orb in states_p:
        print(f"n_index={orb.n_index}, E={orb.energy_au} Hartree")

if __name__ == "__main__":
    main()
