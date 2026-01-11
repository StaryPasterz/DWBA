"""Compare 'legacy' vs 'advanced' oscillatory methods for L=5 upturn analysis."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from dwba_matrix_elements import radial_ME_all_L

E_INC_EV = 69.26
print(f"COMPARISON: Legacy vs Advanced at {E_INC_EV} eV")
print("=" * 80)

grid = make_r_grid(r_max=200.0, n_points=12000)
core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V_core = V_core_on_grid(grid, core_params)

states_s = solve_bound_states(grid, V_core, l=0, n_states_max=5)
orb_1s = min(states_s, key=lambda s: abs(s.energy_au + 0.5))
orb_2s = min(states_s, key=lambda s: abs(s.energy_au + 0.125))

E_f = E_INC_EV - (orb_2s.energy_au - orb_1s.energy_au)*27.211
ki = k_from_E_eV(E_INC_EV)
kf = k_from_E_eV(E_f)

U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, ki, kf, use_exchange=False)

print(f"{'L':>3} | {'Legacy Sum':>12} | {'Adv Sum':>12} | {'Ratio':>8} | {'Notes'}")
print("-" * 60)

for L in [3, 4, 5, 6, 7]:
    chi_i = solve_continuum_wave(grid, U_i, L, E_INC_EV, z_ion=0.0)
    chi_f = solve_continuum_wave(grid, U_f, L, E_f, z_ion=0.0)
    
    # Legacy method
    result_legacy = radial_ME_all_L(
        grid, V_core, U_i.U_of_r, orb_1s, orb_2s, chi_i, chi_f, 
        L_max=3, oscillatory_method="legacy"
    )
    sum_legacy = sum(abs(v) for v in result_legacy.I_L_direct.values())
    
    # Advanced method  
    result_adv = radial_ME_all_L(
        grid, V_core, U_i.U_of_r, orb_1s, orb_2s, chi_i, chi_f,
        L_max=3, oscillatory_method="advanced"
    )
    sum_adv = sum(abs(v) for v in result_adv.I_L_direct.values())
    
    ratio = sum_adv / sum_legacy if sum_legacy > 1e-15 else float('inf')
    notes = "DIFFERS!" if abs(ratio - 1.0) > 0.2 else ""
    
    print(f"{L:>3} | {sum_legacy:>12.4e} | {sum_adv:>12.4e} | {ratio:>8.3f} | {notes}")

print("=" * 80)
