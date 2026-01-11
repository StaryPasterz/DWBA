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

# High energy where upturn was reported
E_INC_EV = 69.26
print(f"====================================================")
print(f"DIAGNOSTIC: Partial Wave Convergence at {E_INC_EV} eV")
print(f"====================================================")

# Setup high-res grid
grid = make_r_grid(r_max=200.0, n_points=12000)
core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
V_core = V_core_on_grid(grid, core_params)

# Get 1s and 2s states
states_s = solve_bound_states(grid, V_core, l=0, n_states_max=5)
orb_1s = min(states_s, key=lambda s: abs(s.energy_au + 0.5))
orb_2s = min(states_s, key=lambda s: abs(s.energy_au + 0.125))

E_f = E_INC_EV - (orb_2s.energy_au - orb_1s.energy_au)*27.211
ki = k_from_E_eV(E_INC_EV)
kf = k_from_E_eV(E_f)

U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, ki, kf, use_exchange=False)

print(f"ki = {ki:.4f}, kf = {kf:.4f}")
print("-" * 110)
print(f"{'L':>3} | {'Phase_i':>8} | {'Phase_f':>8} | {'I0':>10} | {'I1':>10} | {'I2':>10} | {'Sum(|ID|)':>10} | {'Notes'}")
print("-" * 110)

prev_sum_D = 0
for L in range(25):
    try:
        chi_i = solve_continuum_wave(grid, U_i, L, E_INC_EV, z_ion=0.0)
        chi_f = solve_continuum_wave(grid, U_f, L, E_f, z_ion=0.0)
        
        result = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_1s, orb_2s, chi_i, chi_f, L_max=3)
        ID = result.I_L_direct
        sum_D = sum(abs(v) for v in ID.values())
        
        notes = ""
        if sum_D > prev_sum_D and L > 2:
            notes = "UPTURN!"
        
        I0 = ID.get(0, 0.0)
        I1 = ID.get(1, 0.0)
        I2 = ID.get(2, 0.0)
        
        # Access amplitudes if available (might need to check attribute names)
        Ai = getattr(chi_i, 'amplitude', 1.0) # try 'amplitude' or 'A_amp'
        Af = getattr(chi_f, 'amplitude', 1.0)
        
        print(f"{L:>3} | {chi_i.phase_shift:>8.4f} | {chi_f.phase_shift:>8.4f} | {I0:>10.3e} | {I1:>10.3e} | {I2:>10.3e} | {sum_D:>10.3e} | {chi_i.r_match:>6.1f} | {notes}")
        
        prev_sum_D = sum_D
        
    except Exception as e:
        print(f"{L:>3} | ERROR: {e}")

print("====================================================")
