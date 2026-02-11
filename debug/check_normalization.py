"""
Quick diagnostic: verify continuum wave normalization at a specific energy.
Check if chi_of_r has amplitude 1 or sqrt(2/pi) asymptotically.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from grid import make_r_grid
from continuum import solve_continuum_wave
from distorting_potential import DistortingPotential

# Setup: simple grid
grid = make_r_grid(r_min=1e-5, r_max=200.0, n_points=2000)
r = grid.r

# Zero distorting potential for simplicity 
U_zero = np.zeros_like(r)
U_pot = DistortingPotential(U_zero, r)

# Solve for l=0 continuum wave at 50 eV 
for E_eV in [50.0, 100.0, 200.0]:
    cw = solve_continuum_wave(grid, U_pot, l=0, E_eV=E_eV, z_ion=0.0)
    
    if cw is None:
        print(f"E={E_eV} eV: solve_continuum_wave returned None")
        continue
    
    print(f"\nE={E_eV} eV, k={cw.k_au:.4f} a.u.:")
    
    # Measure asymptotic amplitude using RMS of chi in the tail
    n_tail = 200
    chi_tail = cw.chi_of_r[-n_tail:]
    r_tail = r[-n_tail:]
    
    chi_rms = np.sqrt(np.mean(chi_tail**2))
    A_measured = chi_rms * np.sqrt(2.0)  # RMS of A*sin(...) = A/sqrt(2)
    
    A_unit = 1.0
    A_sqrt2pi = np.sqrt(2.0/np.pi)
    
    print(f"  Measured amplitude:          {A_measured:.6f}")
    print(f"  Expected if unit:            {A_unit:.6f} (ratio={A_measured/A_unit:.4f})")
    print(f"  Expected if sqrt(2/pi):      {A_sqrt2pi:.6f} (ratio={A_measured/A_sqrt2pi:.4f})")
    
    # Also check max absolute value in tail (should be ≈ amplitude)
    A_max = np.max(np.abs(chi_tail))
    print(f"  Max |chi| in tail:           {A_max:.6f}")

print(f"\n--- Reference values ---")
print(f"  sqrt(2/pi)  = {np.sqrt(2.0/np.pi):.6f}")
print(f"  2/pi        = {2.0/np.pi:.6f}")
print(f"  (2/pi)^2    = {(2.0/np.pi)**2:.6f}")
print(f"  (2*pi)^4    = {(2.0*np.pi)**4:.6f}")
