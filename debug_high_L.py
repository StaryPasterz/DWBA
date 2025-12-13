
import numpy as np
from grid import make_r_grid
from distorting_potential import DistortingPotential
from continuum import solve_continuum_wave

def test_high_L():
    print("=== HIGH L CONTINUUM CHECK (1000 eV, L=100) -- He+ Case ===")
    
    # Grid
    grid = make_r_grid(r_min=1e-5, r_max=100.0, n_points=3000)
    
    # Potential: He+ (Z=2 nucleus, 1 electron).
    # Short range: -2/r. Long range: -1/r.
    # U = -1/r - 1/r * exp(-2r) (approx)
    U_arr = -1.0/grid.r - (1.0/grid.r)*np.exp(-2*grid.r)
    U_obj = DistortingPotential(U_arr, grid)
    
    E_eV = 1000.0
    L = 100
    Z_ion = 1.0 # Or 0 for neutral H? Projectile sees neutral H at long range.
    # But close range sees nucleus.
    # In dwba code: Z_ion is Isymptotic charge?
    # For H target (neutral), Z_ion = 0.
    # For He+ target, Z_ion = 1.
    # If I use Z_ion=0 (neutral).
    
    z_eff = 0.0 # Neutral H
    
    print(f"Solving for L={L}, E={E_eV} eV...")
    cw = solve_continuum_wave(grid, U_obj, L, E_eV, z_eff)
    
    if cw is None:
        print("FAILED: returned None.")
        return

    chi = cw.chi_of_r
    max_chi = np.max(np.abs(chi))
    print(f"Max Amplitude of Chi: {max_chi:.4e}")
    print(f"Chi at r_max: {chi[-1]:.4e}")

    # Sample points
    indices = np.linspace(0, len(grid.r)-1, 10, dtype=int)
    print("\n--- Wavefunction Sample ---")
    for idx in indices:
        print(f"r={grid.r[idx]:.1f}, chi={chi[idx]:.4e}")

    
    # Check if it exploded
    if max_chi > 10.0:
        print("!! WARNING: Amplitude > 10. Instability detected.")
        
    # Also check small amplitude (zeros?)
    if max_chi < 0.1:
        print("!! WARNING: Amplitude < 0.1. Normalization failure?")

if __name__ == "__main__":
    test_high_L()
