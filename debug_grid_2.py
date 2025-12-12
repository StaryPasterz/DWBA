
import time
import numpy as np
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from potential_core import CorePotentialParams
from grid import make_r_grid
from continuum import solve_continuum_wave
from distorting_potential import DistortingPotential
# Add MockPotential if DistortingPotential import fails or reuse logic
class MockPotential:
    def __init__(self, U_arr):
        self.U_of_r = U_arr

def test_grid_resolution():
    print("=== GRID RESOLUTION TEST (1000 eV, L=50) ===")
    
    Z = 1.0
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    E = 1000.0
    L = 50
    
    # We want to check radial integral magnitude or just continuum wave quality?
    # Actually, continuum wave quality is checked by Solve.
    # The integral involves two waves.
    # Let's run a FULL calc for ONE channel (L=50) with different grids.
    
    spec = ExcitationChannelSpec(
        l_i=L, l_f=L+1, n_index_i=1, n_index_f=2, 
        N_equiv=1, L_max_integrals=15, L_target_i=0, L_target_f=1, L_max_projectile=L # Force single L
    )
    
    grids = [2000, 3000, 5000, 10000]
    
    for n_pts in grids:
        print(f"\n--- n_points = {n_pts} ---")
        try:
            t0 = time.perf_counter()
            # Note: We must hack compute_total_excitation_cs to ONLY run L (via spec)
            # Actually, compute_total_excitation_cs loops 0..L_max.
            # If we set L_max_projectile=50, it runs 0..50. That's slow.
            # We want to isolate ONE partial wave contribution.
            # But compute_total_excitation_cs returns TOTAL sigma.
            # If we assume l=50 is dominant or representative of 'noise'.
            # Or we can just run the full sum up to 5 (fast) and see if it changes.
            # Let's run full sum up to L=5. 
            # If grid affects L=5, it affects everything.
            
            # Wait, noise happens at HIGH L?
            # Let's run full cal with tiny L_max (e.g. 5) to check sensitivity first.
            # Removed bad call

            # wait, spec has l_i=L=50.
            # compute_total uses spec.l_i as TARGET quantum number? No.
            # spec.l_i is TARGET initial l.
            # spec.L_max_projectile is projectile.
            # Ah. spec L=50 is wrong. 1s->2p means l_i=0, l_f=1.
            
            spec_real = ExcitationChannelSpec(
                l_i=0, l_f=1, n_index_i=1, n_index_f=2, 
                N_equiv=1, L_max_integrals=15, L_target_i=0, L_target_f=1, L_max_projectile=20
            )
            
            res = compute_total_excitation_cs(E, spec_real, core_params, r_max=100.0, n_points=n_pts)
            
            print(f"Sigma(L_max=10) = {res.sigma_total_cm2:.4e} cm2")
            print(f"Time: {time.perf_counter() - t0:.3f}s")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_grid_resolution()
