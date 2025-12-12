
import time
import numpy as np
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from potential_core import CorePotentialParams

def debug_1000ev_full():
    print("=== DEBUG HIGH ENERGY (1000 eV) FULL ===")
    
    Z = 1.0
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    E = 1000.0
    
    # Case 2: 1s -> 2p (S-P)
    print("\n--- H 1s -> 2p at 1000 eV ---")
    # Reduced points for speed, but enough for accuracy
    spec_2p = ExcitationChannelSpec(
        l_i=0, l_f=1, n_index_i=1, n_index_f=2, 
        N_equiv=1, L_max_integrals=15, L_target_i=0, L_target_f=1, L_max_projectile=-1 # Use dynamic
    )
    
    try:
        t0 = time.perf_counter()
        # n_points=2000 slightly faster
        res = compute_total_excitation_cs(E, spec_2p, core_params, r_max=100.0, n_points=2000)
        dt = time.perf_counter() - t0
        print(f"SUCCESS 1s->2p. Time: {dt:.2f}s")
        print(f"Sigma = {res.sigma_total_cm2:.3e} cm2")
        
        # Also check 1s->2s for reference
        print("\n--- H 1s -> 2s at 1000 eV ---")
        spec_2s = ExcitationChannelSpec(
            l_i=0, l_f=0, n_index_i=1, n_index_f=2, 
            N_equiv=1, L_max_integrals=15, L_target_i=0, L_target_f=0, L_max_projectile=-1
        )
        res2 = compute_total_excitation_cs(E, spec_2s, core_params, r_max=100.0, n_points=2000)
        print(f"Sigma = {res2.sigma_total_cm2:.3e} cm2")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_1000ev_full()
