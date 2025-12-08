
import numpy as np
import time
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from potential_core import CorePotentialParams

def run_test():
    Z = 1.0
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    
    # Base Spec with L_proj=25 (Default)
    base_spec = ExcitationChannelSpec(
        l_i=0, l_f=0, n_index_i=1, n_index_f=2, 
        N_equiv=1, L_max_integrals=10, L_target_i=0, L_target_f=0, 
        L_max_projectile=25
    )
    
    # High L Spec with L_proj=60
    highL_spec = ExcitationChannelSpec(
        l_i=0, l_f=0, n_index_i=1, n_index_f=2, 
        N_equiv=1, L_max_integrals=10, L_target_i=0, L_target_f=0, 
        L_max_projectile=60
    )

    print("=== Convergence Test 1000 eV ===")
    
    # 1. Baseline: L=25, Theta=200
    t0 = time.perf_counter()
    res_base = compute_total_excitation_cs(1000.0, base_spec, core_params, n_points=3000, n_theta=200)
    print(f"Base (L=25, Th=200):  {res_base.sigma_total_cm2:.4e} cm2 (Time: {time.perf_counter()-t0:.1f}s)")
    
    # 2. Angle Check: L=25, Theta=2000
    t0 = time.perf_counter()
    res_ang = compute_total_excitation_cs(1000.0, base_spec, core_params, n_points=3000, n_theta=2000)
    print(f"Angle(L=25, Th=2000): {res_ang.sigma_total_cm2:.4e} cm2 (Time: {time.perf_counter()-t0:.1f}s)")
    
    # 3. L Check: L=60, Theta=200
    t0 = time.perf_counter()
    res_L = compute_total_excitation_cs(1000.0, highL_spec, core_params, n_points=3000, n_theta=200)
    print(f"L_max(L=60, Th=200):  {res_L.sigma_total_cm2:.4e} cm2 (Time: {time.perf_counter()-t0:.1f}s)")

    # Analysis
    diff_ang = (res_ang.sigma_total_cm2 - res_base.sigma_total_cm2)/res_base.sigma_total_cm2 * 100
    diff_L = (res_L.sigma_total_cm2 - res_base.sigma_total_cm2)/res_base.sigma_total_cm2 * 100
    
    print("\nanalysis:")
    print(f"Angular Effect (200->2000): {diff_ang:.2f}%")
    print(f"L_max Effect (25->60):      {diff_L:.2f}%")

if __name__ == "__main__":
    run_test()
