
import numpy as np
import time
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from potential_core import CorePotentialParams

def check_angular_convergence():
    Z = 1.0
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    exc_spec = ExcitationChannelSpec(
        l_i=0, l_f=0, n_index_i=1, n_index_f=2, 
        N_equiv=1, L_max_integrals=10, L_target_i=0, L_target_f=0, L_max_projectile=10
    )
    
    print("--- 50 eV Convergence ---")
    res_50_low = compute_total_excitation_cs(50.0, exc_spec, core_params, n_points=3000, n_theta=200)
    print(f"Points=200  | Sigma={res_50_low.sigma_total_cm2:.4e} cm2")
    
    res_50_high = compute_total_excitation_cs(50.0, exc_spec, core_params, n_points=3000, n_theta=2000)
    print(f"Points=2000 | Sigma={res_50_high.sigma_total_cm2:.4e} cm2")
    
    diff_50 = (res_50_high.sigma_total_cm2 - res_50_low.sigma_total_cm2)/res_50_low.sigma_total_cm2 * 100
    print(f"Change: {diff_50:.2f}%")

    print("\n--- 1000 eV Convergence ---")
    res_1000_low = compute_total_excitation_cs(1000.0, exc_spec, core_params, n_points=3000, n_theta=200)
    print(f"Points=200  | Sigma={res_1000_low.sigma_total_cm2:.4e} cm2")
    
    res_1000_high = compute_total_excitation_cs(1000.0, exc_spec, core_params, n_points=3000, n_theta=2000)
    print(f"Points=2000 | Sigma={res_1000_high.sigma_total_cm2:.4e} cm2")
    
    diff_1000 = (res_1000_high.sigma_total_cm2 - res_1000_low.sigma_total_cm2)/res_1000_low.sigma_total_cm2 * 100
    print(f"Change: {diff_1000:.2f}%")

if __name__ == "__main__":
    check_angular_convergence()
