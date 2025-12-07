
import numpy as np
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from potential_core import CorePotentialParams

def verify():
    # Hydrogen 1s -> 2s
    Z = 1.0
    E_eV = 100.0
    
    spec = ExcitationChannelSpec(
        l_i=0, l_f=0,
        n_index_i=1, n_index_f=2, # n=1 (1s), n=2 (2s)
        N_equiv=1,
        L_max_integrals=15, 
        L_target_i=0,
        L_target_f=0
    )
    
    # Pure Coulomb
    core = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    
    print(f"Running H 1s->2s at {E_eV} eV...")
    res = compute_total_excitation_cs(E_eV, spec, core, r_max=60.0, n_points=1000)
    
    print(f"Sigma (cm^2): {res.sigma_total_cm2:.4e}")
    print(f"Sigma (a.u.): {res.sigma_total_au:.4e}")

if __name__ == "__main__":
    verify()
