import numpy as np
import time
import sys
import traceback

def test_exc():
    from driver import compute_total_excitation_cs, ExcitationChannelSpec
    from atom_library import get_atom
    
    print("--- Test Excitation H 1s->2s (50 eV) ---")
    atom = get_atom("H")
    spec = ExcitationChannelSpec(
        l_i=0, l_f=0, n_index_i=1, n_index_f=2, N_equiv=1, 
        L_max_integrals=15, L_target_i=0, L_target_f=0
    )
    
    t0 = time.time()
    res = compute_total_excitation_cs(
        E_incident_eV=50.0,
        chan=spec,
        core_params=atom.core_params,
        n_points=2000,
        r_max=100.0,
        exchange_method='fumc'
    )
    dt = time.time() - t0
    
    print(f"Sigma = {res.sigma_total_cm2:.3e} cm2")
    print(f"Time  = {dt:.2f} s")

def test_ion():
    import ionization
    print(f"DEBUG: ionization file: {ionization.__file__}")
    with open(ionization.__file__, 'r') as f:
        print(f"DEBUG: line count: {len(f.readlines())}")
    
    from ionization import compute_ionization_cs, IonizationChannelSpec
    from atom_library import get_atom
    
    print("\n--- Test Ionization H 1s (1 Steps) ---")
    atom = get_atom("H")
    spec = IonizationChannelSpec(
        l_i=0, n_index_i=1, N_equiv=1,
        l_eject_max=1, L_max=10, L_i_total=0, L_max_projectile=10
    )
    
    t0 = time.time()
    res = compute_ionization_cs(
        E_incident_eV=50.0,
        chan=spec,
        core_params=atom.core_params,
        n_points=2000,
        n_energy_steps=1
    )
    dt = time.time() - t0
    
    print(f"Sigma Ion = {res.sigma_total_cm2:.3e} cm2")
    print(f"Time = {dt:.2f} s")

if __name__ == "__main__":
    try:
        test_exc()
        test_ion()
    except Exception:
        print("CRITICAL ERROR IN TEST:")
        traceback.print_exc()
