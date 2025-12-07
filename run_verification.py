
import numpy as np
import time
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from ionization import compute_ionization_cs, IonizationChannelSpec
from potential_core import CorePotentialParams

def run_verification():
    print("=== DWBA Verification Run ===")
    
    # Common Parameters for Hydrogen (Z=1)
    Z = 1.0
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0) # Pure Coulomb -1/r for H core (Z-1=0 charge seen by e)
    # Wait, Core Potential for H atom:
    # V_core is the potential of the NUCLEUS (or core).
    # For H atom, we have 1 electron. Core is just the Proton.
    # SAE model: The "core" is the system WITHOUT the active electron.
    # For H, if we remove the active electron, we have bare proton H+.
    # So V_A+ should be -1/r ?
    # Let's check potential_core.py: V = -(Zc + S)/r.
    # If a_i=0, V = -Zc/r.
    # If target is H, Zc=1. So V = -1/r. Correct.
    
    r_max = 500.0 # Large grid for accuracy
    n_points = 5000
    
    # --- TEST 1: Excitation H 1s -> 2s at 50 eV ---
    print("\n--- TEST 1: Excitation H 1s -> 2s (50 eV) ---")
    exc_spec = ExcitationChannelSpec(
        l_i=0, l_f=0,          # s -> s
        n_index_i=1, n_index_f=2, # 1s -> 2s
        N_equiv=1,
        L_max_integrals=10,
        L_target_i=0, L_target_f=0,
        L_max_projectile=10
    )
    
    try:
        t0 = time.perf_counter()
        res_exc = compute_total_excitation_cs(
            E_incident_eV=50.0,
            chan=exc_spec,
            core_params=core_params,
            r_max=r_max,
            n_points=n_points
        )
        dt = time.perf_counter() - t0
        print(f"Excitation Calc Time: {dt:.2f} s")
        if res_exc.ok_open_channel:
            print(f"Sigma Exc (H 1s->2s @ 50eV): {res_exc.sigma_total_cm2:.4e} cm^2")
            print(f"Sigma Exc (H 1s->2s @ 50eV): {res_exc.sigma_total_au:.4e} a.u.")
        else:
            print("Channel closed or failed.")
    except Exception as e:
        print(f"Excitation Failed: {e}")

    # --- TEST 2: Ionization H 1s at 50 eV ---
    print("\n--- TEST 2: Ionization H 1s (50 eV) ---")
    ion_spec = IonizationChannelSpec(
        l_i=0,
        n_index_i=1,
        N_equiv=1,
        l_eject_max=3,   # Sum up to F-wave ejected
        L_max=10,
        L_i_total=0,
        L_max_projectile=15
    )
    
    try:
        t0 = time.perf_counter()
        res_ion = compute_ionization_cs(
            E_incident_eV=50.0,
            chan=ion_spec,
            core_params=core_params,
            r_max=r_max,
            n_points=n_points,
            n_energy_steps=10 # quick scan
        )
        dt = time.perf_counter() - t0
        print(f"Ionization Calc Time: {dt:.2f} s")
        print(f"Sigma Ion (H 1s @ 50eV): {res_ion.sigma_total_cm2:.4e} cm^2")
        print(f"Sigma Ion (H 1s @ 50eV): {res_ion.sigma_total_au:.4e} a.u.")
        print(f"IP: {res_ion.IP_eV:.2f} eV")
    except Exception as e:
        print(f"Ionization Failed: {e}")

if __name__ == "__main__":
    run_verification()
