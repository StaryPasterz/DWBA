"""
Quick test script for ionization module after fixes.
Tests H(1s) ionization at 50 eV.
NIST BEB reference: ~0.6 × 10^-16 cm²
"""

import sys
sys.path.insert(0, r"c:\Users\Jan\Desktop\Projekty\QRS\DW_antigravity_v2")

from ionization import compute_ionization_cs, IonizationChannelSpec
from potential_core import CorePotentialParams

def main():
    # Hydrogen parameters (pure Coulomb, Z=1)
    H_params = CorePotentialParams(
        Zc=1.0,
        a1=0.0, a2=1.0, a3=0.0, a4=1.0, a5=0.0, a6=1.0
    )

    # Ionization channel: H(1s) -> H+ + e-
    # Using adaptive L_max - set high limit, convergence will truncate
    spec = IonizationChannelSpec(
        l_i=0,             # s orbital
        n_index_i=1,       # ground state (n=1)
        N_equiv=1,         # 1 electron
        l_eject_max=2,     # sum ejected waves up to l=2 (d)
        L_max=10,          # multipole max
        L_i_total=0,       # initial L=0
        L_max_projectile=20,  # Set high - adaptive convergence will stop early
        convergence_threshold=0.02  # 2% relative change threshold
    )

    # Test at 50 eV (NIST reference ~0.6e-16 cm²)
    E_test = 50.0

    print("=" * 60)
    print("IONIZATION TEST: H(1s) at 50 eV")
    print("=" * 60)
    print(f"NIST BEB Reference: ~0.6 × 10^-16 cm²")
    print()

    try:
        result = compute_ionization_cs(
            E_test, 
            spec, 
            H_params,
            r_max=100.0, # Reduced range
            n_points=1000,
            n_energy_steps=3  # Minimal steps
        )
        
        print()
        print("=" * 60)
        print(f"RESULT: σ = {result.sigma_total_cm2:.3e} cm²")
        print(f"IP used: {result.IP_eV:.2f} eV")
        print("=" * 60)
        
        # Compare with NIST
        nist_ref = 0.6e-16
        ratio = result.sigma_total_cm2 / nist_ref
        print(f"Ratio to NIST: {ratio:.2f}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

