"""
Diagnostic: r_max impact on DWBA calculations

Tests how r_max choice affects:
1. Wavefunction accuracy (phase extraction)
2. Cross section results
3. Numerical stability

Key concerns:
- Too small r_max: truncates wavefunctions, wrong phase shifts
- Too large r_max: numerical issues in oscillatory tail, longer computation
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials
from continuum import solve_continuum_wave

# Hydrogen 1s -> 2s parameters
core = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

def test_r_max_impact(E_eV, L, r_max_values, n_points_base=3000):
    """Test phase shift extraction for different r_max values."""
    print(f"\n{'='*70}")
    print(f" r_max IMPACT TEST: E = {E_eV} eV, L = {L}")
    print(f"{'='*70}")
    
    k = k_from_E_eV(E_eV)
    wavelength = 2 * np.pi / k
    
    print(f"\n k = {k:.4f} a.u., wavelength = {wavelength:.2f} a.u.")
    print(f" Turning point r_t(L) = {(L+0.5)/k:.1f} a.u.")
    print()
    
    # Classical turning point
    r_turning = (L + 0.5) / k
    
    results = []
    
    print(f" {'r_max':>8} | {'n_pts':>6} | {'r_max/λ':>8} | {'r_max/r_t':>8} | {'δ (rad)':>10} | {'A':>10} | {'method':>15}")
    print(" " + "-" * 85)
    
    for r_max in r_max_values:
        # Scale n_points proportionally to maintain density
        n_pts = int(n_points_base * r_max / 200.0)
        n_pts = max(1000, min(n_pts, 20000))  # Reasonable bounds
        
        try:
            grid = make_r_grid(r_max, n_pts)
            V = V_core_on_grid(grid, core)
            orb = solve_bound_states(grid, V, l=0, n_states_max=1)[0]
            U, _ = build_distorting_potentials(grid, V, orb, orb, k, k, use_exchange=False)
            
            cw = solve_continuum_wave(grid, U, L, E_eV, phase_extraction_method='hybrid')
            
            if cw:
                delta = cw.phase_shift
                A = 1.0  # Normalized
                method = cw.phase_method if hasattr(cw, 'phase_method') else 'unknown'
                
                ratio_wavelength = r_max / wavelength
                ratio_turning = r_max / r_turning if r_turning > 0 else 999
                
                print(f" {r_max:8.1f} | {n_pts:6d} | {ratio_wavelength:8.1f} | {ratio_turning:8.1f} | {delta:+10.6f} | {A:10.2e} | {method:>15}")
                
                results.append({
                    'r_max': r_max,
                    'n_pts': n_pts,
                    'delta': delta,
                    'method': method
                })
            else:
                print(f" {r_max:8.1f} | {n_pts:6d} | {'FAILED':>8} | {'':>8} | {'':>10} | {'':>10} | {'':>15}")
        except Exception as e:
            print(f" {r_max:8.1f} | ERROR: {str(e)[:50]}")
    
    # Check consistency
    if len(results) >= 2:
        print()
        deltas = [r['delta'] for r in results]
        delta_range = max(deltas) - min(deltas)
        # Unwrap if needed
        delta_range_unwrapped = min(delta_range, 2*np.pi - delta_range)
        
        print(f" Phase range: {delta_range_unwrapped:.4f} rad ({np.degrees(delta_range_unwrapped):.2f}°)")
        if delta_range_unwrapped < 0.01:
            print(" ✓ STABLE: Phase shift is consistent across r_max values")
        elif delta_range_unwrapped < 0.1:
            print(" ~ MODERATE: Some variation in phase shift")
        else:
            print(" ✗ UNSTABLE: Significant phase variation - check r_max choice!")
    
    return results

# =============================================================================
# MAIN TESTS
# =============================================================================

print("\n" + "="*70)
print(" r_max SCALING VERIFICATION")
print("="*70)

# Test 1: Low energy (where r_max matters most)
print("\n" + "="*70)
print(" TEST 1: LOW ENERGY (10 eV) - r_max sensitivity")
print("="*70)
r_max_values = [50, 100, 150, 200, 300, 500, 1000]
for L in [0, 5, 10]:
    test_r_max_impact(10.0, L, r_max_values)

# Test 2: Medium energy
print("\n" + "="*70)
print(" TEST 2: MEDIUM ENERGY (50 eV)")
print("="*70)
for L in [0, 10, 20]:
    test_r_max_impact(50.0, L, r_max_values)

# Test 3: High energy (where r_max should matter less)
print("\n" + "="*70)
print(" TEST 3: HIGH ENERGY (300 eV)")
print("="*70)
for L in [0, 20, 40]:
    test_r_max_impact(300.0, L, r_max_values)

# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "="*70)
print(" KEY FINDINGS & RECOMMENDATIONS")
print("="*70)
print("""
r_max SELECTION GUIDELINES:

1. MINIMUM REQUIREMENT:
   r_max > C × r_turning = C × (L_max + 0.5) / k
   where C ≈ 2.5 is safety factor
   
   For LOW energy: k is small → need LARGE r_max
   For HIGH energy: k is large → can use SMALLER r_max

2. MAXIMUM PRACTICAL LIMIT:
   r_max should not be unnecessarily large because:
   - More grid points needed (slower computation)
   - Potential numerical issues in far tail
   - Bound state wavefunctions decay exponentially anyway

3. ADAPTIVE STRATEGY:
   Current code uses: r_max = max(base_r_max, C × (L+0.5)/k)
   This ensures r_max grows for low energy but stays at base for high energy.

4. TYPICAL VALUES:
   - H atoms: base_r_max = 200 a.u. is usually sufficient
   - Heavier atoms with diffuse orbitals: may need 300-500 a.u.
   - Very low energy (<5 eV): may need 500+ a.u.
""")
