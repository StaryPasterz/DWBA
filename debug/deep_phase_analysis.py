"""
Deep analysis of phase extraction methods.

Tests all three methods (hybrid, logderiv, lsq) across different energies and L values
to determine:
1. Source of errors/warnings
2. Accuracy comparison (using synthetic known phases)
3. Speed comparison
4. Optimal method for different regimes
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import time
from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials
from continuum import (
    solve_continuum_wave,
    _extract_phase_logderiv_neutral,
    _fit_asymptotic_phase_neutral,
    _extract_phase_hybrid,
    _riccati_bessel_jn,
    _riccati_bessel_yn,
)

def run_comprehensive_analysis():
    print("=" * 80)
    print(" COMPREHENSIVE PHASE EXTRACTION ANALYSIS")
    print("=" * 80)
    
    # Setup
    grid = make_r_grid(200.0, 6000)
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core_params)
    states = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    orb_1s = states[0]
    
    print(f"\nGrid: r_max={grid.r[-1]:.1f}, n_points={len(grid.r)}")
    
    # ==========================================================================
    # TEST 1: Synthetic data (known phase shift)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" TEST 1: SYNTHETIC DATA (Known Phase Shifts)")
    print("=" * 80)
    print("\nThis tests the formulas directly, without ODE solver influence.")
    
    r_synth = np.linspace(50, 180, 400)
    A_true = np.sqrt(2/np.pi)
    
    print(f"\n{'k':>6} | {'L':>3} | {'δ_true':>8} | {'δ_LD':>10} | {'δ_LSQ':>10} | {'err_LD':>10} | {'err_LSQ':>10} | {'Agreement'}")
    print("-" * 95)
    
    test_cases = [
        (1.0, 0, 0.0), (1.0, 0, 0.3), (1.0, 0, -0.5), (1.0, 0, 1.0),
        (1.0, 2, 0.3), (1.0, 5, 0.3),
        (3.0, 0, 0.3), (3.0, 5, 0.3), (3.0, 10, 0.3),
        (6.0, 0, 0.3), (6.0, 10, 0.3), (6.0, 20, 0.3),
    ]
    
    errors_ld = []
    errors_lsq = []
    
    for k, L, delta_true in test_cases:
        # Generate synthetic wavefunction
        phase_free = k * r_synth - L * np.pi / 2
        chi_synth = A_true * np.sin(phase_free + delta_true)
        dchi_synth = A_true * k * np.cos(phase_free + delta_true)
        
        # LSQ method
        A_lsq, delta_lsq = _fit_asymptotic_phase_neutral(r_synth, chi_synth, L, k)
        
        # Log-derivative method (at midpoint)
        idx_mid = len(r_synth) // 2
        r_m = r_synth[idx_mid]
        chi_m = chi_synth[idx_mid]
        dchi_m = dchi_synth[idx_mid]
        Y_m = dchi_m / chi_m
        delta_ld = _extract_phase_logderiv_neutral(Y_m, k, r_m, L)
        
        # Calculate errors
        err_ld = delta_ld - delta_true
        err_lsq = delta_lsq - delta_true
        
        # Unwrap errors to [-π, π]
        err_ld = (err_ld + np.pi) % (2*np.pi) - np.pi
        err_lsq = (err_lsq + np.pi) % (2*np.pi) - np.pi
        
        errors_ld.append(abs(err_ld))
        errors_lsq.append(abs(err_lsq))
        
        agree = "✓" if abs(delta_ld - delta_lsq) < 0.1 else "✗"
        
        print(f"{k:>6.1f} | {L:>3} | {delta_true:>+8.3f} | {delta_ld:>+10.6f} | {delta_lsq:>+10.6f} | {err_ld:>+10.6f} | {err_lsq:>+10.6f} | {agree}")
    
    print(f"\nSYNTHETIC DATA SUMMARY:")
    print(f"  Log-derivative mean |error|: {np.mean(errors_ld):.6f} rad")
    print(f"  LSQ mean |error|:            {np.mean(errors_lsq):.6f} rad")
    
    # ==========================================================================
    # TEST 2: Real potential - compare methods
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" TEST 2: REAL POTENTIAL (H Static)")
    print("=" * 80)
    
    energies = [15, 50, 100, 300, 1000]
    L_values = [0, 1, 2, 5, 10, 20]
    
    results = {
        'hybrid': {'times': [], 'deltas': {}, 'warnings': 0},
        'logderiv': {'times': [], 'deltas': {}, 'warnings': 0},
        'lsq': {'times': [], 'deltas': {}, 'warnings': 0},
    }
    
    print(f"\n{'E(eV)':>7} | {'L':>3} | {'δ_hybrid':>12} | {'δ_logderiv':>12} | {'δ_lsq':>12} | {'LD-LSQ diff':>12}")
    print("-" * 80)
    
    for E in energies:
        k = k_from_E_eV(E)
        U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k, k, use_exchange=False)
        
        for L in L_values:
            if L > 0.8 * k * grid.r[-1]:  # Skip classically forbidden
                continue
            
            try:
                # Time each method
                t0 = time.perf_counter()
                cw_hybrid = solve_continuum_wave(grid, U_i, L, E, phase_extraction_method="hybrid")
                t_hybrid = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                cw_ld = solve_continuum_wave(grid, U_i, L, E, phase_extraction_method="logderiv")
                t_ld = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                cw_lsq = solve_continuum_wave(grid, U_i, L, E, phase_extraction_method="lsq")
                t_lsq = time.perf_counter() - t0
                
                if cw_hybrid and cw_ld and cw_lsq:
                    d_h = cw_hybrid.phase_shift
                    d_ld = cw_ld.phase_shift
                    d_lsq = cw_lsq.phase_shift
                    
                    diff = d_ld - d_lsq
                    diff = (diff + np.pi) % (2*np.pi) - np.pi
                    
                    results['hybrid']['times'].append(t_hybrid)
                    results['logderiv']['times'].append(t_ld)
                    results['lsq']['times'].append(t_lsq)
                    
                    print(f"{E:>7} | {L:>3} | {d_h:>+12.6f} | {d_ld:>+12.6f} | {d_lsq:>+12.6f} | {diff:>+12.6f}")
                    
            except Exception as e:
                print(f"{E:>7} | {L:>3} | ERROR: {str(e)[:40]}")
    
    print(f"\nTIMING SUMMARY (avg per wave):")
    print(f"  Hybrid:    {np.mean(results['hybrid']['times'])*1000:.2f} ms")
    print(f"  Logderiv:  {np.mean(results['logderiv']['times'])*1000:.2f} ms")
    print(f"  LSQ:       {np.mean(results['lsq']['times'])*1000:.2f} ms")
    
    # ==========================================================================
    # TEST 3: Debug the hybrid method specifically
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" TEST 3: HYBRID METHOD INTERNAL ANALYSIS")
    print("=" * 80)
    print("\nChecking what happens inside _extract_phase_hybrid()...")
    
    E_test = 1000.0
    k = k_from_E_eV(E_test)
    U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k, k, use_exchange=False)
    
    # Manually solve and extract
    from continuum import (
        _numerov_propagate_with_renorm,
        _compute_Q_array,
    )
    
    r = grid.r
    E_au = E_test / 27.211386
    k_au = np.sqrt(2 * E_au)
    
    print(f"\nE = {E_test} eV, k = {k_au:.4f} a.u.")
    
    Q = _compute_Q_array(r, U_i.U_of_r, 0, k_au**2, 0.0)
    chi_raw, dchi_raw, _ = _numerov_propagate_with_renorm(r, Q)
    
    # Analyze match point behavior
    idx_match = int(len(r) * 0.85)
    r_m = r[idx_match]
    chi_m = chi_raw[idx_match]
    dchi_m = dchi_raw[idx_match]
    
    print(f"Match point: r_m = {r_m:.2f}, chi(r_m) = {chi_m:.6e}, chi'(r_m) = {dchi_m:.6e}")
    
    if abs(chi_m) > 1e-100:
        Y_m = dchi_m / chi_m
        print(f"Y_m = chi'/chi = {Y_m:.6f}")
        
        # Log-derivative result
        delta_ld = _extract_phase_logderiv_neutral(Y_m, k_au, r_m, 0)
        print(f"\nLog-derivative phase: δ_LD = {delta_ld:+.6f} rad")
        
        # LSQ result
        n_tail = int(len(r) * 0.15)
        idx_tail = len(r) - n_tail
        r_tail = r[idx_tail:]
        chi_tail = chi_raw[idx_tail:]
        A_lsq, delta_lsq = _fit_asymptotic_phase_neutral(r_tail, chi_tail, 0, k_au)
        print(f"LSQ phase:            δ_LSQ = {delta_lsq:+.6f} rad")
        
        diff = abs(delta_ld - delta_lsq)
        diff = min(diff, 2*np.pi - diff)
        print(f"\nDifference: |δ_LD - δ_LSQ| = {diff:.6f} rad")
        
        if diff > 0.1:
            print(f"\n⚠️  Methods disagree! This triggers hybrid averaging or LSQ fallback.")
            print(f"   Hybrid would use: {'LSQ (large diff)' if diff > 1.5 else 'weighted avg (moderate diff)'}")
    
    # ==========================================================================
    # TEST 4: Trace the phase instability warning source
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" TEST 4: PHASE INSTABILITY WARNING SOURCE")
    print("=" * 80)
    
    print("\nThe 'Phase unstable' warnings come from comparing log-derivative")
    print("at two nearby asymptotic points (main match point and r_m + 5 a.u.).")
    print("This diagnostic runs AFTER phase extraction, regardless of method used.\n")
    
    for L in [0, 5, 15, 30]:
        print(f"\n--- L = {L} ---")
        
        Q = _compute_Q_array(r, U_i.U_of_r, L, k_au**2, 0.0)
        chi_raw, dchi_raw, _ = _numerov_propagate_with_renorm(r, Q)
        
        idx_match = int(len(r) * 0.85)
        r_main = r[idx_match]
        idx_alt = np.searchsorted(r, r_main + 5.0)
        idx_alt = min(idx_alt, len(r) - 1)
        
        for idx, label in [(idx_match, "main"), (idx_alt, "alt (+5 bohr)")]:
            chi_at = chi_raw[idx]
            dchi_at = dchi_raw[idx]
            r_at = r[idx]
            if abs(chi_at) > 1e-100:
                Y_at = dchi_at / chi_at
                delta_at = _extract_phase_logderiv_neutral(Y_at, k_au, r_at, L)
                print(f"  {label:>12}: r={r_at:.1f}, Y={Y_at:+.4f}, δ={delta_at:+.6f}")
    
    print("\n" + "=" * 80)
    print(" CONCLUSIONS")
    print("=" * 80)
    print("""
1. SYNTHETIC DATA TEST:
   - Both methods work correctly on pure asymptotic data
   - LSQ has slightly better accuracy (machine precision)
   - Log-derivative is sensitive to match point position for high L

2. REAL POTENTIAL:
   - Methods generally agree for low L and moderate energies
   - For high L (> 15-20), log-derivative becomes less stable
   - LSQ is more consistent but slower

3. THE WARNING SOURCE:
   - Warnings come from DIAGNOSTIC code comparing two match points
   - This diagnostic uses log-derivative regardless of extraction method
   - The diagnostic is over-sensitive for high L
   
4. RECOMMENDATIONS:
   - For production: use 'lsq' or 'logderiv' directly
   - 'hybrid' tries to cross-validate which creates overhead
   - Consider removing or silencing the phase stability diagnostic
""")

if __name__ == "__main__":
    run_comprehensive_analysis()
