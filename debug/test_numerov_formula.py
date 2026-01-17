#!/usr/bin/env python
"""
test_numerov_formula.py
=======================

Diagnostic test to verify the Numerov solver formula implementation.

Tests against known analytical solutions:
1. Free particle (U=0): Should give phase shift δ = 0
2. Known potential: Compare with RK45 as reference

Run:
    python debug/test_numerov_formula.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn

# Import the Numerov propagator
from continuum import _numerov_propagate, _derivative_5point


def test_numerov_free_particle():
    """
    Test Numerov against free particle (U=0).
    
    For free particle: χ'' = [l(l+1)/r² - k²] χ
    Solution: χ(r) = A·r·j_l(kr) (regular Riccati-Bessel)
    Phase shift: δ_l = 0
    """
    print("=" * 60)
    print("TEST: Numerov vs Free Particle (U=0)")
    print("=" * 60)
    
    # Test parameters
    r_min, r_max = 0.01, 50.0
    n_points = 2000
    k_au = 1.0  # ~13.6 eV
    
    for l in [0, 1, 2, 5]:
        print(f"\n--- L = {l} ---")
        
        # Create uniform grid for simplicity
        r_grid = np.linspace(r_min, r_max, n_points)
        
        # Q(r) = l(l+1)/r² - k² (free particle, U=0)
        Q_arr = l * (l + 1) / (r_grid**2) - k_au**2
        
        # Initial conditions: χ ~ r^(l+1) at small r
        chi0 = r_grid[0]**(l + 1)
        chi1 = r_grid[1]**(l + 1)
        
        # Run Numerov
        chi_num, log_scale = _numerov_propagate(r_grid, Q_arr, chi0, chi1)
        
        # Analytical solution: Riccati-Bessel ĵ_l(kr) = kr · j_l(kr)
        rho = k_au * r_grid
        jl = spherical_jn(l, rho)
        chi_exact = rho * jl  # ĵ_l(ρ) = ρ·j_l(ρ)
        
        # Normalize both to match at some interior point (avoiding endpoints)
        idx_norm = len(r_grid) // 2
        if abs(chi_exact[idx_norm]) > 1e-10:
            scale = chi_exact[idx_norm] / chi_num[idx_norm]
            chi_num_scaled = chi_num * scale
        else:
            chi_num_scaled = chi_num
        
        # Compare in the "tail" region (last 20%)
        idx_tail = int(0.8 * n_points)
        
        # Check relative agreement
        mask = np.abs(chi_exact[idx_tail:]) > 1e-10
        if np.any(mask):
            rel_error = np.abs(chi_num_scaled[idx_tail:][mask] / chi_exact[idx_tail:][mask] - 1)
            max_rel_error = np.max(rel_error)
            mean_rel_error = np.mean(rel_error)
        else:
            max_rel_error = float('inf')
            mean_rel_error = float('inf')
        
        # Extract phase from Numerov result using LSQ fit
        n_fit = 100
        r_fit = r_grid[-n_fit:]
        chi_fit = chi_num[-n_fit:]
        
        # Fit to A·sin(kr - lπ/2 + δ)
        phase_free = k_au * r_fit - l * np.pi / 2
        sin_part = np.sin(phase_free)
        cos_part = np.cos(phase_free)
        M = np.vstack([sin_part, cos_part]).T
        coeffs, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
        A_s, A_c = coeffs
        delta_fitted = np.arctan2(A_c, A_s)
        
        # Phase should be ~0 for free particle
        print(f"  Fitted phase shift: δ = {delta_fitted:.4f} rad ({np.degrees(delta_fitted):.2f}°)")
        print(f"  (Expected: δ = 0)")
        print(f"  Max relative error in tail: {max_rel_error:.2e}")
        print(f"  Mean relative error in tail: {mean_rel_error:.2e}")
        
        if abs(delta_fitted) < 0.1:
            print(f"  ✓ PASS: Phase shift is small (< 0.1 rad)")
        else:
            print(f"  ✗ FAIL: Phase shift is too large! Indicates sign error.")
        
        if max_rel_error < 0.1:
            print(f"  ✓ PASS: Wavefunction matches analytical")
        else:
            print(f"  ✗ FAIL: Wavefunction does NOT match analytical!")


def test_numerov_formula_derivation():
    """
    Direct test of Numerov coefficients against expected formula.
    
    For χ'' = Q·χ, the correct Numerov formula (Wikipedia) is:
    
    γ_{n+1}·χ_{n+1} = 2·γ_n·χ_n - γ_{n-1}·χ_{n-1}
    
    where γ_n = 1 + (h²/12)·Q_n  (for y'' = -g·y with g = -Q)
    
    Wait, let's be precise. For y'' = f(x)·y:
    
    Numerov: (1 - h²f_{n+1}/12)y_{n+1} = 2(1 + 5h²f_n/12)y_n - (1 - h²f_{n-1}/12)y_{n-1}
    
    So for χ'' = Q·χ (f = Q):
    
    (1 - h²Q_{n+1}/12)χ_{n+1} = 2(1 + 5h²Q_n/12)χ_n - (1 - h²Q_{n-1}/12)χ_{n-1}
    """
    print("\n" + "=" * 60)
    print("TEST: Numerov coefficient signs")
    print("=" * 60)
    
    # The code has (lines 949-951):
    # a_prev = 1.0 - (h1_sq / 12.0) * Q_arr[i - 1]
    # b_curr = 2.0 + (5.0 * h_center_sq / 6.0) * Q_arr[i]  # = 2 + 10h²Q/12 = 2(1 + 5h²Q/12)
    # a_next = 1.0 - (h2_sq / 12.0) * Q_arr[i + 1]
    #
    # χ_{i+1} = (b_curr·χ_i - a_prev·χ_{i-1}) / a_next
    
    # Let's check with concrete values
    h = 0.1
    h_sq = h * h
    
    # For Q = -1 (like -k² with k=1):
    Q = -1.0
    
    # Current code coefficients:
    a_prev_code = 1.0 - (h_sq / 12.0) * Q
    b_curr_code = 2.0 + (5.0 * h_sq / 6.0) * Q
    a_next_code = 1.0 - (h_sq / 12.0) * Q
    
    # Correct formula coefficients (for χ'' = Q·χ):
    # (1 - h²Q/12)χ_{n+1} = 2(1 + 5h²Q/12)χ_n - (1 - h²Q/12)χ_{n-1}
    #
    # χ_{n+1} = [2(1 + 5h²Q_n/12)χ_n - (1 - h²Q_{n-1}/12)χ_{n-1}] / (1 - h²Q_{n+1}/12)
    #
    # coeff_prev (subtracted from numerator) = (1 - h²Q/12)
    # coeff_curr (multiplied in numerator) = 2(1 + 5h²Q/12) = 2 + 10h²Q/12 = 2 + 5h²Q/6
    # coeff_next (divisor) = (1 - h²Q/12)
    
    coeff_prev_correct = 1.0 - (h_sq / 12.0) * Q
    coeff_curr_correct = 2.0 + (5.0 * h_sq / 6.0) * Q  # 2(1 + 5h²Q/12) = 2 + 5h²Q/6
    coeff_next_correct = 1.0 - (h_sq / 12.0) * Q
    
    print(f"\nFor h={h}, Q={Q}:")
    print(f"\nCode coefficients:")
    print(f"  a_prev = {a_prev_code:.6f}")
    print(f"  b_curr = {b_curr_code:.6f}")
    print(f"  a_next = {a_next_code:.6f}")
    
    print(f"\nCorrect coefficients (for χ'' = Q·χ):")
    print(f"  coeff_prev = {coeff_prev_correct:.6f}")
    print(f"  coeff_curr = {coeff_curr_correct:.6f}")
    print(f"  coeff_next = {coeff_next_correct:.6f}")
    
    # Check if they match
    if (abs(a_prev_code - coeff_prev_correct) < 1e-10 and
        abs(b_curr_code - coeff_curr_correct) < 1e-10 and
        abs(a_next_code - coeff_next_correct) < 1e-10):
        print("\n✓ Coefficients MATCH the correct formula")
    else:
        print("\n✗ Coefficients DO NOT match!")
        print(f"  Differences:")
        print(f"    a_prev: {a_prev_code - coeff_prev_correct:.2e}")
        print(f"    b_curr: {b_curr_code - coeff_curr_correct:.2e}")
        print(f"    a_next: {a_next_code - coeff_next_correct:.2e}")
    
    # Now let's trace what happens with the alternative sign convention
    # If someone derived for χ'' = -Q·χ (wrong convention), they'd get:
    # (1 + h²Q/12)χ_{n+1} = 2(1 - 5h²Q/12)χ_n - (1 + h²Q/12)χ_{n-1}
    
    print("\n--- Alternative (WRONG for our equation) formula ---")
    coeff_prev_alt = 1.0 + (h_sq / 12.0) * Q  # Note: + instead of -
    coeff_curr_alt = 2.0 - (5.0 * h_sq / 6.0) * Q  # Note: - instead of +
    coeff_next_alt = 1.0 + (h_sq / 12.0) * Q
    
    print(f"  coeff_prev_alt = {coeff_prev_alt:.6f}")
    print(f"  coeff_curr_alt = {coeff_curr_alt:.6f}")
    print(f"  coeff_next_alt = {coeff_next_alt:.6f}")


def test_simple_oscillator():
    """
    Test with simple harmonic oscillator where we know the behavior.
    
    χ'' = -ω²·χ  =>  Q = -ω²
    Solution: χ = A·sin(ωr + φ)
    """
    print("\n" + "=" * 60)
    print("TEST: Simple Harmonic Oscillator (Q = -ω²)")
    print("=" * 60)
    
    omega = 2.0  # frequency
    Q_val = -omega**2  # Q = -ω² (constant)
    
    r_min, r_max = 0.01, 10.0
    n_points = 1000
    r_grid = np.linspace(r_min, r_max, n_points)
    Q_arr = np.full(n_points, Q_val)
    
    # Initial conditions: χ(0) = 0, χ'(0) = ω (gives sin(ωr))
    # But since our grid starts at r_min > 0:
    chi0 = np.sin(omega * r_grid[0])
    chi1 = np.sin(omega * r_grid[1])
    
    chi_num, log_scale = _numerov_propagate(r_grid, Q_arr, chi0, chi1)
    
    # Analytical solution
    chi_exact = np.sin(omega * r_grid)
    
    # Normalize
    idx_mid = len(r_grid) // 2
    if abs(chi_exact[idx_mid]) > 1e-10:
        scale = chi_exact[idx_mid] / chi_num[idx_mid]
        chi_num_scaled = chi_num * scale
    else:
        chi_num_scaled = chi_num
    
    # Compare
    mask = np.abs(chi_exact) > 0.1  # Avoid zeros
    rel_error = np.abs(chi_num_scaled[mask] / chi_exact[mask] - 1)
    max_error = np.max(rel_error)
    mean_error = np.mean(rel_error)
    
    print(f"  ω = {omega}, Q = {Q_val}")
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")
    
    if max_error < 0.01:
        print(f"  ✓ PASS: Numerov correctly solves harmonic oscillator")
    else:
        print(f"  ✗ FAIL: Large errors indicate formula problem!")
        
        # Debug: print first few values
        print(f"\n  First 5 points:")
        for i in range(5):
            print(f"    r={r_grid[i]:.4f}: numerical={chi_num_scaled[i]:.6f}, exact={chi_exact[i]:.6f}")
        print(f"\n  Last 5 points:")
        for i in range(-5, 0):
            print(f"    r={r_grid[i]:.4f}: numerical={chi_num_scaled[i]:.6f}, exact={chi_exact[i]:.6f}")


if __name__ == "__main__":
    test_numerov_formula_derivation()
    test_simple_oscillator()
    test_numerov_free_particle()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
