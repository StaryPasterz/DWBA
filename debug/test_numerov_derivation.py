#!/usr/bin/env python
"""
test_numerov_derivation.py
==========================

Deep investigation of the Numerov formula and where the phase error comes from.

This test directly propagates using both the current formula and the "alternative"
formula to see which one gives correct phases.

Run:
    python debug/test_numerov_derivation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def numerov_propagate_current(r_grid, Q_arr, chi0, chi1):
    """Current implementation: a = 1 - h²Q/12, b = 2 + 5h²Q/6"""
    N = len(r_grid)
    chi = np.zeros(N)
    chi[0] = chi0
    chi[1] = chi1
    
    h_arr = np.diff(r_grid)
    
    for i in range(1, N - 1):
        h1 = h_arr[i - 1]
        h2 = h_arr[i]
        h1_sq = h1 * h1
        h2_sq = h2 * h2
        h_center_sq = h1 * h2
        
        # CURRENT formula
        a_prev = 1.0 - (h1_sq / 12.0) * Q_arr[i - 1]
        b_curr = 2.0 + (5.0 * h_center_sq / 6.0) * Q_arr[i]
        a_next = 1.0 - (h2_sq / 12.0) * Q_arr[i + 1]
        
        if abs(a_next) < 1e-15:
            a_next = 1e-15
            
        chi[i + 1] = (b_curr * chi[i] - a_prev * chi[i - 1]) / a_next
        
        # Renormalize
        if (i + 1) % 200 == 0 and abs(chi[i+1]) > 1e30:
            chi[:i+2] /= chi[i+1]
    
    return chi


def numerov_propagate_alternative(r_grid, Q_arr, chi0, chi1):
    """Alternative formula: a = 1 + h²Q/12, b = 2 - 5h²Q/6 (for χ'' = -Q·χ)"""
    N = len(r_grid)
    chi = np.zeros(N)
    chi[0] = chi0
    chi[1] = chi1
    
    h_arr = np.diff(r_grid)
    
    for i in range(1, N - 1):
        h1 = h_arr[i - 1]
        h2 = h_arr[i]
        h1_sq = h1 * h1
        h2_sq = h2 * h2
        h_center_sq = h1 * h2
        
        # ALTERNATIVE formula (opposite signs)
        a_prev = 1.0 + (h1_sq / 12.0) * Q_arr[i - 1]
        b_curr = 2.0 - (5.0 * h_center_sq / 6.0) * Q_arr[i]
        a_next = 1.0 + (h2_sq / 12.0) * Q_arr[i + 1]
        
        if abs(a_next) < 1e-15:
            a_next = 1e-15
            
        chi[i + 1] = (b_curr * chi[i] - a_prev * chi[i - 1]) / a_next
        
        # Renormalize
        if (i + 1) % 200 == 0 and abs(chi[i+1]) > 1e30:
            chi[:i+2] /= chi[i+1]
    
    return chi


def test_free_particle_both_formulas():
    """Test both formulas against free particle (U=0)."""
    print("=" * 70)
    print("TEST: Current vs Alternative Numerov formula")
    print("=" * 70)
    
    r_min, r_max = 0.01, 100.0
    n_points = 5000
    k_au = 1.5  # More oscillations
    
    # Uniform grid
    r_grid = np.linspace(r_min, r_max, n_points)
    
    for l in [0, 1, 2]:
        print(f"\n--- L = {l} ---")
        
        # Q(r) for free particle: l(l+1)/r² - k²
        Q_arr = l * (l + 1) / (r_grid**2) - k_au**2
        
        # Initial conditions: Bessel
        rho0 = k_au * r_grid[0]
        rho1 = k_au * r_grid[1]
        chi0 = r_grid[0] * spherical_jn(l, rho0)
        chi1 = r_grid[1] * spherical_jn(l, rho1)
        
        # Analytical solution: Riccati-Bessel
        rho = k_au * r_grid
        chi_exact = r_grid * spherical_jn(l, rho)
        
        # Current formula
        chi_curr = numerov_propagate_current(r_grid, Q_arr, chi0, chi1)
        
        # Alternative formula
        chi_alt = numerov_propagate_alternative(r_grid, Q_arr, chi0, chi1)
        
        # Normalize at midpoint
        idx_mid = n_points // 2
        if abs(chi_exact[idx_mid]) > 1e-10:
            chi_curr_norm = chi_curr * (chi_exact[idx_mid] / chi_curr[idx_mid])
            chi_alt_norm = chi_alt * (chi_exact[idx_mid] / chi_alt[idx_mid])
        else:
            chi_curr_norm = chi_curr
            chi_alt_norm = chi_alt
        
        # Compare in tail (last 20%)
        idx_tail = int(0.8 * n_points)
        
        # Extract phases by LSQ fit
        n_fit = 200
        r_fit = r_grid[-n_fit:]
        
        def extract_phase(chi, name):
            chi_fit = chi[-n_fit:]
            phase_free = k_au * r_fit - l * np.pi / 2
            sin_part = np.sin(phase_free)
            cos_part = np.cos(phase_free)
            M = np.vstack([sin_part, cos_part]).T
            coeffs, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
            A_s, A_c = coeffs
            delta = np.arctan2(A_c, A_s)
            print(f"  {name:20s}: δ = {delta:+.4f} rad ({np.degrees(delta):+.1f}°)")
            return delta
        
        # Exact should have δ=0
        delta_exact = extract_phase(chi_exact, "Exact (Bessel)")
        delta_curr = extract_phase(chi_curr_norm, "Current formula")
        delta_alt = extract_phase(chi_alt_norm, "Alternative formula")
        
        # Which is closer to exact?
        err_curr = abs(delta_curr - delta_exact)
        err_alt = abs(delta_alt - delta_exact)
        err_curr = min(err_curr, 2*np.pi - err_curr)
        err_alt = min(err_alt, 2*np.pi - err_alt)
        
        print(f"\n  Error (Current):     {err_curr:.4f} rad")
        print(f"  Error (Alternative): {err_alt:.4f} rad")
        
        if err_curr < err_alt:
            print(f"  → CURRENT formula is better")
        else:
            print(f"  → ALTERNATIVE formula is better")
        
        # Also check wavefunction shape agreement
        mask = np.abs(chi_exact[idx_tail:]) > 0.1
        if np.any(mask):
            rel_err_curr = np.mean(np.abs(chi_curr_norm[idx_tail:][mask] / chi_exact[idx_tail:][mask] - 1))
            rel_err_alt = np.mean(np.abs(chi_alt_norm[idx_tail:][mask] / chi_exact[idx_tail:][mask] - 1))
            print(f"\n  Wavefunction mean rel error (Current):     {rel_err_curr:.2e}")
            print(f"  Wavefunction mean rel error (Alternative): {rel_err_alt:.2e}")


if __name__ == "__main__":
    test_free_particle_both_formulas()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
