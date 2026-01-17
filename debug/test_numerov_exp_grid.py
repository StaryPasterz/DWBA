#!/usr/bin/env python
"""
test_numerov_exp_grid.py
========================

Test whether the issue is with exponential (non-uniform) grid handling.

Run:
    python debug/test_numerov_exp_grid.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


def make_exp_grid(r_min, r_max, n_points):
    """Create exponential grid like the main code uses."""
    scale = np.log(r_max / r_min)
    xi = np.linspace(0, 1, n_points)
    r = r_min * np.exp(scale * xi)
    return r


def numerov_propagate(r_grid, Q_arr, chi0, chi1):
    """Numerov with non-uniform grid handling (current code implementation)."""
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
        h_center_sq = h1 * h2  # Geometric mean for center term
        
        a_prev = 1.0 - (h1_sq / 12.0) * Q_arr[i - 1]
        b_curr = 2.0 + (5.0 * h_center_sq / 6.0) * Q_arr[i]
        a_next = 1.0 - (h2_sq / 12.0) * Q_arr[i + 1]
        
        if abs(a_next) < 1e-15:
            a_next = 1e-15
            
        chi[i + 1] = (b_curr * chi[i] - a_prev * chi[i - 1]) / a_next
        
        # Renormalize
        if (i + 1) % 200 == 0:
            max_val = max(abs(chi[i]), abs(chi[i+1]))
            if max_val > 1e50:
                chi[:i+2] /= max_val
    
    return chi


def rk45_solve(r_grid, Q_arr, chi0, chi1, k_au, l):
    """RK45 solver for comparison."""
    # Build spline of Q for interpolation
    Q_spline = CubicSpline(r_grid, Q_arr)
    
    def rhs(r, y):
        chi = y[0]
        dchi = y[1]
        Q_val = float(Q_spline(r))
        return np.array([dchi, Q_val * chi])
    
    # Initial derivative from Bessel
    h = r_grid[1] - r_grid[0]
    dchi0 = (chi1 - chi0) / h  # Simple approximation
    
    sol = solve_ivp(
        fun=rhs,
        t_span=(r_grid[0], r_grid[-1]),
        y0=[chi0, dchi0],
        t_eval=r_grid,
        method="RK45",
        max_step=0.1
    )
    
    if sol.success:
        return sol.y[0]
    else:
        return None


def test_exponential_grid():
    """Compare Numerov vs RK45 on exponential grid for free particle."""
    print("=" * 70)
    print("TEST: Numerov vs RK45 on EXPONENTIAL grid (free particle)")
    print("=" * 70)
    
    r_min, r_max = 0.01, 200.0
    n_points = 3000
    k_au = 1.5
    
    # Create exponential grid (like main code)
    r_grid = make_exp_grid(r_min, r_max, n_points)
    
    print(f"\nGrid: exponential, r_min={r_min}, r_max={r_max}, N={n_points}")
    print(f"Step sizes: h_min={np.diff(r_grid)[0]:.4f}, h_max={np.diff(r_grid)[-1]:.4f}")
    print(f"k = {k_au} a.u.\n")
    
    for l in [0, 1, 2, 5]:
        print(f"\n--- L = {l} ---")
        
        # Q for free particle
        Q_arr = l * (l + 1) / (r_grid**2) - k_au**2
        
        # Initial conditions: Bessel
        rho0 = k_au * r_grid[0]
        rho1 = k_au * r_grid[1]
        chi0 = r_grid[0] * spherical_jn(l, rho0)
        chi1 = r_grid[1] * spherical_jn(l, rho1)
        
        # Analytical
        chi_exact = r_grid * spherical_jn(l, k_au * r_grid)
        
        # Numerov
        chi_num = numerov_propagate(r_grid, Q_arr, chi0, chi1)
        
        # RK45
        chi_rk = rk45_solve(r_grid, Q_arr, chi0, chi1, k_au, l)
        
        # Normalize at midpoint
        idx_mid = n_points // 2
        if abs(chi_exact[idx_mid]) > 1e-10:
            chi_num *= chi_exact[idx_mid] / chi_num[idx_mid]
            if chi_rk is not None:
                chi_rk *= chi_exact[idx_mid] / chi_rk[idx_mid]
        
        # Extract phases from tail
        n_fit = 300
        r_fit = r_grid[-n_fit:]
        
        def extract_phase(chi, name):
            if chi is None:
                print(f"  {name:20s}: FAILED")
                return None
            chi_fit = chi[-n_fit:]
            phase_free = k_au * r_fit - l * np.pi / 2
            sin_part = np.sin(phase_free)
            cos_part = np.cos(phase_free)
            M = np.vstack([sin_part, cos_part]).T
            coeffs, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
            A_s, A_c = coeffs
            delta = np.arctan2(A_c, A_s)
            print(f"  {name:20s}: δ = {delta:+.4f} rad")
            return delta
        
        delta_exact = extract_phase(chi_exact, "Exact (Bessel)")
        delta_num = extract_phase(chi_num, "Numerov")
        delta_rk = extract_phase(chi_rk, "RK45")
        
        if delta_num is not None and delta_exact is not None:
            err_num = abs(delta_num - delta_exact)
            err_num = min(err_num, 2*np.pi - err_num)
            print(f"\n  Numerov error: {err_num:.4f} rad ({err_num > 0.1 and '✗ BAD' or '✓ OK'})")
        
        if delta_rk is not None and delta_exact is not None:
            err_rk = abs(delta_rk - delta_exact)
            err_rk = min(err_rk, 2*np.pi - err_rk)
            print(f"  RK45 error:    {err_rk:.4f} rad ({err_rk > 0.1 and '✗ BAD' or '✓ OK'})")


def test_with_potential():
    """Test with actual potential (H atom)."""
    print("\n\n" + "=" * 70)
    print("TEST: Numerov vs RK45 with H atom potential")
    print("=" * 70)
    
    from grid import make_r_grid
    from potential_core import CorePotentialParams, V_core_on_grid
    from bound_states import solve_bound_states
    from distorting_potential import build_distorting_potentials
    
    r_max = 200.0
    n_points = 3000
    grid = make_r_grid(r_max=r_max, n_points=n_points)
    
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, params)
    
    orbs = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    orb_1s = orbs[0]
    orb_2s = orbs[1]
    
    dE = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
    E_eV = 50.0
    k_i = np.sqrt(2 * E_eV / 27.211)
    k_f = np.sqrt(2 * (E_eV - dE) / 27.211)
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
    
    print(f"\nH atom at E = {E_eV} eV, k = {k_i:.4f} a.u.")
    
    for l in [0, 1, 2]:
        print(f"\n--- L = {l} ---")
        
        ell = float(l)
        k2 = k_i * k_i
        
        # Q with potential
        Q_arr = ell * (ell + 1.0) / (grid.r**2) + 2.0 * U_i.U_of_r - k2
        
        # Initial conditions
        rho0 = k_i * grid.r[0]
        rho1 = k_i * grid.r[1]
        chi0 = grid.r[0] * spherical_jn(l, rho0)
        chi1 = grid.r[1] * spherical_jn(l, rho1)
        
        # Numerov
        chi_num = numerov_propagate(grid.r, Q_arr, chi0, chi1)
        
        # RK45
        chi_rk = rk45_solve(grid.r, Q_arr, chi0, chi1, k_i, l)
        
        # Normalize
        idx_mid = len(grid.r) // 2
        chi_num_normed = chi_num / chi_num[idx_mid] if abs(chi_num[idx_mid]) > 1e-100 else chi_num
        chi_rk_normed = chi_rk / chi_rk[idx_mid] if chi_rk is not None and abs(chi_rk[idx_mid]) > 1e-100 else chi_rk
        
        # Extract phases
        n_fit = 300
        r_fit = grid.r[-n_fit:]
        
        def extract_phase(chi, name):
            if chi is None:
                print(f"  {name:20s}: FAILED")
                return None
            chi_fit = chi[-n_fit:]
            phase_free = k_i * r_fit - l * np.pi / 2
            sin_part = np.sin(phase_free)
            cos_part = np.cos(phase_free)
            M = np.vstack([sin_part, cos_part]).T
            coeffs, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
            A_s, A_c = coeffs
            delta = np.arctan2(A_c, A_s)
            print(f"  {name:20s}: δ = {delta:+.4f} rad")
            return delta
        
        delta_num = extract_phase(chi_num_normed, "Numerov")
        delta_rk = extract_phase(chi_rk_normed, "RK45")
        
        if delta_num is not None and delta_rk is not None:
            diff = abs(delta_num - delta_rk)
            diff = min(diff, 2*np.pi - diff)
            print(f"\n  Difference: {diff:.4f} rad ({diff > 0.1 and '✗ SIGNIFICANT' or '✓ OK'})")


if __name__ == "__main__":
    test_exponential_grid()
    test_with_potential()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
