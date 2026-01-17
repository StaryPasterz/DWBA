#!/usr/bin/env python
"""
test_johnson_solver.py
======================

Deep analysis of Johnson log-derivative solver implementation.

Verifies:
1. Correct formula derivation: dY/dr = -Y² + S(r) or dY/dr = -Y² - S(r)?
2. Comparison with RK45 on same energy/L
3. Free particle test (exact solution known)

Run:
    python debug/test_johnson_solver.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp


def johnson_solve_current(r_grid, Q_arr, l, k_au):
    """Current implementation from continuum.py (dY/dr = -Y² - S)"""
    N = len(r_grid)
    k2 = k_au ** 2
    ell = float(l)
    
    chi = np.zeros(N, dtype=float)
    dchi = np.zeros(N, dtype=float)
    
    r0 = r_grid[0]
    S0 = Q_arr[0]  # S = Q
    
    if S0 > 0:
        Y_init = np.sqrt(S0)
    else:
        Y_init = (ell + 1.0) / r0
    
    chi[0] = 1.0
    dchi[0] = Y_init * chi[0]
    Y_current = Y_init
    
    for i in range(N - 1):
        r = r_grid[i]
        r_next = r_grid[i + 1]
        h = r_next - r
        
        # S(r) interpolation
        def S_func(rr, idx_hint):
            if idx_hint < N - 1:
                t = (rr - r_grid[idx_hint]) / (r_grid[idx_hint + 1] - r_grid[idx_hint])
                S_interp = Q_arr[idx_hint] * (1 - t) + Q_arr[idx_hint + 1] * t
            else:
                S_interp = Q_arr[-1]
            return S_interp
        
        # CURRENT: dY/dr = -Y² - S
        def dY(rr, Y_val, idx):
            return -Y_val**2 - S_func(rr, idx)
        
        k1 = dY(r, Y_current, i)
        k2_rk = dY(r + 0.5*h, Y_current + 0.5*h*k1, i)
        k3 = dY(r + 0.5*h, Y_current + 0.5*h*k2_rk, i)
        k4 = dY(r_next, Y_current + h*k3, i)
        
        Y_next = Y_current + (h/6.0) * (k1 + 2*k2_rk + 2*k3 + k4)
        Y_next = np.clip(Y_next, -100.0, 100.0)
        
        Y_avg = 0.5 * (Y_current + Y_next)
        exp_arg = np.clip(h * Y_avg, -30, 30)
        
        chi[i + 1] = chi[i] * np.exp(exp_arg)
        dchi[i + 1] = Y_next * chi[i + 1]
        Y_current = Y_next
        
        if (i + 1) % 50 == 0:
            max_val = np.max(np.abs(chi[:i + 2]))
            if max_val > 1e5:
                chi[:i + 2] /= max_val
                dchi[:i + 2] /= max_val
    
    max_val = np.max(np.abs(chi))
    if max_val > 0:
        chi /= max_val
        dchi /= max_val
    
    return chi, dchi


def johnson_solve_alternative(r_grid, Q_arr, l, k_au):
    """Alternative formula: dY/dr = -Y² + S (note: +S instead of -S)"""
    N = len(r_grid)
    k2 = k_au ** 2
    ell = float(l)
    
    chi = np.zeros(N, dtype=float)
    dchi = np.zeros(N, dtype=float)
    
    r0 = r_grid[0]
    S0 = Q_arr[0]
    
    if S0 > 0:
        Y_init = np.sqrt(S0)
    else:
        Y_init = (ell + 1.0) / r0
    
    chi[0] = 1.0
    dchi[0] = Y_init * chi[0]
    Y_current = Y_init
    
    for i in range(N - 1):
        r = r_grid[i]
        r_next = r_grid[i + 1]
        h = r_next - r
        
        def S_func(rr, idx_hint):
            if idx_hint < N - 1:
                t = (rr - r_grid[idx_hint]) / (r_grid[idx_hint + 1] - r_grid[idx_hint])
                S_interp = Q_arr[idx_hint] * (1 - t) + Q_arr[idx_hint + 1] * t
            else:
                S_interp = Q_arr[-1]
            return S_interp
        
        # ALTERNATIVE: dY/dr = -Y² + S
        def dY(rr, Y_val, idx):
            return -Y_val**2 + S_func(rr, idx)  # Changed sign!
        
        k1 = dY(r, Y_current, i)
        k2_rk = dY(r + 0.5*h, Y_current + 0.5*h*k1, i)
        k3 = dY(r + 0.5*h, Y_current + 0.5*h*k2_rk, i)
        k4 = dY(r_next, Y_current + h*k3, i)
        
        Y_next = Y_current + (h/6.0) * (k1 + 2*k2_rk + 2*k3 + k4)
        Y_next = np.clip(Y_next, -100.0, 100.0)
        
        Y_avg = 0.5 * (Y_current + Y_next)
        exp_arg = np.clip(h * Y_avg, -30, 30)
        
        chi[i + 1] = chi[i] * np.exp(exp_arg)
        dchi[i + 1] = Y_next * chi[i + 1]
        Y_current = Y_next
        
        if (i + 1) % 50 == 0:
            max_val = np.max(np.abs(chi[:i + 2]))
            if max_val > 1e5:
                chi[:i + 2] /= max_val
                dchi[:i + 2] /= max_val
    
    max_val = np.max(np.abs(chi))
    if max_val > 0:
        chi /= max_val
        dchi /= max_val
    
    return chi, dchi


def rk45_solve_direct(r_grid, Q_arr, l, k_au):
    """Direct RK45 solve of χ'' = Q·χ for reference."""
    from scipy.interpolate import CubicSpline
    
    Q_spline = CubicSpline(r_grid, Q_arr)
    
    def rhs(r, y):
        chi, dchi = y
        Q = float(Q_spline(r))
        return [dchi, Q * chi]
    
    # Initial: χ ~ r^(l+1) for small r
    r0 = r_grid[0]
    rho0 = k_au * r0
    chi0 = r0 * spherical_jn(l, rho0) if rho0 > 1e-10 else r0**(l + 1)
    
    r1 = r_grid[1]
    rho1 = k_au * r1
    chi1 = r1 * spherical_jn(l, rho1) if rho1 > 1e-10 else r1**(l + 1)
    
    dchi0 = (chi1 - chi0) / (r1 - r0)
    
    sol = solve_ivp(
        rhs, (r_grid[0], r_grid[-1]), [chi0, dchi0],
        t_eval=r_grid, method="RK45", max_step=0.1
    )
    
    if sol.success:
        return sol.y[0], sol.y[1]
    else:
        return None, None


def test_johnson_derivation():
    """
    Verify Johnson formula by checking the log-derivative equation.
    
    Theory:
    χ'' = Q·χ
    Y = χ'/χ
    
    d/dr[Y] = d/dr[χ'/χ] = (χ''·χ - χ'·χ') / χ² = χ''/χ - (χ'/χ)² = Q - Y²
    
    So the correct equation is:
        dY/dr = Q - Y² = S - Y²  (where S = Q)
    
    NOT:
        dY/dr = -Y² - S
    """
    print("=" * 70)
    print("JOHNSON FORMULA DERIVATION CHECK")
    print("=" * 70)
    
    print("""
    Given: χ'' = Q·χ  (where Q = l(l+1)/r² + 2U - k²)
    
    Define: Y = χ'/χ (log-derivative)
    
    Then: χ' = Y·χ
          χ'' = Y'·χ + Y·χ' = Y'·χ + Y²·χ = (Y' + Y²)·χ
    
    Substituting into χ'' = Q·χ:
          (Y' + Y²)·χ = Q·χ
          Y' + Y² = Q
          Y' = Q - Y²
    
    Therefore: dY/dr = Q - Y² = S - Y²  (NOT -Y² - S)
    
    The current implementation has:
        dY/dr = -Y² - S = -(Y² + S)
    
    This is WRONG by a sign! It should be:
        dY/dr = -Y² + S = S - Y²
    """)


def test_free_particle():
    """Compare both Johnson formulas against free particle."""
    print("\n" + "=" * 70)
    print("FREE PARTICLE TEST: Johnson Current vs Alternative vs RK45")
    print("=" * 70)
    
    # Use uniform grid for fair comparison
    r_min, r_max = 0.01, 100.0
    n_points = 2000
    r_grid = np.linspace(r_min, r_max, n_points)
    k_au = 1.5
    
    for l in [0, 1, 2]:
        print(f"\n--- L = {l} ---")
        
        # Q = l(l+1)/r² - k² (free particle)
        Q_arr = l * (l + 1) / r_grid**2 - k_au**2
        
        # Exact solution
        chi_exact = r_grid * spherical_jn(l, k_au * r_grid)
        
        # Johnson current (dY = -Y² - S)
        chi_curr, _ = johnson_solve_current(r_grid, Q_arr, l, k_au)
        
        # Johnson alternative (dY = -Y² + S)
        chi_alt, _ = johnson_solve_alternative(r_grid, Q_arr, l, k_au)
        
        # RK45 reference
        chi_rk, _ = rk45_solve_direct(r_grid, Q_arr, l, k_au)
        
        # Normalize at midpoint
        idx = n_points // 2
        if abs(chi_exact[idx]) > 1e-10:
            chi_curr *= chi_exact[idx] / chi_curr[idx] if abs(chi_curr[idx]) > 1e-100 else 1
            chi_alt *= chi_exact[idx] / chi_alt[idx] if abs(chi_alt[idx]) > 1e-100 else 1
            if chi_rk is not None:
                chi_rk *= chi_exact[idx] / chi_rk[idx] if abs(chi_rk[idx]) > 1e-100 else 1
        
        # Extract phases
        n_fit = 200
        r_fit = r_grid[-n_fit:]
        
        def extract_phase(chi, name):
            if chi is None or np.all(np.isnan(chi)) or np.max(np.abs(chi)) < 1e-100:
                print(f"  {name:25s}: FAILED/NaN")
                return None
            chi_fit = chi[-n_fit:]
            if np.any(np.isnan(chi_fit)) or np.all(np.abs(chi_fit) < 1e-100):
                print(f"  {name:25s}: FAILED (tail issues)")
                return None
            phase_free = k_au * r_fit - l * np.pi / 2
            sin_part = np.sin(phase_free)
            cos_part = np.cos(phase_free)
            M = np.vstack([sin_part, cos_part]).T
            try:
                coeffs, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
                A_s, A_c = coeffs
                delta = np.arctan2(A_c, A_s)
                print(f"  {name:25s}: δ = {delta:+.4f} rad")
                return delta
            except:
                print(f"  {name:25s}: LSQ failed")
                return None
        
        delta_exact = extract_phase(chi_exact, "Exact (Bessel)")
        delta_curr = extract_phase(chi_curr, "Johnson Current (-Y²-S)")
        delta_alt = extract_phase(chi_alt, "Johnson Alt (-Y²+S)")
        delta_rk = extract_phase(chi_rk, "RK45 (reference)")
        
        # Which is better?
        if delta_exact is not None and delta_curr is not None and delta_alt is not None:
            err_curr = abs(delta_curr - delta_exact)
            err_alt = abs(delta_alt - delta_exact)
            err_curr = min(err_curr, 2*np.pi - err_curr)
            err_alt = min(err_alt, 2*np.pi - err_alt)
            
            print(f"\n  Error Current:     {err_curr:.4f} rad")
            print(f"  Error Alternative: {err_alt:.4f} rad")
            if err_alt < err_curr:
                print("  → ALTERNATIVE formula (-Y²+S) is CORRECT!")
            else:
                print("  → CURRENT formula might be correct (unexpected)")


def test_with_potential():
    """Test with H atom potential."""
    print("\n" + "=" * 70)
    print("H ATOM TEST: Johnson vs RK45")
    print("=" * 70)
    
    from grid import make_r_grid, k_from_E_eV
    from potential_core import CorePotentialParams, V_core_on_grid
    from bound_states import solve_bound_states
    from distorting_potential import build_distorting_potentials
    from continuum import solve_continuum_wave
    
    grid = make_r_grid(r_max=200.0, n_points=3000)
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, params)
    
    orbs = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    orb_1s = orbs[0]
    orb_2s = orbs[1]
    
    dE = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
    print(f"Threshold: {dE:.2f} eV")
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, 
                                            k_from_E_eV(50.0), k_from_E_eV(50.0 - dE))
    
    for E_eV in [50.0, 20.0]:
        print(f"\n--- E = {E_eV} eV ---")
        
        for l in [0, 2, 5]:
            print(f"\n  L = {l}:")
            
            try:
                cw_johnson = solve_continuum_wave(grid, U_i, l, E_eV, z_ion=0.0, solver="johnson")
                cw_rk45 = solve_continuum_wave(grid, U_i, l, E_eV, z_ion=0.0, solver="rk45")
                
                print(f"    Johnson: δ = {cw_johnson.phase_shift:+.4f} rad, method={cw_johnson.phase_method}")
                print(f"    RK45:    δ = {cw_rk45.phase_shift:+.4f} rad, method={cw_rk45.phase_method}")
                
                diff = abs(cw_johnson.phase_shift - cw_rk45.phase_shift)
                diff = min(diff, 2*np.pi - diff)
                status = "✓" if diff < 0.1 else "✗"
                print(f"    Difference: {diff:.4f} rad {status}")
                
            except Exception as e:
                print(f"    ERROR: {e}")


if __name__ == "__main__":
    test_johnson_derivation()
    test_free_particle()
    test_with_potential()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
