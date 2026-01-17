#!/usr/bin/env python
"""
test_johnson_deep.py
====================

Deep diagnostic of Johnson log-derivative solver.

Tests:
1. Formula verification: dY/dr = S - Y² (from Riccati equation)
2. Uniform vs non-uniform grid comparison
3. Simple analytical cases (free particle, SHO)
4. Comparison with direct integration

This will definitively answer: Is Johnson suitable for our case?

Run:
    python debug/test_johnson_deep.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp

print("=" * 70)
print("JOHNSON LOG-DERIVATIVE SOLVER - DEEP DIAGNOSTIC")
print("=" * 70)

# ===========================================================================
# TEST 1: Formula derivation verification
# ===========================================================================
print("\n[TEST 1] FORMULA DERIVATION")
print("-" * 70)

print("""
The radial Schrödinger equation:
    χ''(r) = Q(r)·χ(r)
    
where Q(r) = l(l+1)/r² + 2U(r) - k²

Define log-derivative: Y(r) = χ'(r)/χ(r)

Then:
    χ' = Y·χ
    χ'' = Y'·χ + Y·χ' = Y'·χ + Y²·χ = (Y' + Y²)·χ

Substituting into Schrödinger:
    (Y' + Y²)·χ = Q·χ
    Y' + Y² = Q
    Y' = Q - Y²

THEREFORE: dY/dr = Q - Y² = S - Y²  ✓

This is the Riccati equation for log-derivative propagation.
The original code had: dY/dr = -Y² - S which is WRONG.
""")


# ===========================================================================
# TEST 2: Johnson on UNIFORM grid (should work well)
# ===========================================================================
print("\n[TEST 2] JOHNSON ON UNIFORM GRID")
print("-" * 70)

def johnson_propagate(r_grid, Q_func, Y_init, chi_init):
    """Pure Johnson propagation with RK4 for Y."""
    N = len(r_grid)
    chi = np.zeros(N)
    dchi = np.zeros(N)
    
    chi[0] = chi_init
    dchi[0] = Y_init * chi_init
    Y = Y_init
    
    for i in range(N - 1):
        r = r_grid[i]
        r_next = r_grid[i + 1]
        h = r_next - r
        
        # RK4 for Y: dY/dr = Q(r) - Y²
        def dY(rr, Yval):
            return Q_func(rr) - Yval**2
        
        k1 = dY(r, Y)
        k2 = dY(r + 0.5*h, Y + 0.5*h*k1)
        k3 = dY(r + 0.5*h, Y + 0.5*h*k2)
        k4 = dY(r_next, Y + h*k3)
        
        Y = Y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        Y = np.clip(Y, -500, 500)  # Prevent explosion
        
        # Reconstruct chi: d(log chi)/dr = Y, so chi_new = chi * exp(h * Y_avg)
        Y_avg = 0.5 * (dchi[i]/chi[i] if abs(chi[i]) > 1e-100 else 0) + 0.5 * Y
        chi[i+1] = chi[i] * np.exp(np.clip(h * Y_avg, -50, 50))
        dchi[i+1] = Y * chi[i+1]
        
        # Renormalize periodically
        if (i+1) % 100 == 0 and np.max(np.abs(chi[:i+2])) > 1e10:
            scale = np.max(np.abs(chi[:i+2]))
            chi[:i+2] /= scale
            dchi[:i+2] /= scale
    
    return chi, dchi


def rk45_direct(r_grid, Q_func, chi_init, dchi_init):
    """Direct RK45 integration of χ'' = Q·χ."""
    def rhs(r, y):
        return [y[1], Q_func(r) * y[0]]
    
    sol = solve_ivp(rhs, (r_grid[0], r_grid[-1]), [chi_init, dchi_init],
                    t_eval=r_grid, method='RK45', max_step=0.05)
    
    if sol.success:
        return sol.y[0], sol.y[1]
    return None, None


def extract_phase(r, chi, k, l):
    """Extract phase from asymptotic region."""
    n_fit = min(200, len(r)//5)
    r_fit = r[-n_fit:]
    chi_fit = chi[-n_fit:]
    
    if np.any(np.isnan(chi_fit)) or np.max(np.abs(chi_fit)) < 1e-50:
        return None
    
    phase_arg = k * r_fit - l * np.pi / 2
    M = np.vstack([np.sin(phase_arg), np.cos(phase_arg)]).T
    try:
        c, *_ = np.linalg.lstsq(M, chi_fit, rcond=None)
        return np.arctan2(c[1], c[0])
    except:
        return None


# Free particle on UNIFORM grid
k = 1.5
l = 0
r_uniform = np.linspace(0.01, 100.0, 5000)  # Uniform spacing

Q_free = lambda r: l*(l+1)/r**2 - k**2

# Exact solution
chi_exact = r_uniform * spherical_jn(l, k * r_uniform)

# Initial conditions from exact solution
rho0 = k * r_uniform[0]
jl0 = spherical_jn(l, rho0)
chi0 = r_uniform[0] * jl0
if l == 0:
    dchi0 = np.cos(rho0)
else:
    jlm1 = spherical_jn(l-1, rho0)
    dchi0 = rho0 * jlm1 - l * jl0

Y0 = dchi0 / chi0 if abs(chi0) > 1e-50 else (l+1)/r_uniform[0]

print(f"Free particle: k={k}, L={l}")
print(f"Grid: UNIFORM, {len(r_uniform)} points, r=[{r_uniform[0]:.3f}, {r_uniform[-1]:.1f}]")
print(f"Initial: chi0={chi0:.6f}, dchi0={dchi0:.6f}, Y0={Y0:.4f}")

# Johnson
chi_j, dchi_j = johnson_propagate(r_uniform, Q_free, Y0, chi0)

# RK45
chi_rk, dchi_rk = rk45_direct(r_uniform, Q_free, chi0, dchi0)

# Normalize and extract phases
if chi_j is not None and np.max(np.abs(chi_j)) > 1e-50:
    chi_j *= chi_exact[-100] / chi_j[-100]
    delta_j = extract_phase(r_uniform, chi_j, k, l)
else:
    delta_j = None

if chi_rk is not None:
    chi_rk *= chi_exact[-100] / chi_rk[-100]
    delta_rk = extract_phase(r_uniform, chi_rk, k, l)
else:
    delta_rk = None

delta_ex = extract_phase(r_uniform, chi_exact, k, l)

print(f"\nPhase shifts:")
print(f"  Exact:   {delta_ex:+.6f} rad" if delta_ex else "  Exact:   FAILED")
print(f"  Johnson: {delta_j:+.6f} rad" if delta_j else "  Johnson: FAILED")
print(f"  RK45:    {delta_rk:+.6f} rad" if delta_rk else "  RK45:    FAILED")

if delta_j and delta_ex:
    err_j = abs(delta_j - delta_ex)
    err_j = min(err_j, 2*np.pi - err_j)
    print(f"\n  Johnson error: {err_j:.6f} rad", "✓" if err_j < 0.01 else "✗")


# ===========================================================================
# TEST 3: Johnson on EXPONENTIAL grid (our actual case)
# ===========================================================================
print("\n\n[TEST 3] JOHNSON ON EXPONENTIAL GRID")
print("-" * 70)

r_min, r_max, n_pts = 0.01, 100.0, 3000
scale = np.log(r_max / r_min)
xi = np.linspace(0, 1, n_pts)
r_exp = r_min * np.exp(scale * xi)

print(f"Grid: EXPONENTIAL, {len(r_exp)} points")
print(f"  r = [{r_exp[0]:.4f}, ..., {r_exp[-1]:.2f}]")
print(f"  h ranges from {r_exp[1]-r_exp[0]:.6f} to {r_exp[-1]-r_exp[-2]:.4f}")

# Same IC
rho0 = k * r_exp[0]
jl0 = spherical_jn(l, rho0)
chi0 = r_exp[0] * jl0
if l == 0:
    dchi0 = np.cos(rho0)
else:
    jlm1 = spherical_jn(l-1, rho0)
    dchi0 = rho0 * jlm1 - l * jl0

Y0 = dchi0 / chi0 if abs(chi0) > 1e-50 else (l+1)/r_exp[0]

# Johnson
chi_j_exp, _ = johnson_propagate(r_exp, Q_free, Y0, chi0)

# RK45
chi_rk_exp, _ = rk45_direct(r_exp, Q_free, chi0, dchi0)

# Exact
chi_exact_exp = r_exp * spherical_jn(l, k * r_exp)

# Normalize and extract phases
if chi_j_exp is not None and np.max(np.abs(chi_j_exp)) > 1e-50:
    chi_j_exp *= chi_exact_exp[-100] / chi_j_exp[-100]
    delta_j_exp = extract_phase(r_exp, chi_j_exp, k, l)
else:
    delta_j_exp = None

if chi_rk_exp is not None:
    chi_rk_exp *= chi_exact_exp[-100] / chi_rk_exp[-100]
    delta_rk_exp = extract_phase(r_exp, chi_rk_exp, k, l)
else:
    delta_rk_exp = None

delta_ex_exp = extract_phase(r_exp, chi_exact_exp, k, l)

print(f"\nPhase shifts:")
print(f"  Exact:   {delta_ex_exp:+.6f} rad" if delta_ex_exp else "  Exact:   FAILED")
print(f"  Johnson: {delta_j_exp:+.6f} rad" if delta_j_exp else "  Johnson: FAILED")
print(f"  RK45:    {delta_rk_exp:+.6f} rad" if delta_rk_exp else "  RK45:    FAILED")

if delta_j_exp and delta_ex_exp:
    err_j_exp = abs(delta_j_exp - delta_ex_exp)
    err_j_exp = min(err_j_exp, 2*np.pi - err_j_exp)
    status = "✓" if err_j_exp < 0.01 else "✗"
    print(f"\n  Johnson error: {err_j_exp:.6f} rad {status}")
    
if delta_rk_exp and delta_ex_exp:
    err_rk_exp = abs(delta_rk_exp - delta_ex_exp)
    err_rk_exp = min(err_rk_exp, 2*np.pi - err_rk_exp)
    status = "✓" if err_rk_exp < 0.01 else "✗"
    print(f"  RK45 error:    {err_rk_exp:.6f} rad {status}")


# ===========================================================================
# TEST 4: Diagnose the reconstruction step
# ===========================================================================
print("\n\n[TEST 4] DIAGNOSE RECONSTRUCTION: chi from Y")
print("-" * 70)

print("""
The Johnson method propagates Y = chi'/chi.
Then reconstructs chi using: chi(r+h) = chi(r) * exp(∫Y dr)

The issue: exp(h * Y_avg) assumes Y is approximately constant over [r, r+h].
On exponential grids with large h at large r, this may introduce errors.

Alternative: Instead of using Y_avg, integrate properly or use smaller steps.
""")

# Let's check how Y varies over one step at different positions
print("Y variation over one step (exponential grid):")
for i in [10, 100, 1000, 2000]:
    if i+1 < len(r_exp):
        r_i = r_exp[i]
        h = r_exp[i+1] - r_exp[i]
        # For free particle, Y ≈ k * cot(k*r) at large r
        Y_at_r = k / np.tan(k * r_i) if abs(np.sin(k*r_i)) > 0.1 else 0
        Y_at_r_next = k / np.tan(k * r_exp[i+1]) if abs(np.sin(k*r_exp[i+1])) > 0.1 else 0
        delta_Y = abs(Y_at_r_next - Y_at_r)
        print(f"  i={i:4d}: r={r_i:8.4f}, h={h:.6f}, |ΔY|={delta_Y:.4f}")


# ===========================================================================
# CONCLUSION
# ===========================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print("""
The Johnson log-derivative method has the following characteristics:

1. The CORRECT formula is: dY/dr = Q(r) - Y²
   (NOT -Y² - Q as in original code)

2. On UNIFORM grids: Johnson should work well because:
   - Fixed step size h
   - exp(h * Y_avg) is accurate when Y varies slowly over h
   
3. On EXPONENTIAL grids: Johnson may have issues because:
   - Step size h varies by ~4 orders of magnitude
   - At large r, h is large and Y oscillates rapidly
   - The reconstruction chi = chi*exp(h*Y_avg) becomes inaccurate

4. RK45 with adaptive stepping handles non-uniform grids better because:
   - It integrates χ and χ' directly (not Y)
   - Adaptive step control ensures accuracy

RECOMMENDATION:
- Use RK45 as the primary/default solver for exponential grids
- Johnson may be useful for uniform grids or as fallback for high-L
- Consider reformulating Johnson's reconstruction for non-uniform grids
""")

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
