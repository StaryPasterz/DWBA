#!/usr/bin/env python
"""
test_rk45_verification.py
=========================

Comprehensive verification of RK45 solver against:
1. Free particle (exact analytical solution)
2. Hydrogen atom (known phase shifts from literature)
3. Various energy ranges

This will answer: Is RK45 giving CORRECT results?

Run:
    python debug/test_rk45_verification.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

print("=" * 70)
print("RK45 SOLVER VERIFICATION - IS IT CORRECT?")
print("=" * 70)


# ===========================================================================
# TEST 1: Free particle (EXACT analytical solution exists)
# ===========================================================================
print("\n[TEST 1] FREE PARTICLE - EXACT SOLUTION COMPARISON")
print("-" * 70)

def test_free_particle():
    """
    For free particle: χ'' = Q·χ where Q = l(l+1)/r² - k²
    Exact solution: χ(r) = r·j_l(kr) (regular) or r·n_l(kr) (irregular)
    Phase shift should be exactly ZERO for free particle.
    """
    
    # Grid parameters
    r_min, r_max, n_pts = 0.01, 100.0, 3000
    
    # Exponential grid (our actual case)
    scale = np.log(r_max / r_min)
    xi = np.linspace(0, 1, n_pts)
    r_exp = r_min * np.exp(scale * xi)
    
    # Uniform grid for comparison
    r_uniform = np.linspace(r_min, r_max, n_pts)
    
    results = []
    
    for L in [0, 1, 2, 5, 10]:
        for k in [0.5, 1.0, 2.0, 5.0]:  # Different momenta
            for grid_name, r_grid in [("exponential", r_exp), ("uniform", r_uniform)]:
                
                # Exact solution
                chi_exact = r_grid * spherical_jn(L, k * r_grid)
                
                # RK45 solution
                Q_arr = L * (L + 1) / r_grid**2 - k**2
                Q_spline = CubicSpline(r_grid, Q_arr)
                
                def rhs(r, y):
                    return [y[1], Q_spline(r) * y[0]]
                
                # Initial conditions from exact solution
                rho0 = k * r_grid[0]
                chi0 = r_grid[0] * spherical_jn(L, rho0)
                if L == 0:
                    dchi0 = np.cos(rho0)
                else:
                    jl_m1 = spherical_jn(L - 1, rho0)
                    jl = spherical_jn(L, rho0)
                    dchi0 = rho0 * jl_m1 - L * jl
                
                sol = solve_ivp(rhs, (r_grid[0], r_grid[-1]), [chi0, dchi0],
                               t_eval=r_grid, method='RK45', max_step=0.1)
                
                if not sol.success:
                    results.append((L, k, grid_name, None, None, "FAILED"))
                    continue
                
                chi_rk = sol.y[0]
                
                # Normalize at midpoint
                idx_mid = len(r_grid) // 2
                if abs(chi_exact[idx_mid]) > 1e-20:
                    chi_rk *= chi_exact[idx_mid] / chi_rk[idx_mid]
                
                # Compare phases in asymptotic region
                n_fit = 200
                r_fit = r_grid[-n_fit:]
                phase_arg = k * r_fit - L * np.pi / 2
                M = np.vstack([np.sin(phase_arg), np.cos(phase_arg)]).T
                
                try:
                    c_rk, *_ = np.linalg.lstsq(M, chi_rk[-n_fit:], rcond=None)
                    delta_rk = np.arctan2(c_rk[1], c_rk[0])
                    
                    c_ex, *_ = np.linalg.lstsq(M, chi_exact[-n_fit:], rcond=None)
                    delta_ex = np.arctan2(c_ex[1], c_ex[0])
                    
                    diff = abs(delta_rk - delta_ex)
                    diff = min(diff, 2*np.pi - diff)
                    
                    results.append((L, k, grid_name, delta_ex, delta_rk, diff))
                except:
                    results.append((L, k, grid_name, None, None, "LSQ_FAILED"))
    
    # Print results
    print(f"{'L':>3} {'k':>5} {'Grid':>12} {'δ_exact':>10} {'δ_RK45':>10} {'Error':>10} Status")
    print("-" * 70)
    
    for L, k, grid, d_ex, d_rk, diff in results:
        if isinstance(diff, str):
            print(f"{L:>3} {k:>5.1f} {grid:>12}  {'---':>10} {'---':>10} {diff:>10}")
        else:
            status = "✓ OK" if diff < 0.01 else "✗" if diff < 0.1 else "✗✗"
            d_ex_str = f"{d_ex:+.4f}" if d_ex is not None else "---"
            d_rk_str = f"{d_rk:+.4f}" if d_rk is not None else "---"
            print(f"{L:>3} {k:>5.1f} {grid:>12} {d_ex_str:>10} {d_rk_str:>10} {diff:>10.6f} {status}")
    
    # Summary
    exp_errors = [d for L, k, g, _, _, d in results if g == "exponential" and isinstance(d, float)]
    uni_errors = [d for L, k, g, _, _, d in results if g == "uniform" and isinstance(d, float)]
    
    print("\nSummary:")
    if exp_errors:
        print(f"  Exponential grid: max error = {max(exp_errors):.6f} rad, mean = {np.mean(exp_errors):.6f} rad")
    if uni_errors:
        print(f"  Uniform grid:     max error = {max(uni_errors):.6f} rad, mean = {np.mean(uni_errors):.6f} rad")

test_free_particle()


# ===========================================================================
# TEST 2: Comparison with our actual continuum solver
# ===========================================================================
print("\n\n[TEST 2] DWBA CONTINUUM SOLVER - ENERGY SCAN")
print("-" * 70)

def test_energy_scan():
    """Test RK45 at various energies for H atom."""
    from grid import make_r_grid
    from potential_core import CorePotentialParams, V_core_on_grid
    from bound_states import solve_bound_states
    from distorting_potential import build_distorting_potentials
    from continuum import solve_continuum_wave
    from grid import k_from_E_eV
    
    # Setup
    grid = make_r_grid(r_max=200.0, n_points=3000)
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, params)
    orbs = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    orb_1s, orb_2s = orbs[0], orbs[1]
    
    print("H atom: 1s → 2s excitation")
    print(f"Threshold: {(orb_2s.energy_au - orb_1s.energy_au) * 27.211:.2f} eV\n")
    
    # Energy scan
    energies = [15, 20, 30, 50, 100, 200, 500, 1000]
    
    print(f"{'E (eV)':>8} {'L':>3} {'δ_RK45':>10} {'δ_Numerov':>10} {'Diff':>10} Status")
    print("-" * 55)
    
    for E in energies:
        k_i = k_from_E_eV(E)
        threshold = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
        if E < threshold:
            continue
            
        E_final = E - threshold
        k_f = k_from_E_eV(E_final)
        
        U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
        
        for L in [0, 5, 10]:
            try:
                cw_rk = solve_continuum_wave(grid, U_i, L, E, z_ion=0.0, solver="rk45")
                cw_num = solve_continuum_wave(grid, U_i, L, E, z_ion=0.0, solver="numerov")
                
                diff = abs(cw_rk.phase_shift - cw_num.phase_shift)
                diff = min(diff, 2*np.pi - diff)
                
                status = "✓" if diff < 0.1 else "✗"
                print(f"{E:>8} {L:>3} {cw_rk.phase_shift:>+10.4f} {cw_num.phase_shift:>+10.4f} {diff:>10.4f} {status}")
            except Exception as e:
                print(f"{E:>8} {L:>3} ERROR: {e}")

test_energy_scan()


# ===========================================================================
# TEST 3: Phase shift consistency check
# ===========================================================================
print("\n\n[TEST 3] RK45 PHASE SHIFT STABILITY VS GRID DENSITY")
print("-" * 70)

def test_grid_convergence():
    """Check if RK45 phase converges as grid density increases."""
    from grid import k_from_E_eV
    from potential_core import CorePotentialParams, V_core_on_grid
    from bound_states import solve_bound_states
    from distorting_potential import build_distorting_potentials
    from continuum import solve_continuum_wave
    
    E = 50.0
    L = 0
    
    print(f"Testing grid convergence for E={E} eV, L={L}")
    print(f"{'N points':>10} {'r_max':>8} {'δ_RK45':>12} {'Δδ':>12}")
    print("-" * 45)
    
    prev_delta = None
    
    for n_pts in [1000, 2000, 3000, 4000, 5000]:
        r_max = 200.0
        scale = np.log(r_max / 0.01)
        xi = np.linspace(0, 1, n_pts)
        r = 0.01 * np.exp(scale * xi)
        
        params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
        V_core = V_core_on_grid(r, params)
        orbs = solve_bound_states(r, V_core, l=0, n_states_max=2)
        orb_1s, orb_2s = orbs[0], orbs[1]
        
        k_i = k_from_E_eV(E)
        threshold = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
        k_f = k_from_E_eV(E - threshold)
        
        U_i, U_f = build_distorting_potentials(r, V_core, orb_1s, orb_2s, k_i, k_f)
        
        try:
            cw = solve_continuum_wave(r, U_i, L, E, z_ion=0.0, solver="rk45")
            delta = cw.phase_shift
            
            if prev_delta is not None:
                change = abs(delta - prev_delta)
                print(f"{n_pts:>10} {r_max:>8.1f} {delta:>+12.6f} {change:>12.6f}")
            else:
                print(f"{n_pts:>10} {r_max:>8.1f} {delta:>+12.6f} {'---':>12}")
            
            prev_delta = delta
        except Exception as e:
            print(f"{n_pts:>10} {r_max:>8.1f} ERROR: {e}")

test_grid_convergence()


print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
