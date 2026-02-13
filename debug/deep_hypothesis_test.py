#!/usr/bin/env python
"""
Deep diagnostic script for upturn hypotheses verification.
Tests each potential cause systematically and measures impact.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from dwba_matrix_elements import radial_ME_all_L_gpu
from dwba_coupling import calculate_amplitude_contribution
from sigma_total import dcs_dwba, integrate_dcs_over_angles, sigma_au_to_cm2

def setup_H_atom(r_max=200.0, n_points=3000):
    """Setup H atom for testing."""
    grid = make_r_grid(r_min=1e-5, r_max=r_max, n_points=n_points)
    core_params = CorePotentialParams(Zc=1.0, a1=0.0, a2=1.0, a3=0.0, a4=1.0, a5=0.0, a6=1.0)
    V_core = V_core_on_grid(grid, core_params)
    
    states_i = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    states_f = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    
    orb_i = states_i[0]  # 1s
    orb_f = states_f[1]  # 2s
    
    return grid, V_core, core_params, orb_i, orb_f


def test_hypothesis_1_phase_sampling():
    """
    Hypothesis 1: Oscillatory Quadrature Undersampling
    
    At high energies (E=1000 eV, k~8.5), wavelength λ~0.74 a.u.
    With exponential grid, dr at large r is significant.
    Check if we have enough points per wavelength.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: OSCILLATORY QUADRATURE UNDERSAMPLING")
    print("="*70)
    
    for E_eV in [50.0, 200.0, 500.0, 1000.0]:
        k = float(k_from_E_eV(E_eV))
        wavelength = 2 * np.pi / k
        
        # Standard grid parameters
        r_max = 200.0
        n_points = 3000
        grid = make_r_grid(1e-5, r_max, n_points)
        
        # Check points per wavelength at different r
        r_test = [10, 50, 100, 150]
        print(f"\nE = {E_eV} eV, k = {k:.3f} a.u., λ = {wavelength:.3f} a.u.")
        print(f"  r(a.u.) | dr(a.u.) | pts/λ | Status")
        print(f"  --------|----------|-------|--------")
        
        for r_val in r_test:
            idx = np.searchsorted(grid.r, r_val)
            if idx > 0 and idx < len(grid.r) - 1:
                dr = grid.r[idx+1] - grid.r[idx]
                pts_per_wavelength = wavelength / dr
                status = "OK" if pts_per_wavelength >= 10 else "⚠️ LOW" if pts_per_wavelength >= 5 else "❌ CRITICAL"
                print(f"  {r_val:7.1f} | {dr:8.4f} | {pts_per_wavelength:5.1f} | {status}")


def test_hypothesis_2_kernel_precision():
    """
    Hypothesis 2: Kernel Recurrence Precision Loss
    
    The kernel K_L = inv_gtr * exp(L * log_ratio) may underflow for high L.
    Test kernel values at various L.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: KERNEL RECURRENCE PRECISION")
    print("="*70)
    
    grid = make_r_grid(1e-5, 200.0, 3000)
    r = grid.r
    
    # Sample points for kernel test
    idx_test = [500, 1000, 2000]  # Small, medium, large r
    
    r1_col = r[:, np.newaxis]
    r2_row = r[np.newaxis, :]
    
    r_less = np.minimum(r1_col, r2_row)
    r_gtr = np.maximum(r1_col, r2_row)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = r_less / r_gtr
        inv_gtr = 1.0 / r_gtr
    
    ratio[~np.isfinite(ratio)] = 0.0
    inv_gtr[~np.isfinite(inv_gtr)] = 0.0
    
    ratio_clamped = np.minimum(ratio, 1.0 - 1e-12)
    log_ratio = np.log(ratio_clamped + 1e-300)
    
    print("\nKernel value K_L(r1, r2) at sample points:")
    print(f"L    | K(r=10, r=50) | K(r=50, r=100) | K(r=100, r=150)")
    print("-" * 60)
    
    pairs = [(np.searchsorted(r, 10), np.searchsorted(r, 50)),
             (np.searchsorted(r, 50), np.searchsorted(r, 100)),
             (np.searchsorted(r, 100), np.searchsorted(r, 150))]
    
    for L in [0, 5, 10, 15, 20, 25, 30]:
        kernel_L = inv_gtr * np.exp(L * log_ratio)
        
        values = []
        for (i1, i2) in pairs:
            K_val = kernel_L[i1, i2]
            values.append(f"{K_val:.2e}" if np.isfinite(K_val) and abs(K_val) > 1e-300 else "UNDERFLOW")
        
        print(f"{L:4d} | {values[0]:^14s} | {values[1]:^14s} | {values[2]:^14s}")


def test_hypothesis_3_radial_integral_L_dependence():
    """
    Hypothesis 3: Radial Integral L-Dependence
    
    Check if I_L values decrease monotonically with L (expected for multipole expansion).
    Non-monotonic behavior indicates numerical issues.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: RADIAL INTEGRAL L-DEPENDENCE")
    print("="*70)
    
    grid, V_core, core_params, orb_i, orb_f = setup_H_atom()
    
    for E_eV in [50.0, 1000.0]:
        print(f"\n--- E = {E_eV} eV ---")
        
        dE_eV = (orb_f.energy_au - orb_i.energy_au) * 27.211
        E_final_eV = E_eV - dE_eV
        
        if E_final_eV <= 0:
            print("  E_final <= 0, skipping")
            continue
        
        ki = float(k_from_E_eV(E_eV))
        kf = float(k_from_E_eV(E_final_eV))
        
        U_i, U_f = build_distorting_potentials(
            grid, V_core, orb_i, orb_f,
            k_i_au=ki, k_f_au=kf, use_polarization=False
        )
        
        # Pick a specific l_i, l_f pair
        l_i, l_f = 5, 5
        
        chi_i = solve_continuum_wave(grid, U_i, l_i, E_eV, 0.0, solver="rk45")
        chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, 0.0, solver="rk45")
        
        if not chi_i or not chi_f:
            print(f"  Solver failed for l_i={l_i}, l_f={l_f}")
            continue
        
        # Compute radial integrals for L=0 to 15
        integrals = radial_ME_all_L_gpu(
            grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f,
            L_max=15,
            use_oscillatory_quadrature=True,
            oscillatory_method="advanced",
            CC_nodes=5,
            phase_increment=np.pi/2,
            gpu_block_size="auto",
            min_grid_fraction=0.1,
            k_threshold=0.5,
            gpu_memory_mode="auto",
            gpu_memory_threshold=0.8,
            gpu_cache=None,
            U_f_array=U_f.U_of_r
        )
        
        print(f"  l_i={l_i}, l_f={l_f}")
        print(f"  L   |  I_L_direct   | I_L_exchange | Ratio to prev")
        print(f"  ----|---------------|--------------|---------------")
        
        prev_I_dir = None
        for L in range(16):
            I_dir = integrals.I_L_direct.get(L, 0.0)
            I_exc = integrals.I_L_exchange.get(L, 0.0)
            
            if prev_I_dir and abs(prev_I_dir) > 1e-30:
                ratio = abs(I_dir / prev_I_dir)
                ratio_str = f"{ratio:.3f}"
                if ratio > 1.0:
                    ratio_str += " ⚠️ UPTURN"
            else:
                ratio_str = "-"
            
            print(f"  {L:3d} | {I_dir:+.6e} | {I_exc:+.6e} | {ratio_str}")
            prev_I_dir = I_dir


def test_hypothesis_4_bound_state_extent():
    """
    Hypothesis 4: Bound State Extent
    
    Check if 99% probability extent is correctly calculated for H states.
    For H 2s, extent should be ~13 a₀.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 4: BOUND STATE EXTENT")
    print("="*70)
    
    grid, V_core, core_params, orb_i, orb_f = setup_H_atom()
    
    for orb, name in [(orb_i, "1s"), (orb_f, "2s")]:
        u_sq = orb.u_of_r ** 2
        prob_cum = np.cumsum(u_sq * grid.w_trapz)
        prob_cum /= prob_cum[-1]
        
        idx_90 = np.searchsorted(prob_cum, 0.90)
        idx_95 = np.searchsorted(prob_cum, 0.95)
        idx_99 = np.searchsorted(prob_cum, 0.99)
        
        r_90 = grid.r[idx_90]
        r_95 = grid.r[idx_95]
        r_99 = grid.r[idx_99]
        
        print(f"\n{name}: E = {orb.energy_au * 27.211:.3f} eV")
        print(f"  90% extent: r = {r_90:.2f} a₀ (idx={idx_90})")
        print(f"  95% extent: r = {r_95:.2f} a₀ (idx={idx_95})")
        print(f"  99% extent: r = {r_99:.2f} a₀ (idx={idx_99})")
        
        # Expected: 1s ~ 4 a₀, 2s ~ 13 a₀
        if name == "1s" and abs(r_99 - 4.0) > 2.0:
            print(f"  ⚠️ WARNING: 1s extent {r_99:.1f} a₀ differs from expected ~4 a₀")
        if name == "2s" and abs(r_99 - 13.0) > 4.0:
            print(f"  ⚠️ WARNING: 2s extent {r_99:.1f} a₀ differs from expected ~13 a₀")


def test_grid_density_impact():
    """
    Test: Does increasing grid density improve results?
    
    Compare sigma at 1000 eV with different grid sizes.
    """
    print("\n" + "="*70)
    print("TEST: GRID DENSITY IMPACT ON σ")
    print("="*70)
    
    E_eV = 1000.0
    
    results = []
    for n_points in [3000, 5000, 8000]:
        grid, V_core, core_params, orb_i, orb_f = setup_H_atom(n_points=n_points)
        
        dE_eV = (orb_f.energy_au - orb_i.energy_au) * 27.211
        E_final_eV = E_eV - dE_eV
        
        ki = float(k_from_E_eV(E_eV))
        kf = float(k_from_E_eV(E_final_eV))
        
        U_i, U_f = build_distorting_potentials(
            grid, V_core, orb_i, orb_f,
            k_i_au=ki, k_f_au=kf, use_polarization=False
        )
        
        # Compute sigma for L=0 to 10
        sigma_total = 0.0
        theta_grid = np.linspace(0, np.pi, 50)
        L_max_integrals = 8
        
        for l_i in range(0, 11):
            chi_i = solve_continuum_wave(grid, U_i, l_i, E_eV, 0.0, solver="rk45")
            if not chi_i:
                continue
            
            # Only same parity l_f (for s-s transition)
            for l_f in range(l_i % 2, l_i + 9, 2):
                chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, 0.0, solver="rk45")
                if not chi_f:
                    continue
                
                integrals = radial_ME_all_L_gpu(
                    grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f,
                    L_max_integrals,
                    use_oscillatory_quadrature=True,
                    oscillatory_method="advanced",
                    U_f_array=U_f.U_of_r
                )
                
                amps = calculate_amplitude_contribution(
                    theta_grid,
                    integrals.I_L_direct,
                    integrals.I_L_exchange,
                    l_i, l_f, ki, kf,
                    0, 0, 0, 0  # Li, Lf, Mi, Mf for s-s
                )
                
                dcs = dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, ki, kf, 0, 1)
                sigma_part = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, dcs))
                sigma_total += sigma_part
        
        results.append((n_points, sigma_total))
        print(f"n_points={n_points:5d}: σ = {sigma_total:.4e} cm²")
    
    # Check if results converge
    if len(results) >= 2:
        diff_percent = abs(results[-1][1] - results[-2][1]) / abs(results[-2][1] + 1e-30) * 100
        print(f"\nChange from {results[-2][0]} to {results[-1][0]} points: {diff_percent:.1f}%")
        if diff_percent > 10:
            print("⚠️ Results NOT converged - grid density matters!")
        else:
            print("✓ Results appear converged")


if __name__ == "__main__":
    print("="*70)
    print("DEEP DIAGNOSTIC: UPTURN HYPOTHESES VERIFICATION")
    print("="*70)
    
    test_hypothesis_1_phase_sampling()
    test_hypothesis_2_kernel_precision()
    test_hypothesis_4_bound_state_extent()
    test_hypothesis_3_radial_integral_L_dependence()
    test_grid_density_impact()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
