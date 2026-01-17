#!/usr/bin/env python
"""
test_solver_comparison.py
=========================

Compare Numerov vs Johnson vs RK45 solvers in realistic DWBA conditions.
Identify where the ~100x overestimation originates.

Run:
    python debug/test_solver_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from distorting_potential import build_distorting_potentials
from continuum import solve_continuum_wave


def test_solver_comparison():
    """
    Compare all three solvers for H 1s→2s at 50 eV.
    """
    print("=" * 70)
    print("SOLVER COMPARISON: H excitation at 50 eV")
    print("=" * 70)
    
    # Setup grid and potential (H atom)
    r_max = 200.0
    n_points = 3000
    grid = make_r_grid(r_max=r_max, n_points=n_points)
    
    # H atom SAE parameters
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, params)
    
    # Solve for bound states
    orbs_s = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    orb_1s = orbs_s[0]
    orb_2s = orbs_s[1]
    
    print(f"\nBound states:")
    print(f"  1s: E = {orb_1s.energy_au * 27.211:.4f} eV")
    print(f"  2s: E = {orb_2s.energy_au * 27.211:.4f} eV")
    
    # Excitation threshold
    dE = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
    print(f"  Threshold: {dE:.2f} eV")
    
    # Build distorting potentials
    E_inc_eV = 50.0
    k_i = k_from_E_eV(E_inc_eV)
    k_f = k_from_E_eV(E_inc_eV - dE)
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
    
    print(f"\nIncident: E = {E_inc_eV} eV, k_i = {k_i:.4f} a.u.")
    print(f"Final:    E = {E_inc_eV - dE:.2f} eV, k_f = {k_f:.4f} a.u.")
    
    # Test all three solvers for various L
    solvers = ["numerov", "johnson", "rk45"]
    L_values = [0, 1, 2, 5, 10, 15, 20]
    
    print("\n" + "-" * 70)
    print(f"{'L':>3} | {'Solver':<10} | {'Phase (rad)':>12} | {'Amplitude':>12} | {'Method':>15}")
    print("-" * 70)
    
    results = {}
    
    for L in L_values:
        results[L] = {}
        for solver in solvers:
            try:
                cw = solve_continuum_wave(
                    grid, U_i, L, E_inc_eV, 
                    z_ion=0.0, 
                    solver=solver,
                    phase_extraction_method="hybrid"
                )
                
                # Get asymptotic amplitude
                idx_tail = int(0.9 * len(grid.r))
                chi_tail = cw.chi_of_r[idx_tail:]
                A_asymp = np.sqrt(np.mean(chi_tail**2)) * np.sqrt(2)  # RMS amplitude
                
                results[L][solver] = {
                    'phase': cw.phase_shift,
                    'amplitude': A_asymp,
                    'method': cw.phase_method,
                    'solver_used': cw.solver_method
                }
                
                print(f"{L:>3} | {solver:<10} | {cw.phase_shift:>+12.4f} | {A_asymp:>12.4e} | {cw.phase_method}")
                
            except Exception as e:
                print(f"{L:>3} | {solver:<10} | ERROR: {str(e)[:40]}")
                results[L][solver] = None
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Phase differences between solvers")
    print("=" * 70)
    
    for L in L_values:
        if all(results[L].get(s) for s in solvers):
            num_phase = results[L]['numerov']['phase']
            joh_phase = results[L]['johnson']['phase']
            rk_phase = results[L]['rk45']['phase']
            
            diff_nj = abs(num_phase - joh_phase)
            diff_nr = abs(num_phase - rk_phase)
            diff_jr = abs(joh_phase - rk_phase)
            
            max_diff = max(diff_nj, diff_nr, diff_jr)
            
            status = "✓ OK" if max_diff < 0.1 else "✗ DIFFER"
            print(f"L={L:>2}: Num-Joh={diff_nj:.4f}, Num-RK={diff_nr:.4f}, Joh-RK={diff_jr:.4f}  {status}")
    
    # Check amplitude ratios
    print("\n" + "=" * 70)
    print("ANALYSIS: Amplitude ratios (should be ~1.0 = sqrt(2/π) ≈ 0.798)")
    print("=" * 70)
    expected_amp = np.sqrt(2/np.pi)
    print(f"Expected asymptotic amplitude: {expected_amp:.4f}")
    
    for L in L_values:
        if results[L].get('numerov'):
            amp = results[L]['numerov']['amplitude']
            ratio = amp / expected_amp
            status = "✓" if 0.5 < ratio < 2.0 else "✗"
            print(f"L={L:>2}: A={amp:.4e}, ratio to expected: {ratio:.2f}  {status}")


def test_radial_integrals_comparison():
    """
    Compare radial integral results using different solvers.
    This tests the full pipeline through to cross sections.
    """
    print("\n" + "=" * 70)
    print("RADIAL INTEGRALS COMPARISON")
    print("=" * 70)
    
    from dwba_matrix_elements import radial_ME_all_L
    
    # Setup
    r_max = 200.0
    n_points = 3000
    grid = make_r_grid(r_max=r_max, n_points=n_points)
    
    params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, params)
    
    orbs_s = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    orb_1s = orbs_s[0]
    orb_2s = orbs_s[1]
    
    dE = (orb_2s.energy_au - orb_1s.energy_au) * 27.211
    E_inc_eV = 50.0
    k_i = k_from_E_eV(E_inc_eV)
    k_f = k_from_E_eV(E_inc_eV - dE)
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, k_i, k_f)
    
    # Solve continuum waves with different solvers
    L_max = 5
    
    for solver in ["numerov", "johnson", "rk45"]:
        print(f"\n--- Solver: {solver.upper()} ---")
        
        # Solve for l_i=0, l_f=0
        try:
            chi_i = solve_continuum_wave(grid, U_i, 0, E_inc_eV, z_ion=0.0, solver=solver)
            chi_f = solve_continuum_wave(grid, U_f, 0, E_inc_eV - dE, z_ion=0.0, solver=solver)
            
            print(f"  chi_i(L=0): phase={chi_i.phase_shift:+.4f}, solver={chi_i.solver_method}")
            print(f"  chi_f(L=0): phase={chi_f.phase_shift:+.4f}, solver={chi_f.solver_method}")
            
            # Compute radial integrals
            integrals = radial_ME_all_L(
                grid, V_core, U_i.U_of_r, orb_1s, orb_2s, 
                chi_i, chi_f, L_max,
                use_oscillatory_quadrature=True,
                method="advanced"
            )
            
            print(f"  Radial integrals I_L (direct):")
            for L in range(min(4, L_max+1)):
                I_L = integrals.I_L_direct.get(L, 0.0)
                print(f"    I_{L} = {I_L:.6e}")
                
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    test_solver_comparison()
    # test_radial_integrals_comparison()  # Uncomment for deeper test
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
