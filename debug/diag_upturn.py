#!/usr/bin/env python
"""
Diagnostic script to trace the source of partial wave upturn at L=9.
This isolates each component to find where numerical instability originates.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from config_loader import load_config
from potential_core import CorePotentialParams, V_core_on_grid
from grid import make_r_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials, DistortingPotential
from dwba_matrix_elements import radial_ME_all_L_gpu
from dwba_coupling import calculate_amplitude_contribution
from sigma_total import dcs_dwba, integrate_dcs_over_angles, sigma_au_to_cm2
from grid import make_r_grid, k_from_E_eV

def analyze_partial_wave(E_eV: float, L_range: range):
    """Analyze individual partial wave contributions."""
    
    # Setup grid and potential (same as pilot)
    print(f"\n{'='*60}")
    print(f"PARTIAL WAVE UPTURN DIAGNOSTIC - E = {E_eV} eV")
    print(f"{'='*60}")
    
    # Load H atom config
    r_min, r_max, n_points = 1e-5, 200.0, 3000
    grid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    print(f"Grid: r_min={r_min}, r_max={r_max}, n_points={n_points}")
    
    # Core potential (H atom: Zc=1, all a_params=0 for pure Coulomb)
    core_params = CorePotentialParams(Zc=1.0, a1=0.0, a2=1.0, a3=0.0, a4=1.0, a5=0.0, a6=1.0)
    V_core = V_core_on_grid(grid, core_params)
    
    # Bound states (1s → 2s)
    states_i = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    states_f = solve_bound_states(grid, V_core, l=0, n_states_max=3)
    
    orb_i = states_i[0]  # 1s
    orb_f = states_f[1]  # 2s
    
    print(f"Initial state: n={orb_i.n_index+1}, l=0, E={orb_i.energy_au*27.211:.3f} eV")
    print(f"Final state: n={orb_f.n_index+1}, l=0, E={orb_f.energy_au*27.211:.3f} eV")
    
    dE_eV = (orb_f.energy_au - orb_i.energy_au) * 27.211
    print(f"Threshold: {dE_eV:.3f} eV")
    
    E_final_eV = E_eV - dE_eV
    if E_final_eV <= 0:
        print("ERROR: E_final <= 0")
        return
    
    ki = float(k_from_E_eV(E_eV))
    kf = float(k_from_E_eV(E_final_eV))
    z_ion = 0.0  # Neutral residue
    
    print(f"k_i = {ki:.4f} a.u., k_f = {kf:.4f} a.u.")
    
    # Build distorting potentials
    U_i, U_f = build_distorting_potentials(
        grid, V_core, orb_i, orb_f,
        k_i_au=ki, k_f_au=kf, use_polarization=False
    )
    
    # Angular setup (H 1s → 2s: Li=0, Lf=0)
    Li, Lf = 0, 0
    theta_grid = np.linspace(0, np.pi, 50)
    L_max_integrals = 8
    
    print(f"\nAnalyzing partial waves L_i = {list(L_range)}")
    print("-" * 60)
    print(f"{'L_i':>4} | {'chi_i norm':>12} | {'chi_f norm':>12} | {'phase_i':>10} | {'phase_f':>10} | {'I_L[0]':>12} | {'sigma_L':>12}")
    print("-" * 60)
    
    results = []
    
    for l_i in L_range:
        try:
            # Solve continuum waves
            chi_i = solve_continuum_wave(grid, U_i, l_i, E_eV, z_ion, solver="rk45")
            if chi_i is None:
                print(f"{l_i:>4} | FAILED chi_i")
                continue
                
            # Check chi_i normalization - use RMS amplitude like the enforcement code
            asym_region = chi_i.chi_of_r[chi_i.idx_match:]
            chi_i_norm = np.sqrt(np.mean(asym_region**2)) * np.sqrt(2.0)  # RMS→amplitude
            
            # l_f ranges
            l_f_values = []
            for l_f in range(max(0, l_i - L_max_integrals), l_i + L_max_integrals + 1):
                # Parity check: (l_i + l_f) must be even for 1s→2s (Li=0, Lf=0)
                if (l_i + l_f) % 2 == 0:
                    l_f_values.append(l_f)
            
            sigma_li = 0.0
            best_I_L0 = 0.0
            chi_f_norm_best = 0.0
            phase_f_best = 0.0
            
            for l_f in l_f_values:
                chi_f = solve_continuum_wave(grid, U_f, l_f, E_final_eV, z_ion, solver="rk45")
                if chi_f is None:
                    continue
                
                chi_f_asym = chi_f.chi_of_r[chi_f.idx_match:]
                chi_f_norm = np.sqrt(np.mean(chi_f_asym**2)) * np.sqrt(2.0)
                
                # Compute radial integrals
                integrals = radial_ME_all_L_gpu(
                    grid, V_core, U_i.U_of_r, orb_i, orb_f, chi_i, chi_f,
                    L_max_integrals,
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
                
                # Check I_L[0] (monopole term - largest for s-s transition)
                I_L0 = abs(integrals.I_L_direct.get(0, 0.0))
                if I_L0 > best_I_L0:
                    best_I_L0 = I_L0
                    chi_f_norm_best = chi_f_norm
                    phase_f_best = chi_f.phase_shift
                
                # Compute amplitude contribution
                amps = calculate_amplitude_contribution(
                    theta_grid,
                    integrals.I_L_direct,
                    integrals.I_L_exchange,
                    l_i, l_f, ki, kf,
                    Li, Lf, 0, 0  # Mi=Mf=0 for s-s
                )
                
                # Compute DCS for this l_f
                dcs = dcs_dwba(theta_grid, amps.f_theta, amps.g_theta, ki, kf, Li, 1)
                sigma_part = sigma_au_to_cm2(integrate_dcs_over_angles(theta_grid, dcs))
                sigma_li += sigma_part
            
            print(f"{l_i:>4} | {chi_i_norm:>12.4e} | {chi_f_norm_best:>12.4e} | {chi_i.phase_shift:>10.4f} | {phase_f_best:>10.4f} | {best_I_L0:>12.4e} | {sigma_li:>12.4e}")
            results.append((l_i, chi_i_norm, chi_f_norm_best, chi_i.phase_shift, phase_f_best, best_I_L0, sigma_li))
            
        except Exception as e:
            print(f"{l_i:>4} | ERROR: {e}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if len(results) >= 3:
        for i in range(1, len(results)):
            L_curr, _, _, _, _, I_L0_curr, sigma_curr = results[i]
            L_prev, _, _, _, _, I_L0_prev, sigma_prev = results[i-1]
            
            I_ratio = I_L0_curr / I_L0_prev if I_L0_prev > 0 else float('inf')
            sigma_ratio = sigma_curr / sigma_prev if sigma_prev > 0 else float('inf')
            
            upturn_marker = " <-- UPTURN" if sigma_ratio > 1.0 else ""
            print(f"L={L_prev}→{L_curr}: I_L[0] ratio={I_ratio:.3f}, sigma ratio={sigma_ratio:.3f}{upturn_marker}")
    
    return results


if __name__ == "__main__":
    # Analyze at pilot energy (1000 eV) where problem was reported
    results = analyze_partial_wave(1000.0, range(6, 14))
    
    print("\n\n" + "="*60)
    print("ALSO TESTING AT LOWER ENERGY (50 eV)")
    results_low = analyze_partial_wave(50.0, range(0, 12))
