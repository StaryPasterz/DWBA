#!/usr/bin/env python
"""
Diagnostic Script: Phase Extraction and Cross Section Analysis
=============================================================

Analyzes the anomalies observed in H 1s→2s cross sections:
1. Cross section dip at 11.6-17.3 eV
2. Upturn at L=9 for high energies (≥69 eV)
3. Born approximation bypass behavior

Usage:
    python debug/diag_phase_extraction.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from grid import make_r_grid, k_from_E_eV
from continuum import solve_continuum_wave, ContinuumWave
from distorting_potential import build_distorting_potentials, DistortingPotential
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states, BoundOrbital
from scipy.special import spherical_jn

# Test energies spanning the anomaly region
TEST_ENERGIES = [10.7, 11.0, 11.5, 12.0, 12.65, 14.37, 17.28, 22.24, 30.66, 69.26, 110.56]
L_VALUES = [0, 1, 2, 3, 5, 7, 9, 11]

def create_hydrogen_potential(grid):
    """Create H distorting potential (static, neutral)."""
    # Pure Coulomb -1/r (hydrogen)
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core_params)
    
    # Get initial state for building potential
    states_1s = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    orb_1s = [s for s in states_1s if s.n_index == 1][0]
    
    # k doesn't matter much for neutral static potential
    k_dummy = 0.5
    U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k_dummy, k_dummy)
    
    return U_i, V_core

def analyze_phase_extraction(E_eV: float, grid, U_channel, L_max: int = 12):
    """Analyze phase extraction for given energy across L values."""
    k = k_from_E_eV(E_eV)
    results = []
    
    print(f"\n{'='*60}")
    print(f"Energy: {E_eV:.2f} eV, k = {k:.4f} a.u.")
    print(f"{'='*60}")
    print(f"{'L':>3} | {'delta_l (rad)':>12} | {'sigma_L (a.u.)':>15} | {'Method':>15} | {'Notes':>20}")
    print("-" * 75)
    
    for L in range(L_max + 1):
        try:
            wave = solve_continuum_wave(grid, U_channel, L, E_eV, z_ion=0.0)
            delta = wave.delta_l
            
            # Calculate partial cross section contribution
            sigma_L = (2*L + 1) * np.sin(delta)**2 / (k**2)
            
            # Check if Born bypass was used
            method = "Born bypass" if abs(delta) < 0.1 else "Full solver"
            
            # Check for suspicious values
            notes = ""
            if abs(delta) > np.pi/2:
                notes = "! |delta| > pi/2"
            elif np.isnan(delta):
                notes = "X NaN"
            elif abs(delta) < 1e-10:
                notes = "~0"
            
            results.append({
                'L': L,
                'delta': delta,
                'sigma_L': sigma_L,
                'method': method,
                'notes': notes
            })
            
            print(f"{L:>3} | {delta:>12.6f} | {sigma_L:>15.4e} | {method:>15} | {notes:>20}")
            
        except Exception as e:
            print(f"{L:>3} | {'ERROR':>12} | {'N/A':>12} | {'N/A':>15} | {str(e)[:20]:>20}")
            results.append({
                'L': L,
                'delta': np.nan,
                'sigma_L': np.nan,
                'method': 'ERROR',
                'notes': str(e)[:20]
            })
    
    return results

def check_born_phase(grid, U_arr, k, L):
    """Calculate Born approximation phase shift directly for comparison."""
    r = grid.r
    rho = k * r
    
    # Spherical Bessel function
    jl = spherical_jn(L, rho)
    
    # Born phase: δ_Born = -k ∫ U(r) [j_l(kr)]² r² dr
    integrand = U_arr * jl**2 * r**2
    delta_born = -k * np.sum(grid.w_trapz * integrand)
    
    return delta_born

def analyze_wavefunction_quality(E_eV: float, grid, U_channel, L: int):
    """Analyze wavefunction quality for specific (E, L)."""
    k = k_from_E_eV(E_eV)
    r = grid.r
    
    try:
        wave = solve_continuum_wave(grid, U_channel, E_eV, L, z_ion=0.0)
        chi = wave.chi
        delta = wave.delta_l
        
        # Expected asymptotic form: A * sin(kr - Lπ/2 + δ)
        # Check at large r (last 20% of grid)
        idx_asymp = int(0.8 * len(r))
        r_asymp = r[idx_asymp:]
        chi_asymp = chi[idx_asymp:]
        
        # Theoretical asymptotic
        A = np.sqrt(2 / np.pi)
        chi_theory = A * np.sin(k * r_asymp - L * np.pi / 2 + delta)
        
        # Check amplitude consistency
        amp_computed = np.max(np.abs(chi_asymp))
        amp_theory = A
        amp_ratio = amp_computed / amp_theory if amp_theory > 0 else np.nan
        
        # Check phase consistency by cross-correlation
        if len(chi_asymp) > 10:
            correlation = np.corrcoef(chi_asymp, chi_theory)[0, 1]
        else:
            correlation = np.nan
        
        return {
            'amp_ratio': amp_ratio,
            'correlation': correlation,
            'delta': delta,
            'chi_max': np.max(np.abs(chi))
        }
        
    except Exception as e:
        return {'error': str(e)}

def run_full_diagnostic():
    """Run full diagnostic analysis."""
    print("=" * 70)
    print(" DWBA Phase Extraction Diagnostic ")
    print("=" * 70)
    
    # Create grid with high resolution
    n_points = 10000
    r_max = 200.0
    grid = make_r_grid(r_max, n_points)
    print(f"Grid: r_max={r_max}, n_points={n_points}")
    print(f"Grid dr at r=100: {grid.r[np.argmin(np.abs(grid.r - 100)) + 1] - grid.r[np.argmin(np.abs(grid.r - 100))]:.4f} a.u.")
    
    # Create H potential
    U_channel, V_core = create_hydrogen_potential(grid)
    print(f"Potential: H (static, neutral)")
    
    # Collect results for plotting
    all_results = {}
    
    for E in TEST_ENERGIES:
        threshold = 10.2  # H 1s→2s threshold
        if E <= threshold:
            print(f"\nSkipping E={E} eV (below threshold {threshold} eV)")
            continue
            
        results = analyze_phase_extraction(E, grid, U_channel, L_max=12)
        all_results[E] = results
    
    # Summary plots
    create_diagnostic_plots(all_results, grid, U_channel)
    
    print("\n" + "=" * 70)
    print(" Diagnostic complete. Check debug/diag_phase_*.png ")
    print("=" * 70)

def create_diagnostic_plots(all_results: dict, grid, U_channel):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Phase shift vs L for different energies
    ax1 = axes[0, 0]
    for E, results in all_results.items():
        L_vals = [r['L'] for r in results]
        delta_vals = [r['delta'] for r in results]
        ax1.plot(L_vals, delta_vals, 'o-', label=f'{E:.1f} eV')
    ax1.set_xlabel('L')
    ax1.set_ylabel('δ_L (rad)')
    ax1.set_title('Phase Shifts vs Angular Momentum')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Partial cross sections vs L
    ax2 = axes[0, 1]
    for E, results in all_results.items():
        L_vals = [r['L'] for r in results]
        sigma_vals = [r['sigma_L'] for r in results if not np.isnan(r['sigma_L'])]
        L_valid = [r['L'] for r in results if not np.isnan(r['sigma_L'])]
        if sigma_vals:
            ax2.semilogy(L_valid, sigma_vals, 'o-', label=f'{E:.1f} eV')
    ax2.set_xlabel('L')
    ax2.set_ylabel('σ_L (a.u.)')
    ax2.set_title('Partial Cross Sections vs L')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total cross section vs energy
    ax3 = axes[1, 0]
    energies = sorted(all_results.keys())
    sigma_total = []
    for E in energies:
        results = all_results[E]
        sigma = np.nansum([r['sigma_L'] for r in results])
        sigma_total.append(sigma)
    ax3.semilogy(energies, sigma_total, 'ko-', markersize=8)
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('σ_total (a.u.)')
    ax3.set_title('Total Cross Section vs Energy')
    ax3.grid(True, alpha=0.3)
    
    # Highlight anomaly region
    ax3.axvspan(11, 18, alpha=0.2, color='red', label='Anomaly region')
    ax3.legend()
    
    # Plot 4: Wavefunction sample at problem energy
    ax4 = axes[1, 1]
    E_problem = 12.65  # Middle of anomaly
    k = k_from_E_eV(E_problem)
    for L in [0, 1, 2, 5]:
        try:
            wave = solve_continuum_wave(grid, U_channel, E_problem, L, z_ion=0.0)
            # Plot last 30 a.u.
            idx = np.where(grid.r > 170)[0]
            ax4.plot(grid.r[idx], wave.chi[idx], label=f'L={L}')
        except Exception as e:
            print(f"  Warning: Could not plot L={L}: {e}")
    ax4.set_xlabel('r (a.u.)')
    ax4.set_ylabel('χ_L(r)')
    ax4.set_title(f'Wavefunctions at E={E_problem} eV (asymptotic region)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug/diag_phase_extraction.png', dpi=150)
    print(f"Saved: debug/diag_phase_extraction.png")
    plt.close()

if __name__ == "__main__":
    run_full_diagnostic()
