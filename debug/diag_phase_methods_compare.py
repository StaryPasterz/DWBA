#!/usr/bin/env python
"""
Diagnostic Script: Phase Extraction Method Comparison
======================================================

Comprehensive comparison of two phase extraction methods:
1. Log-derivative matching (single point) - currently used
2. Least-squares fitting (multi-point) - defined but unused

Tests across:
- Multiple energies (low, medium, high)
- Multiple L values (0 to 15+)
- Different potential scenarios (weak, strong)
- Known analytical cases (free particle, Coulomb)

References:
- B.R. Johnson, J. Comp. Phys. 13, 445 (1973) - log-derivative method
- C.J. Noble, Comp. Phys. Comm. 59, 227 (1990) - phase extraction review
- M. Aymar et al., Rev. Mod. Phys. 68, 1015 (1996) - multichannel quantum defect

Usage:
    python debug/diag_phase_methods_compare.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from grid import make_r_grid, k_from_E_eV, RadialGrid
from continuum import (
    solve_continuum_wave, 
    ContinuumWave,
    _extract_phase_logderiv_neutral,
    _extract_phase_logderiv_coulomb,
    _fit_asymptotic_phase_neutral,
    _fit_asymptotic_phase_coulomb,
    _riccati_bessel_jn,
    _riccati_bessel_yn,
    _derivative_5point,
    _numerov_propagate,
    _find_match_point
)
from distorting_potential import build_distorting_potentials, DistortingPotential
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states, BoundOrbital
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import CubicSpline


# ==============================================================================
# TEST CASES
# ==============================================================================

@dataclass
class PhaseExtractionResult:
    """Results from a phase extraction test."""
    E_eV: float
    L: int
    k_au: float
    
    # Method results
    delta_logderiv: float
    delta_lsq: float
    delta_analytical: Optional[float]
    
    # Quality metrics
    amp_logderiv: float
    amp_lsq: float
    residual_lsq: float
    
    # Match point info
    r_match: float
    idx_match: int
    
    # Diagnostics
    chi_at_match: float
    Y_at_match: float
    method_agreement: bool
    notes: str


def analytical_phase_hard_sphere(k: float, L: int, a: float) -> float:
    """
    Analytical phase shift for hard sphere of radius a.
    
    δ_L = -arctan(j_L(ka) / y_L(ka))
    
    This provides a known reference for testing.
    """
    rho = k * a
    if rho < 1e-10:
        return 0.0
    
    jl = spherical_jn(L, rho)
    yl = spherical_yn(L, rho)
    
    if abs(yl) < 1e-100:
        return np.pi / 2.0 if jl > 0 else -np.pi / 2.0
    
    return -np.arctan(jl / yl)


def analytical_phase_born(U_arr: np.ndarray, r: np.ndarray, w: np.ndarray, 
                          k: float, L: int) -> float:
    """
    Born approximation phase shift.
    
    δ_L^Born ≈ -k ∫ U(r) [j_L(kr)]² r² dr
    
    Valid for weak potentials (|δ| << 1).
    """
    rho = k * r
    jl = spherical_jn(L, rho)
    integrand = U_arr * jl**2 * r**2
    delta_born = -k * np.sum(w * integrand)
    return delta_born


def extract_phase_both_methods(
    chi_raw: np.ndarray,
    dchi_raw: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    k_au: float,
    l: int,
    idx_match: int,
    tail_fraction: float = 0.15
) -> Tuple[float, float, float, float, float]:
    """
    Extract phase using both methods for comparison.
    
    Returns:
        (delta_logderiv, delta_lsq, amp_lsq, residual_lsq, Y_m)
    """
    # Method 1: Log-derivative at single point
    chi_m = chi_raw[idx_match]
    dchi_m = dchi_raw[idx_match]
    r_m = r[idx_match]
    
    if abs(chi_m) < 1e-100:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    Y_m = dchi_m / chi_m
    delta_logderiv = _extract_phase_logderiv_neutral(Y_m, k_au, r_m, l)
    
    # Method 2: Least-squares on tail
    n_tail = int(len(r) * tail_fraction)
    idx_tail_start = max(idx_match, len(r) - n_tail)
    
    r_tail = r[idx_tail_start:]
    chi_tail = chi_raw[idx_tail_start:]
    
    if len(r_tail) < 10:
        return delta_logderiv, np.nan, np.nan, np.nan, Y_m
    
    amp_lsq, delta_lsq = _fit_asymptotic_phase_neutral(r_tail, chi_tail, l, k_au)
    
    # Calculate residual (goodness of fit)
    phase_free = k_au * r_tail - l * np.pi / 2.0
    chi_fitted = amp_lsq * np.sin(phase_free + delta_lsq)
    residual = np.sqrt(np.mean((chi_tail - chi_fitted)**2)) / np.max(np.abs(chi_tail))
    
    return delta_logderiv, delta_lsq, amp_lsq, residual, Y_m


def run_comparison_test(grid: RadialGrid, U_arr: np.ndarray, 
                        E_eV: float, L: int, z_ion: float = 0.0) -> PhaseExtractionResult:
    """
    Run full comparison test for given energy and L.
    """
    k_au = k_from_E_eV(E_eV)
    r = grid.r
    
    # Build Q(r) for Numerov
    ell = float(L)
    k2 = k_au ** 2
    Q_full = ell * (ell + 1.0) / (r * r) + 2.0 * U_arr - k2
    
    # Find starting point (if inside barrier)
    S_at_origin = Q_full[0]
    idx_start = 0
    
    if S_at_origin > 0:
        r_turn = np.sqrt(L*(L+1)) / k_au if k_au > 1e-6 else r[-1]
        r_safe = r_turn * 0.9
        idx_found = np.searchsorted(r, r_safe)
        if 0 < idx_found < len(r) - 20:
            idx_start = idx_found
    
    r_eval = r[idx_start:]
    Q_eval = Q_full[idx_start:]
    
    # Initial conditions
    r0 = r_eval[0]
    r1 = r_eval[1]
    
    S0 = Q_eval[0]
    if S0 > 0:
        h_init = r1 - r0
        kappa = np.sqrt(S0)
        chi0 = 1e-20
        chi1 = chi0 * np.exp(kappa * h_init)
    else:
        if ell < 20:
            chi0 = r0 ** (ell + 1.0)
            chi1 = r1 ** (ell + 1.0)
        else:
            chi0 = 1e-10
            chi1 = chi0 * (r1 / r0) ** (ell + 1.0)
    
    # Numerov propagation
    chi_computed, log_scale = _numerov_propagate(r_eval, Q_eval, chi0, chi1)
    
    # Compute derivative
    dchi_computed = np.zeros_like(chi_computed)
    for i in range(len(chi_computed)):
        dchi_computed[i] = _derivative_5point(chi_computed, r_eval, i)
    
    # Place in full grid
    if idx_start > 0:
        chi_raw = np.zeros_like(r, dtype=float)
        chi_raw[idx_start:] = chi_computed
        dchi_raw = np.zeros_like(r, dtype=float)
        dchi_raw[idx_start:] = dchi_computed
    else:
        chi_raw = chi_computed
        dchi_raw = dchi_computed
    
    # Find match point
    idx_match, r_m = _find_match_point(r, U_arr, k_au, L, threshold=1e-4, idx_start=idx_start)
    
    # Extract phases using both methods
    delta_ld, delta_lsq, amp_lsq, residual, Y_m = extract_phase_both_methods(
        chi_raw, dchi_raw, r, grid.w_trapz, k_au, L, idx_match
    )
    
    # Calculate Born approximation for reference
    delta_born = analytical_phase_born(U_arr, r, grid.w_trapz, k_au, L)
    
    # Amplitude from log-deriv (via reference value)
    rho_m = k_au * r_m
    j_hat, _ = _riccati_bessel_jn(L, rho_m)
    n_hat, _ = _riccati_bessel_yn(L, rho_m)
    ref_value = j_hat * np.cos(delta_ld) - n_hat * np.sin(delta_ld)
    chi_m = chi_raw[idx_match]
    amp_ld = abs(chi_m / ref_value) if abs(ref_value) > 1e-100 else 0.0
    
    # Agreement check
    delta_diff = abs(delta_ld - delta_lsq)
    delta_diff = min(delta_diff, 2*np.pi - delta_diff)  # Unwrap
    method_agreement = delta_diff < 0.1  # Within 0.1 rad
    
    # Notes
    notes = []
    if np.isnan(delta_ld):
        notes.append("LogDeriv:NaN")
    if np.isnan(delta_lsq):
        notes.append("LSQ:NaN")
    if not method_agreement and not np.isnan(delta_ld) and not np.isnan(delta_lsq):
        notes.append(f"Δ={delta_diff:.3f}")
    if abs(chi_m) < 1e-50:
        notes.append("χ(r_m)≈0")
    if residual > 0.1:
        notes.append(f"res={residual:.2f}")
    
    return PhaseExtractionResult(
        E_eV=E_eV,
        L=L,
        k_au=k_au,
        delta_logderiv=delta_ld,
        delta_lsq=delta_lsq,
        delta_analytical=delta_born,  # Born as reference
        amp_logderiv=amp_ld,
        amp_lsq=amp_lsq if not np.isnan(amp_lsq) else 0.0,
        residual_lsq=residual if not np.isnan(residual) else 1.0,
        r_match=r_m,
        idx_match=idx_match,
        chi_at_match=chi_m,
        Y_at_match=Y_m if not np.isnan(Y_m) else 0.0,
        method_agreement=method_agreement,
        notes="; ".join(notes) if notes else "OK"
    )


def create_test_potential(grid: RadialGrid, potential_type: str = "hydrogen") -> np.ndarray:
    """Create test potential on grid."""
    r = grid.r
    
    if potential_type == "hydrogen":
        # Pure Coulomb -1/r
        core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
        V_core = V_core_on_grid(grid, core_params)
        
        # Get 1s orbital for static potential
        states_1s = solve_bound_states(grid, V_core, l=0, n_states_max=2)
        orb_1s = [s for s in states_1s if s.n_index == 1][0]
        
        # Build distorting potential (static, no exchange)
        k_dummy = 0.5
        U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, 
                                              k_dummy, k_dummy)
        return U_i.U_of_r
    
    elif potential_type == "weak_gaussian":
        # Weak Gaussian potential (Born approx should work)
        U = -0.1 * np.exp(-r**2 / 4.0)
        return U
    
    elif potential_type == "strong_well":
        # Finite square well (strong scattering)
        a = 5.0  # well radius
        V0 = 2.0  # well depth
        U = np.where(r < a, -V0, 0.0)
        return U
    
    elif potential_type == "free":
        # Free particle (U=0) - phase should be exactly 0
        return np.zeros_like(r)
    
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")


def run_full_diagnostic():
    """Run comprehensive comparison diagnostic."""
    
    print("=" * 80)
    print(" PHASE EXTRACTION METHOD COMPARISON DIAGNOSTIC")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # Create grid
    n_points = 8000
    r_max = 250.0
    grid = make_r_grid(r_max, n_points)
    print(f"\nGrid: r_max={r_max}, n_points={n_points}")
    
    # Test cases
    test_energies = [10.5, 15.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    test_L_values = [0, 1, 2, 3, 5, 7, 10, 15]
    test_potentials = ["hydrogen", "weak_gaussian", "free"]
    
    all_results = {}
    
    for pot_type in test_potentials:
        print(f"\n{'='*60}")
        print(f" Potential: {pot_type.upper()}")
        print(f"{'='*60}")
        
        U_arr = create_test_potential(grid, pot_type)
        
        results_for_pot = []
        
        for E_eV in test_energies:
            k = k_from_E_eV(E_eV)
            print(f"\n  E = {E_eV:8.1f} eV (k = {k:.4f} a.u.)")
            print(f"  {'L':>3} | {'δ_LD':>10} | {'δ_LSQ':>10} | {'Δδ':>8} | {'A_LD':>10} | {'A_LSQ':>10} | {'res':>6} | {'Notes':<20}")
            print("  " + "-" * 90)
            
            for L in test_L_values:
                # Skip if L is too high for energy (turning point beyond grid)
                r_turn = np.sqrt(L * (L + 1)) / k if k > 1e-6 else 1e10
                if r_turn > 0.8 * r_max:
                    print(f"  {L:>3} | {'SKIP':>10} | {'r_turn > r_max':>30}")
                    continue
                
                try:
                    result = run_comparison_test(grid, U_arr, E_eV, L)
                    results_for_pot.append(result)
                    
                    delta_diff = abs(result.delta_logderiv - result.delta_lsq)
                    delta_diff = min(delta_diff, 2*np.pi - delta_diff)
                    
                    print(f"  {L:>3} | {result.delta_logderiv:>+10.5f} | {result.delta_lsq:>+10.5f} | "
                          f"{delta_diff:>8.5f} | {result.amp_logderiv:>10.4f} | {result.amp_lsq:>10.4f} | "
                          f"{result.residual_lsq:>6.3f} | {result.notes:<20}")
                    
                except Exception as e:
                    print(f"  {L:>3} | ERROR: {str(e)[:60]}")
        
        all_results[pot_type] = results_for_pot
    
    # Create diagnostic plots
    create_diagnostic_plots(all_results, grid)
    
    # Calculate statistics
    print_statistics(all_results)
    
    # Save results
    save_results(all_results)
    
    print("\n" + "=" * 80)
    print(" Diagnostic complete!")
    print(" See debug/diag_phase_methods_*.png for plots")
    print("=" * 80)
    
    return all_results


def create_diagnostic_plots(all_results: Dict[str, List[PhaseExtractionResult]], grid: RadialGrid):
    """Create comprehensive diagnostic plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Method agreement vs energy (all potentials)
    ax1 = axes[0, 0]
    for pot_type, results in all_results.items():
        if not results:
            continue
        energies = sorted(set(r.E_eV for r in results))
        disagreement_frac = []
        for E in energies:
            E_results = [r for r in results if r.E_eV == E]
            n_disagree = sum(1 for r in E_results if not r.method_agreement)
            disagreement_frac.append(n_disagree / len(E_results) if E_results else 0)
        ax1.semilogx(energies, disagreement_frac, 'o-', label=pot_type)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Fraction of disagreements')
    ax1.set_title('Method Disagreement vs Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Phase difference vs L for hydrogen
    ax2 = axes[0, 1]
    if 'hydrogen' in all_results:
        for E in [15.0, 50.0, 200.0, 1000.0]:
            E_results = [r for r in all_results['hydrogen'] if abs(r.E_eV - E) < 1]
            if E_results:
                L_vals = [r.L for r in E_results]
                diffs = []
                for r in E_results:
                    d = abs(r.delta_logderiv - r.delta_lsq)
                    d = min(d, 2*np.pi - d)
                    diffs.append(d)
                ax2.plot(L_vals, diffs, 'o-', label=f'{E:.0f} eV')
    ax2.set_xlabel('L')
    ax2.set_ylabel('|δ_LD - δ_LSQ| (rad)')
    ax2.set_title('Phase Difference vs L (Hydrogen)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: LSQ residual vs energy
    ax3 = axes[0, 2]
    for pot_type, results in all_results.items():
        if not results:
            continue
        energies = sorted(set(r.E_eV for r in results))
        mean_residual = []
        for E in energies:
            E_results = [r for r in results if r.E_eV == E and not np.isnan(r.residual_lsq)]
            if E_results:
                mean_residual.append(np.mean([r.residual_lsq for r in E_results]))
            else:
                mean_residual.append(np.nan)
        ax3.semilogx(energies, mean_residual, 'o-', label=pot_type)
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('Mean LSQ residual')
    ax3.set_title('Fit Quality vs Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Amplitude comparison
    ax4 = axes[1, 0]
    expected_amp = np.sqrt(2 / np.pi)  # δ(k-k') normalization
    for pot_type, results in all_results.items():
        if not results:
            continue
        amp_ld = [r.amp_logderiv for r in results if r.amp_logderiv > 0]
        amp_lsq = [r.amp_lsq for r in results if r.amp_lsq > 0]
        if amp_ld and amp_lsq:
            ax4.scatter(amp_ld, amp_lsq, alpha=0.5, label=pot_type)
    ax4.axline((0, 0), slope=1, color='k', linestyle='--', alpha=0.5, label='y=x')
    ax4.axhline(expected_amp, color='r', linestyle=':', alpha=0.5)
    ax4.axvline(expected_amp, color='r', linestyle=':', alpha=0.5, label=f'√(2/π)={expected_amp:.4f}')
    ax4.set_xlabel('Amplitude (Log-Derivative)')
    ax4.set_ylabel('Amplitude (LSQ)')
    ax4.set_title('Amplitude Comparison')
    ax4.legend()
    ax4.set_xlim(0, 2)
    ax4.set_ylim(0, 2)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Free particle check (should be δ=0)
    ax5 = axes[1, 1]
    if 'free' in all_results:
        free_results = all_results['free']
        L_vals = [r.L for r in free_results]
        delta_ld = [r.delta_logderiv for r in free_results]
        delta_lsq = [r.delta_lsq for r in free_results]
        ax5.scatter(L_vals, delta_ld, alpha=0.7, label='Log-Derivative', marker='o')
        ax5.scatter(L_vals, delta_lsq, alpha=0.7, label='Least-Squares', marker='x')
        ax5.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax5.set_xlabel('L')
        ax5.set_ylabel('δ (rad)')
        ax5.set_title('Free Particle Check (should be δ≈0)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary bar chart
    ax6 = axes[1, 2]
    categories = ['Total Tests', 'Agreements', 'Disagreements', 'LogDeriv NaN', 'LSQ NaN']
    for i, (pot_type, results) in enumerate(all_results.items()):
        if not results:
            continue
        total = len(results)
        agrees = sum(1 for r in results if r.method_agreement)
        disagrees = total - agrees
        ld_nan = sum(1 for r in results if np.isnan(r.delta_logderiv))
        lsq_nan = sum(1 for r in results if np.isnan(r.delta_lsq))
        
        x = np.arange(len(categories))
        width = 0.25
        offset = (i - len(all_results)/2 + 0.5) * width
        ax6.bar(x + offset, [total, agrees, disagrees, ld_nan, lsq_nan], 
                width, label=pot_type, alpha=0.8)
    ax6.set_xticks(np.arange(len(categories)))
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    ax6.set_ylabel('Count')
    ax6.set_title('Summary Statistics')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('debug/diag_phase_methods_comparison.png', dpi=150)
    print(f"\nSaved: debug/diag_phase_methods_comparison.png")
    plt.close()


def print_statistics(all_results: Dict[str, List[PhaseExtractionResult]]):
    """Print detailed statistics."""
    
    print("\n" + "=" * 80)
    print(" STATISTICAL ANALYSIS")
    print("=" * 80)
    
    for pot_type, results in all_results.items():
        if not results:
            continue
            
        print(f"\n  {pot_type.upper()}")
        print("  " + "-" * 40)
        
        total = len(results)
        agrees = sum(1 for r in results if r.method_agreement)
        ld_nan = sum(1 for r in results if np.isnan(r.delta_logderiv))
        lsq_nan = sum(1 for r in results if np.isnan(r.delta_lsq))
        
        valid_results = [r for r in results if not np.isnan(r.delta_logderiv) and not np.isnan(r.delta_lsq)]
        
        if valid_results:
            diffs = []
            for r in valid_results:
                d = abs(r.delta_logderiv - r.delta_lsq)
                d = min(d, 2*np.pi - d)
                diffs.append(d)
            
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            
            mean_residual = np.mean([r.residual_lsq for r in valid_results])
            
            # Amplitude accuracy (relative to √(2/π))
            expected_amp = np.sqrt(2 / np.pi)
            amp_ld_err = [abs(r.amp_logderiv - expected_amp)/expected_amp 
                          for r in valid_results if r.amp_logderiv > 0]
            amp_lsq_err = [abs(r.amp_lsq - expected_amp)/expected_amp 
                          for r in valid_results if r.amp_lsq > 0]
            
            print(f"  Total tests:       {total}")
            print(f"  Agreement rate:    {agrees}/{total} ({100*agrees/total:.1f}%)")
            print(f"  LogDeriv failures: {ld_nan}")
            print(f"  LSQ failures:      {lsq_nan}")
            print(f"  Mean |Δδ|:         {mean_diff:.5f} rad")
            print(f"  Max |Δδ|:          {max_diff:.5f} rad")
            print(f"  Std |Δδ|:          {std_diff:.5f} rad")
            print(f"  Mean LSQ residual: {mean_residual:.4f}")
            if amp_ld_err:
                print(f"  Mean amp error LD: {100*np.mean(amp_ld_err):.2f}%")
            if amp_lsq_err:
                print(f"  Mean amp error LSQ:{100*np.mean(amp_lsq_err):.2f}%")


def save_results(all_results: Dict[str, List[PhaseExtractionResult]]):
    """Save results to JSON."""
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'potentials': {}
    }
    
    for pot_type, results in all_results.items():
        output['potentials'][pot_type] = [
            {
                'E_eV': r.E_eV,
                'L': r.L,
                'k_au': r.k_au,
                'delta_logderiv': float(r.delta_logderiv) if not np.isnan(r.delta_logderiv) else None,
                'delta_lsq': float(r.delta_lsq) if not np.isnan(r.delta_lsq) else None,
                'amp_logderiv': float(r.amp_logderiv),
                'amp_lsq': float(r.amp_lsq),
                'residual_lsq': float(r.residual_lsq) if not np.isnan(r.residual_lsq) else None,
                'r_match': float(r.r_match),
                'method_agreement': r.method_agreement,
                'notes': r.notes
            }
            for r in results
        ]
    
    output_path = Path('debug/results') / f'phase_methods_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_full_diagnostic()
