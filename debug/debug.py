# debug.py
#
# DWBA Comprehensive Diagnostic Suite
# Unified debugging, verification, and analysis framework for DWBA calculations.
#
# Usage: python debug/debug.py
#
# Features:
# - Bound state analysis (normalization, energy, orthogonality, nodes)
# - Continuum wave analysis (phase shifts, asymptotic behavior)
# - Potential diagnostics (core, Hartree, exchange, distorting)
# - Radial integral analysis (I_L breakdown, method comparison)
# - Angular coupling verification (CG, Racah, amplitudes)
# - Cross section analysis (DCS, TCS, partial waves)
# - Convergence studies (grid, theta sampling)
#

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import sys
import time
import json
import traceback
import runpy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from contextlib import contextmanager
from datetime import datetime

# Ensure local path integration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import Core Modules ---
from grid import make_r_grid, ev_to_au, k_from_E_eV, integrate_trapz, RadialGrid
from scipy.integrate import trapezoid
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states, BoundOrbital
from distorting_potential import build_distorting_potentials, DistortingPotential
from continuum import solve_continuum_wave, ContinuumWave
from dwba_matrix_elements import radial_ME_all_L
from dwba_coupling import calculate_amplitude_contribution, Amplitudes, clebsch_gordan, racah_W
from driver import compute_total_excitation_cs, ExcitationChannelSpec
from atom_library import get_atom

# --- Output Directories (inside debug/) ---
DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_RESULTS_DIR = os.path.join(DEBUG_DIR, "results")
DEBUG_PLOTS_DIR = os.path.join(DEBUG_DIR, "plots")

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DEBUG_RESULTS_DIR, exist_ok=True)
    os.makedirs(DEBUG_PLOTS_DIR, exist_ok=True)

def save_result(name: str, data: dict):
    """Save diagnostic result to JSON."""
    ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DEBUG_RESULTS_DIR, f"{name}_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return obj
    
    with open(filename, 'w') as f:
        json.dump(convert(data), f, indent=2)
    print(f"  [SAVED] {filename}")
    return filename

# --- Configuration ---

@dataclass
class DebugConfig:
    """Configuration for diagnostic runs."""
    Z: float = 1.0
    E_inc_eV: float = 50.0
    transition_name: str = "1s -> 2p"
    ni: int = 1
    li: int = 0
    nf: int = 2
    lf: int = 1
    r_max: float = 100.0
    n_points: int = 3000
    check_high_L: bool = True
    plot_waves: bool = True
    mode: str = "excitation"
    verbose: bool = True

# =============================================================================
# DIAGNOSTIC MODULES
# =============================================================================

class DiagnosticResult:
    """Container for diagnostic results with pass/fail tracking."""
    def __init__(self, name: str):
        self.name = name
        self.checks: List[Dict] = []
        self.data: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}
        self.start_time = time.perf_counter()
    
    def add_check(self, description: str, passed: bool, value: Any = None, expected: Any = None):
        self.checks.append({
            "description": description,
            "passed": passed,
            "value": value,
            "expected": expected
        })
        status = "PASS" if passed else "FAIL"
        marker = "✓" if passed else "✗"
        if value is not None and expected is not None:
            print(f"  [{status}] {marker} {description}: {value} (expected: {expected})")
        elif value is not None:
            print(f"  [{status}] {marker} {description}: {value}")
        else:
            print(f"  [{status}] {marker} {description}")
    
    def add_data(self, key: str, value: Any):
        self.data[key] = value
    
    def add_timing(self, step: str, duration_ms: float):
        self.timings[step] = duration_ms
    
    def summary(self) -> Dict:
        passed = sum(1 for c in self.checks if c['passed'])
        total = len(self.checks)
        return {
            "name": self.name,
            "passed": passed,
            "total": total,
            "success_rate": passed/total if total > 0 else 1.0,
            "checks": self.checks,
            "data": self.data,
            "timings": self.timings,
            "total_time_s": time.perf_counter() - self.start_time
        }

@contextmanager
def timed_section(result: DiagnosticResult, step_name: str):
    """Context manager for timing code sections."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        result.add_timing(step_name, duration_ms)
        print(f"  [TIME] {step_name}: {duration_ms:.2f} ms")

# =============================================================================
# 1. BOUND STATE DIAGNOSTICS
# =============================================================================

def run_bound_state_diagnostic(atom_name: str = "H", l_max: int = 1, verbose: bool = True) -> DiagnosticResult:
    """
    Comprehensive bound state analysis including:
    - Normalization check (∫|u|² dr = 1)
    - Energy accuracy (compare with hydrogen formula)
    - Orthogonality between states
    - Node counting (n - l - 1 rule)
    - Radial extent (99% probability radius)
    """
    result = DiagnosticResult("bound_state_analysis")
    print("\n" + "=" * 60)
    print(f"BOUND STATE DIAGNOSTIC: {atom_name}")
    print("=" * 60)
    
    # Setup
    atom = get_atom(atom_name)
    grid = make_r_grid(r_min=1e-5, r_max=200.0, n_points=5000)
    V_core = V_core_on_grid(grid, atom.core_params)
    
    result.add_data("atom", atom_name)
    result.add_data("Z", atom.Z)
    result.add_data("grid_points", len(grid.r))
    result.add_data("r_max", float(grid.r[-1]))
    
    all_states = []
    
    for l in range(l_max + 1):
        print(f"\n--- l = {l} states ---")
        with timed_section(result, f"solve_l{l}"):
            states = solve_bound_states(grid, V_core, l=l, n_states_max=5)
        
        for orb in states:
            n = orb.n_index + l  # Spectroscopic n
            state_key = f"n{n}_l{l}"
            
            # 1. Normalization
            norm = integrate_trapz(orb.u_of_r**2, grid)
            result.add_check(f"Norm {state_key}", abs(norm - 1.0) < 1e-3, f"{norm:.6f}", "1.0")
            
            # 2. Energy (for hydrogen)
            if atom_name == "H":
                expected_E = -0.5 / (n**2)
                result.add_check(f"Energy {state_key}", abs(orb.energy_au - expected_E) < 1e-3, 
                               f"{orb.energy_au:.6f} a.u.", f"{expected_E:.6f} a.u.")
            
            # 3. Node count
            u_max = np.max(np.abs(orb.u_of_r))
            mask = np.abs(orb.u_of_r) > 1e-4 * u_max
            if np.any(mask):
                nodes = np.sum(np.diff(np.sign(orb.u_of_r[mask])) != 0)
            else:
                nodes = 0
            expected_nodes = orb.n_index - 1  # n_index starts at 1
            result.add_check(f"Nodes {state_key}", nodes == expected_nodes, nodes, expected_nodes)
            
            # 4. Radial extent (99% probability)
            cumsum = np.cumsum(orb.u_of_r**2 * grid.w_trapz)
            cumsum /= cumsum[-1]
            r_99 = grid.r[np.searchsorted(cumsum, 0.99)]
            result.add_data(f"r_99_{state_key}", float(r_99))
            print(f"  [INFO] {state_key} radial extent (99%): {r_99:.2f} a.u.")
            
            all_states.append((l, orb))
    
    # 5. Orthogonality between states of same l
    print("\n--- Orthogonality Checks ---")
    for l in range(l_max + 1):
        l_states = [orb for ll, orb in all_states if ll == l]
        for i, orb1 in enumerate(l_states):
            for j, orb2 in enumerate(l_states):
                if i < j:
                    overlap = integrate_trapz(orb1.u_of_r * orb2.u_of_r, grid)
                    result.add_check(f"Ortho l={l} states {orb1.n_index},{orb2.n_index}", 
                                   abs(overlap) < 1e-4, f"{overlap:.2e}", "0")
    
    # Save results
    save_result("bound_states", result.summary())
    return result

# =============================================================================
# 2. CONTINUUM WAVE DIAGNOSTICS  
# =============================================================================

def run_continuum_diagnostic(E_eV: float = 50.0, L_max: int = 10, atom: str = "H") -> DiagnosticResult:
    """
    Continuum wave analysis:
    - Asymptotic amplitude check (sqrt(2/π) normalization)
    - Phase shift extraction stability
    - High-L stability scan
    - Match point analysis
    """
    result = DiagnosticResult("continuum_analysis")
    print("\n" + "=" * 60)
    print(f"CONTINUUM WAVE DIAGNOSTIC: E = {E_eV} eV")
    print("=" * 60)
    
    # Setup
    atom_data = get_atom(atom)
    grid = make_r_grid(r_min=1e-5, r_max=200.0, n_points=5000)
    V_core = V_core_on_grid(grid, atom_data.core_params)
    
    # Get ground state for distorting potential
    states = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    orb_1s = states[0]
    
    k = k_from_E_eV(E_eV)
    U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k, k, use_exchange=False)
    
    result.add_data("E_eV", E_eV)
    result.add_data("k_au", float(k))
    result.add_data("L_max", L_max)
    
    target_amp = np.sqrt(2.0 / np.pi)  # δ(k-k') normalization
    
    print(f"\n{'L':>3} | {'Phase (rad)':>12} | {'Amp':>8} | {'r_match':>8} | {'Status'}")
    print("-" * 55)
    
    phase_shifts = []
    amplitudes = []
    
    for L in range(L_max + 1):
        try:
            with timed_section(result, f"solve_L{L}"):
                wave = solve_continuum_wave(grid, U_i, L, E_eV, z_ion=0.0)
            
            # Asymptotic amplitude
            tail_amp = np.max(np.abs(wave.chi_of_r[-int(len(grid.r)*0.1):]))
            amp_ok = target_amp * 0.85 < tail_amp < target_amp * 1.15
            
            phase_shifts.append(wave.phase_shift)
            amplitudes.append(tail_amp)
            
            status = "OK" if amp_ok else "WARN"
            print(f"{L:>3} | {wave.phase_shift:>12.6f} | {tail_amp:>8.4f} | {wave.r_match:>8.1f} | {status}")
            
            result.add_check(f"Amp L={L}", amp_ok, f"{tail_amp:.4f}", f"~{target_amp:.4f}")
            result.add_data(f"phase_L{L}", float(wave.phase_shift))
            result.add_data(f"r_match_L{L}", float(wave.r_match))
            
        except Exception as e:
            print(f"{L:>3} | ERROR: {str(e)[:40]}")
            result.add_check(f"Solve L={L}", False, str(e))
    
    result.add_data("phase_shifts", phase_shifts)
    result.add_data("amplitudes", amplitudes)
    
    save_result("continuum", result.summary())
    return result

# =============================================================================
# 3. POTENTIAL DIAGNOSTICS
# =============================================================================

def run_potential_diagnostic(atom_names: List[str] = ["H", "Li", "Na"]) -> DiagnosticResult:
    """
    Potential analysis for multiple atoms:
    - Core potential V(r) behavior
    - Effective charge Z_eff(r) = -r × V(r)
    - Distorting potential decay
    """
    result = DiagnosticResult("potential_analysis")
    print("\n" + "=" * 60)
    print("POTENTIAL DIAGNOSTIC")
    print("=" * 60)
    
    for name in atom_names:
        print(f"\n--- {name} ---")
        try:
            atom = get_atom(name)
            grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
            V = V_core_on_grid(grid, atom.core_params)
            
            # Z_eff at various radii
            r_test = [0.01, 0.1, 1.0, 5.0, 10.0]
            for r in r_test:
                idx = np.argmin(np.abs(grid.r - r))
                z_eff = -grid.r[idx] * V[idx]
                print(f"  Z_eff(r={r}) = {z_eff:.3f}")
                result.add_data(f"{name}_Zeff_r{r}", float(z_eff))
            
            # Asymptotic behavior
            z_asymp = -grid.r[-100] * V[-100]
            result.add_check(f"{name} asymptotic Z", abs(z_asymp - atom.core_params.Zc) < 0.1,
                           f"{z_asymp:.3f}", f"{atom.core_params.Zc}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            result.add_check(f"{name} analysis", False, str(e))
    
    save_result("potentials", result.summary())
    return result

# =============================================================================
# 4. RADIAL INTEGRAL DIAGNOSTICS
# =============================================================================

def run_radial_integral_diagnostic(E_eV: float = 50.0, method_compare: bool = True) -> DiagnosticResult:
    """
    Radial matrix element analysis:
    - I_L breakdown by multipole
    - Method comparison (legacy vs advanced)
    - Direct vs exchange contributions
    """
    result = DiagnosticResult("radial_integrals")
    print("\n" + "=" * 60)
    print(f"RADIAL INTEGRAL DIAGNOSTIC: E = {E_eV} eV")
    print("=" * 60)
    
    # Setup for H 1s -> 2s
    grid = make_r_grid(r_max=200.0, n_points=8000)
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    V_core = V_core_on_grid(grid, core_params)
    
    states_s = solve_bound_states(grid, V_core, l=0, n_states_max=5)
    orb_1s = min(states_s, key=lambda s: abs(s.energy_au + 0.5))
    orb_2s = min(states_s, key=lambda s: abs(s.energy_au + 0.125))
    
    E_f = E_eV - (orb_2s.energy_au - orb_1s.energy_au) * 27.211
    ki = k_from_E_eV(E_eV)
    kf = k_from_E_eV(E_f)
    
    U_i, U_f = build_distorting_potentials(grid, V_core, orb_1s, orb_2s, ki, kf, use_exchange=False)
    
    result.add_data("E_inc_eV", E_eV)
    result.add_data("E_final_eV", E_f)
    result.add_data("ki", float(ki))
    result.add_data("kf", float(kf))
    
    # Analyze for several L values
    print(f"\n{'L':>3} | {'I_L0':>12} | {'I_L1':>12} | {'I_L2':>12} | {'Sum':>12}")
    print("-" * 65)
    
    for L in [0, 1, 2, 3, 5, 7]:
        try:
            chi_i = solve_continuum_wave(grid, U_i, L, E_eV, z_ion=0.0)
            chi_f = solve_continuum_wave(grid, U_f, L, E_f, z_ion=0.0)
            
            if chi_i is None or chi_f is None:
                print(f"{L:>3} | Solver failed")
                continue
            
            res = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_1s, orb_2s, chi_i, chi_f, L_max=3)
            
            I0 = res.I_L_direct.get(0, 0.0)
            I1 = res.I_L_direct.get(1, 0.0)
            I2 = res.I_L_direct.get(2, 0.0)
            total = sum(abs(v) for v in res.I_L_direct.values())
            
            print(f"{L:>3} | {I0:>12.4e} | {I1:>12.4e} | {I2:>12.4e} | {total:>12.4e}")
            
            result.add_data(f"I_L{L}", {
                "I0": float(I0) if isinstance(I0, (int, float, np.number)) else complex(I0).real,
                "I1": float(I1) if isinstance(I1, (int, float, np.number)) else complex(I1).real,
                "I2": float(I2) if isinstance(I2, (int, float, np.number)) else complex(I2).real,
                "sum": float(total)
            })
            
        except Exception as e:
            print(f"{L:>3} | ERROR: {str(e)[:50]}")
    
    # Method comparison if requested
    if method_compare:
        print("\n--- Method Comparison (L=3) ---")
        try:
            chi_i = solve_continuum_wave(grid, U_i, 3, E_eV, z_ion=0.0)
            chi_f = solve_continuum_wave(grid, U_f, 3, E_f, z_ion=0.0)
            
            res_legacy = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_1s, orb_2s, 
                                         chi_i, chi_f, L_max=3, oscillatory_method="legacy")
            res_adv = radial_ME_all_L(grid, V_core, U_i.U_of_r, orb_1s, orb_2s,
                                      chi_i, chi_f, L_max=3, oscillatory_method="advanced")
            
            sum_leg = sum(abs(v) for v in res_legacy.I_L_direct.values())
            sum_adv = sum(abs(v) for v in res_adv.I_L_direct.values())
            ratio = sum_adv / sum_leg if sum_leg > 1e-15 else float('inf')
            
            print(f"  Legacy:   {sum_leg:.6e}")
            print(f"  Advanced: {sum_adv:.6e}")
            print(f"  Ratio:    {ratio:.4f}")
            
            result.add_check("Method agreement", 0.8 < ratio < 1.2, f"{ratio:.4f}", "~1.0")
            
        except Exception as e:
            print(f"  Method comparison failed: {e}")
    
    save_result("radial_integrals", result.summary())
    return result

# =============================================================================
# 5. ANGULAR COUPLING DIAGNOSTICS
# =============================================================================

def run_angular_diagnostic() -> DiagnosticResult:
    """
    Angular coupling coefficient verification:
    - Clebsch-Gordan coefficients
    - Racah W coefficients
    - Selection rules
    """
    result = DiagnosticResult("angular_coupling")
    print("\n" + "=" * 60)
    print("ANGULAR COUPLING DIAGNOSTIC")
    print("=" * 60)
    
    # Test CG coefficients
    print("\n--- Clebsch-Gordan Coefficients ---")
    # Note: Sign depends on Condon-Shortley phase convention
    # We check |CG| since physical observables use |CG|²
    test_cases = [
        # (j1, j2, j3, m1, m2, m3, expected_abs)
        (1, 1, 0, 0, 0, 0, 1/np.sqrt(3)),  # C(1,1,0;0,0,0) = ±1/√3
        (1, 1, 2, 0, 0, 0, np.sqrt(2/3)),  # C(1,1,2;0,0,0) = ±√(2/3)
        (1, 0, 1, 0, 0, 0, 1.0),           # C(1,0,1;0,0,0) = ±1
    ]
    
    for j1, j2, j3, m1, m2, m3, expected in test_cases:
        cg = clebsch_gordan(j1, j2, j3, m1, m2, m3)
        result.add_check(f"CG({j1},{j2},{j3};{m1},{m2},{m3})", 
                        abs(abs(cg) - expected) < 1e-6, f"{cg:.6f}", f"±{expected:.6f}")
    
    # Test Racah W
    print("\n--- Racah W Coefficients ---")
    # W(1,0,1,0;1,0) for 1s->2p transition
    w = racah_W(1, 0, 1, 0, 1, 0)
    print(f"  W(1,0,1,0;1,0) = {w:.6f}")
    result.add_data("W_1s_2p", float(w))
    
    # Test amplitude calculation
    print("\n--- Amplitude Channel Test ---")
    theta_grid = np.array([0.0, np.pi/4, np.pi/2, np.pi])
    I_L_dir = {1: 1.0}
    I_L_exc = {1: 0.5}
    
    for Mf in [-1, 0, 1]:
        amps = calculate_amplitude_contribution(
            theta_grid, I_L_dir, I_L_exc,
            0, 1, 1.0, 0.9, 0, 1, 0, Mf
        )
        f_0 = np.abs(amps.f_theta[0])**2
        print(f"  M_f={Mf:+d}: |f(0°)|² = {f_0:.6e}")
        result.add_data(f"amp_Mf{Mf}", float(f_0))
    
    save_result("angular", result.summary())
    return result

# =============================================================================
# 6. CROSS SECTION DIAGNOSTICS
# =============================================================================

def run_cross_section_diagnostic(E_eV: float = 50.0) -> DiagnosticResult:
    """
    Cross section analysis:
    - Partial wave contributions
    - Convergence analysis
    - Comparison with expected values
    """
    result = DiagnosticResult("cross_sections")
    print("\n" + "=" * 60)
    print(f"CROSS SECTION DIAGNOSTIC: E = {E_eV} eV")
    print("=" * 60)
    
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    
    # H 1s -> 2s excitation
    print("\n--- H 1s -> 2s Excitation ---")
    spec = ExcitationChannelSpec(
        l_i=0, l_f=0, n_index_i=1, n_index_f=2,
        N_equiv=1, L_max_integrals=10, L_target_i=0, L_target_f=0, L_max_projectile=15
    )
    
    with timed_section(result, "compute_sigma"):
        res = compute_total_excitation_cs(E_eV, spec, core_params, n_points=3000, n_theta=200)
    
    sigma_cm2 = res.sigma_total_cm2
    print(f"  σ_total = {sigma_cm2:.4e} cm²")
    
    # Expected order of magnitude for H 1s->2s at 50 eV: ~4e-18 cm²
    order_ok = 1e-19 < sigma_cm2 < 1e-16
    result.add_check("Sigma magnitude", order_ok, f"{sigma_cm2:.2e}", "~1e-18 to 1e-17 cm²")
    
    result.add_data("sigma_total_cm2", float(sigma_cm2))
    result.add_data("E_eV", E_eV)
    
    save_result("cross_sections", result.summary())
    return result

# =============================================================================
# 7. CONVERGENCE STUDIES
# =============================================================================

def run_convergence_study(E_eV: float = 50.0) -> DiagnosticResult:
    """
    Convergence analysis:
    - Theta grid convergence
    - Radial grid convergence
    - L_max convergence
    """
    result = DiagnosticResult("convergence")
    print("\n" + "=" * 60)
    print(f"CONVERGENCE STUDY: E = {E_eV} eV")
    print("=" * 60)
    
    core_params = CorePotentialParams(Zc=1.0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    spec = ExcitationChannelSpec(0, 0, 1, 2, 1, 10, 0, 0, 10)
    
    # 1. Theta convergence
    print("\n--- Theta Grid Convergence ---")
    theta_values = [50, 100, 200, 500]
    sigma_theta = []
    
    for n_theta in theta_values:
        res = compute_total_excitation_cs(E_eV, spec, core_params, n_points=3000, n_theta=n_theta)
        sigma_theta.append(res.sigma_total_cm2)
        print(f"  n_theta={n_theta:4d}: σ = {res.sigma_total_cm2:.6e}")
    
    theta_diff = abs(sigma_theta[-1] - sigma_theta[-2]) / sigma_theta[-1] * 100
    result.add_check("Theta converged", theta_diff < 1.0, f"{theta_diff:.3f}%", "<1%")
    result.add_data("sigma_vs_theta", list(zip(theta_values, sigma_theta)))
    
    # 2. Radial grid convergence
    print("\n--- Radial Grid Convergence ---")
    grid_values = [1000, 2000, 3000, 5000]
    sigma_grid = []
    
    for n_pts in grid_values:
        res = compute_total_excitation_cs(E_eV, spec, core_params, n_points=n_pts, n_theta=200)
        sigma_grid.append(res.sigma_total_cm2)
        print(f"  n_points={n_pts:5d}: σ = {res.sigma_total_cm2:.6e}")
    
    grid_diff = abs(sigma_grid[-1] - sigma_grid[-2]) / sigma_grid[-1] * 100
    result.add_check("Grid converged", grid_diff < 5.0, f"{grid_diff:.3f}%", "<5%")
    result.add_data("sigma_vs_grid", list(zip(grid_values, sigma_grid)))
    
    save_result("convergence", result.summary())
    return result

# =============================================================================
# 8. QUICK HEALTH CHECK
# =============================================================================

def quick_health_check() -> DiagnosticResult:
    """Quick integration test for excitation and ionization."""
    result = DiagnosticResult("health_check")
    print("\n" + "=" * 60)
    print("QUICK HEALTH CHECK")
    print("=" * 60)
    
    atom = get_atom("H")
    
    # Excitation
    print("\n1. Excitation (H 1s -> 2s at 50 eV)...")
    try:
        spec = ExcitationChannelSpec(0, 0, 1, 2, 1, 15, 0, 0)
        res = compute_total_excitation_cs(50.0, spec, atom.core_params, n_points=2000, r_max=100.0)
        result.add_check("Excitation runs", True, f"σ = {res.sigma_total_cm2:.3e} cm²")
    except Exception as e:
        result.add_check("Excitation runs", False, str(e))
    
    # Ionization
    print("\n2. Ionization (H 1s at 50 eV)...")
    try:
        from ionization import compute_ionization_cs, IonizationChannelSpec
        spec_ion = IonizationChannelSpec(0, 1, 1, l_eject_max=1, L_max=10, L_i_total=0)
        res_ion = compute_ionization_cs(50.0, spec_ion, atom.core_params, n_points=2000, n_energy_steps=1)
        result.add_check("Ionization runs", True, f"σ = {res_ion.sigma_total_cm2:.3e} cm²")
    except Exception as e:
        result.add_check("Ionization runs", False, str(e))
    
    save_result("health_check", result.summary())
    return result

# =============================================================================
# SPECIALIZED DIAGNOSTICS (from existing scripts)
# =============================================================================

def _run_external_script(script_path: str, label: str):
    """
    Execute an external debug script as __main__ with consistent error handling.
    """
    print(f"\n[INFO] Running {label}...")
    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()


def run_phase_extraction_diagnostic():
    """Run diag_phase_extraction.py - analyzes phase extraction for H 1s→2s."""
    print("\n[INFO] Running phase extraction diagnostic...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("diag_phase", "debug/diag_phase_extraction.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run_full_diagnostic()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()


def run_phase_method_comparison() -> DiagnosticResult:
    """
    Compare log-derivative vs LSQ phase extraction methods (v2.11+).
    Tests both methods on synthetic and real potential data.
    """
    result = DiagnosticResult("phase_method_comparison")
    print("\n" + "=" * 60)
    print("PHASE EXTRACTION METHOD COMPARISON")
    print("=" * 60)
    
    from continuum import (
        _extract_phase_logderiv_neutral,
        _fit_asymptotic_phase_neutral,
        solve_continuum_wave
    )
    
    # 1. Synthetic data test (known phase shift)
    print("\n--- Synthetic Data Test (Known δ) ---")
    known_phases = [0.0, 0.3, 0.5, -0.5]
    k_test = 1.5
    L_test = 0
    r_test = np.linspace(50, 200, 500)
    A_true = np.sqrt(2/np.pi)
    
    for delta_true in known_phases:
        chi_synth = A_true * np.sin(k_test * r_test - L_test * np.pi/2 + delta_true)
        dchi_synth = A_true * k_test * np.cos(k_test * r_test - L_test * np.pi/2 + delta_true)
        
        # LSQ
        A_lsq, delta_lsq = _fit_asymptotic_phase_neutral(r_test, chi_synth, L_test, k_test)
        
        # Log-derivative at midpoint
        idx_mid = len(r_test) // 2
        Y_mid = dchi_synth[idx_mid] / chi_synth[idx_mid]
        delta_ld = _extract_phase_logderiv_neutral(Y_mid, k_test, r_test[idx_mid], L_test)
        
        err_lsq = abs(delta_true - delta_lsq)
        err_ld = abs(delta_true - delta_ld)
        err_ld = min(err_ld, 2*np.pi - err_ld)
        
        lsq_ok = err_lsq < 0.01
        ld_ok = err_ld < 0.1
        
        result.add_check(f"LSQ δ={delta_true:+.1f}", lsq_ok, f"err={err_lsq:.6f}")
        result.add_check(f"LD δ={delta_true:+.1f}", ld_ok, f"err={err_ld:.6f}")
        
        print(f"  δ_true={delta_true:+.2f} | LSQ: {delta_lsq:+.4f} (err={err_lsq:.6f}) | LD: {delta_ld:+.4f} (err={err_ld:.6f})")
    
    # 2. Real potential test
    print("\n--- Real Potential Test (H) ---")
    atom = get_atom("H")
    grid = make_r_grid(200.0, 5000)
    V_core = V_core_on_grid(grid, atom.core_params)
    states = solve_bound_states(grid, V_core, l=0, n_states_max=2)
    orb_1s = states[0]
    k = k_from_E_eV(50.0)
    U_i, _ = build_distorting_potentials(grid, V_core, orb_1s, orb_1s, k, k, use_exchange=False)
    
    print(f"\n  {'L':>3} | {'δ_hybrid':>10} | {'δ_logderiv':>10} | {'δ_lsq':>10} | {'Method Used':<15}")
    print("  " + "-" * 65)
    
    for L in [0, 1, 2, 5]:
        try:
            cw_hybrid = solve_continuum_wave(grid, U_i, L, 50.0, phase_extraction_method="hybrid")
            cw_ld = solve_continuum_wave(grid, U_i, L, 50.0, phase_extraction_method="logderiv")
            cw_lsq = solve_continuum_wave(grid, U_i, L, 50.0, phase_extraction_method="lsq")
            
            if cw_hybrid and cw_ld and cw_lsq:
                diff = abs(cw_ld.phase_shift - cw_lsq.phase_shift)
                diff = min(diff, 2*np.pi - diff)
                result.add_check(f"L={L} agreement", diff < 0.1, f"Δ={diff:.4f}")
                print(f"  {L:>3} | {cw_hybrid.phase_shift:>+10.6f} | {cw_ld.phase_shift:>+10.6f} | {cw_lsq.phase_shift:>+10.6f} | hybrid")
        except Exception as e:
            print(f"  {L:>3} | ERROR: {str(e)[:40]}")
    
    save_result("phase_methods", result.summary())
    return result


def run_L0_L1_anomaly_diagnostic():
    """Run diag_L0_L1_anomaly.py"""
    print("\n[INFO] Running L0/L1 anomaly diagnostic...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("diag_anomaly", "debug/diag_L0_L1_anomaly.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run_diagnostic()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()

def run_upturn_diagnostic():
    """Run diag_upturn.py"""
    _run_external_script("debug/diag_upturn.py", "upturn diagnostic (69.26 eV)")

def run_atoms_diagnostic():
    """Run diag_atoms.py"""
    _run_external_script("debug/diag_atoms.py", "multi-atom diagnostic")

def run_method_comparison():
    """Run diag_method_compare.py - compare oscillatory integral methods."""
    _run_external_script("debug/diag_method_compare.py", "legacy vs advanced method comparison")


def run_phase_method_comparison_deep():
    """Run full diagnostic from diag_phase_methods_compare.py."""
    _run_external_script("debug/diag_phase_methods_compare.py", "deep phase-method comparison")


def run_upturn_hypothesis_deep():
    """Run deep upturn hypothesis test sweep."""
    _run_external_script("debug/deep_hypothesis_test.py", "deep upturn hypotheses")


def run_performance_profile():
    """Run performance profiler script."""
    _run_external_script("debug/profile_performance.py", "performance profiler")


def run_all_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL DIAGNOSTICS")
    print("=" * 60)
    
    results = []
    
    results.append(quick_health_check())
    results.append(run_bound_state_diagnostic())
    results.append(run_continuum_diagnostic())
    results.append(run_potential_diagnostic())
    results.append(run_radial_integral_diagnostic())
    results.append(run_angular_diagnostic())
    results.append(run_cross_section_diagnostic())
    results.append(run_convergence_study())
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_checks = 0
    
    for r in results:
        s = r.summary()
        total_passed += s["passed"]
        total_checks += s["total"]
        status = "✓ PASS" if s["passed"] == s["total"] else "✗ FAIL"
        print(f"  {s['name']}: {s['passed']}/{s['total']} checks {status}")
    
    print(f"\n  TOTAL: {total_passed}/{total_checks} checks passed")
    print(f"  SUCCESS RATE: {total_passed/total_checks*100:.1f}%" if total_checks > 0 else "  No checks run")

# =============================================================================
# MAIN CLI
# =============================================================================

def print_menu():
    """Print the main diagnostic menu."""
    print("\n" + "=" * 60)
    print("  DWBA COMPREHENSIVE DIAGNOSTIC SUITE")
    print("=" * 60)
    print("\n[QUICK TESTS]")
    print("  1. Quick Health Check")
    print("  2. Convergence Study")
    print("\n[BOUND STATES]")
    print("  3. Bound State Analysis")
    print("\n[CONTINUUM WAVES]")
    print("  4. Continuum Wave Analysis")
    print("  5. Phase Extraction Diagnostic (Energy Scan)")
    print("  6. Phase Method Comparison (v2.11+)")
    print("  7. High-L Stability Scan")
    print("\n[POTENTIALS]")
    print("  8. Potential Analysis")
    print("  9. Multi-Atom Comparison")
    print("\n[RADIAL INTEGRALS]")
    print(" 10. Radial Integral Breakdown")
    print(" 11. Method Comparison (legacy vs advanced)")
    print("\n[ANGULAR COUPLING]")
    print(" 12. Angular Coefficient Verification")
    print("\n[CROSS SECTIONS]")
    print(" 13. Cross Section Analysis")
    print(" 14. L0/L1 Anomaly Investigation")
    print(" 15. High-Energy Upturn Analysis")
    print("\n[ADVANCED]")
    print(" 16. Deep Phase-Method Comparison")
    print(" 17. Deep Upturn Hypotheses")
    print(" 18. Performance Profiler")
    print("\n[BATCH]")
    print(" 20. Run ALL Diagnostics")
    print("\n  q. Quit")

def main():
    """Main entry point."""
    ensure_output_dirs()
    
    while True:
        print_menu()
        choice = input("\nSelect Option: ").strip().lower()
        
        try:
            if choice == '1':
                quick_health_check()
            elif choice == '2':
                run_convergence_study()
            elif choice == '3':
                run_bound_state_diagnostic()
            elif choice == '4':
                run_continuum_diagnostic()
            elif choice == '5':
                run_phase_extraction_diagnostic()
            elif choice == '6':
                run_phase_method_comparison()
            elif choice == '7':
                run_continuum_diagnostic(L_max=30)
            elif choice == '8':
                run_potential_diagnostic(["H"])
            elif choice == '9':
                run_atoms_diagnostic()
            elif choice == '10':
                run_radial_integral_diagnostic(method_compare=False)
            elif choice == '11':
                run_method_comparison()
            elif choice == '12':
                run_angular_diagnostic()
            elif choice == '13':
                run_cross_section_diagnostic()
            elif choice == '14':
                run_L0_L1_anomaly_diagnostic()
            elif choice == '15':
                run_upturn_diagnostic()
            elif choice == '16':
                run_phase_method_comparison_deep()
            elif choice == '17':
                run_upturn_hypothesis_deep()
            elif choice == '18':
                run_performance_profile()
            elif choice == '20':
                run_all_diagnostics()
            elif choice == 'q':
                print("Goodbye.")
                break
            else:
                print("Invalid option. Please try again.")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
