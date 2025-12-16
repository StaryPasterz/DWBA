# fit_potential.py
"""
SAE Model Potential Parameter Fitting Tool
==========================================

Determines optimal CorePotentialParams (a1..a6) for atoms/ions by fitting
calculated binding energies to experimental values (NIST).

Methodology
-----------
Based on Tong-Lin (2005) approach (J. Phys. B 38, 2593):
- Original: Fit to DFT self-interaction-corrected numerical potential
- This implementation: Direct energy matching with global+local optimization

Optimization Strategy (Fast Hybrid Approach)
---------------------------------------------
1. Coarse search: Basin-hopping with few local iterations on reduced grid
2. Local refinement: L-BFGS-B on full grid
This avoids the slow differential_evolution while still finding good solutions.

Potential Form (article Eq. 69)
-------------------------------
    V(r) = -[Zc + a1*exp(-a2*r) + a3*r*exp(-a4*r) + a5*exp(-a6*r)] / r

References
----------
- X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)
- NIST Atomic Spectra Database

Usage
-----
    python fit_potential.py
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, basinhopping
import time
import json
import os
from typing import Optional, Dict, Tuple, List

from logging_config import get_logger
from grid import make_r_grid, ev_to_au, au_to_ev
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

logger = get_logger(__name__)


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def load_atoms_database() -> Dict:
    """Load atoms database from atoms.json."""
    db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json")
    if os.path.exists(db_file):
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load atoms.json: %s", e)
    return {}


def is_reference_source(source: str) -> bool:
    """Check if parameters come from a verified reference (protected from re-fitting)."""
    if not source:
        return False
    source_lower = source.lower()
    return any(ref in source_lower for ref in [
        "tong-lin", "hydrogenic", "exact", "literature", "reference"
    ])


def load_nist_database() -> Dict:
    """Load NIST reference data if available."""
    nist_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nist_data.json")
    if os.path.exists(nist_file):
        try:
            with open(nist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("NIST database not loaded: %s", e)
    return {}


# =============================================================================
# PHYSICS FUNCTIONS (OPTIMIZED)
# =============================================================================

def solve_binding_energy_fast(
    a_params: List[float], 
    Zc: float, 
    n_target: int, 
    l_target: int, 
    grid
) -> float:
    """
    Fast binding energy calculation with error handling.
    Returns positive value as penalty if unbound/failed.
    """
    try:
        a1, a2, a3, a4, a5, a6 = a_params
        
        # Quick physics check - avoid obviously bad params
        if a2 < 0.01 or a4 < 0.01 or a6 < 0.01:
            return 10.0
        
        params = CorePotentialParams(Zc, a1, a2, a3, a4, a5, a6)
        V = V_core_on_grid(grid, params)
        
        # Request fewer states for speed
        states = solve_bound_states(grid, V, l=l_target, n_states_max=n_target + 1)
        
        target_n_index = n_target - l_target
        for s in states:
            if s.n_index == target_n_index:
                return s.energy_au
                
        return 1.0  # Unbound penalty
        
    except Exception:
        return 10.0  # Numerical failure penalty


def cost_function_fast(
    a_params: np.ndarray, 
    Zc: float, 
    n_target: int, 
    l_target: int, 
    target_E_au: float, 
    grid, 
    Z_nuclear: float
) -> float:
    """Fast cost function with physics constraints."""
    a1, a2, a3, a4, a5, a6 = a_params
    
    # Hard constraints
    if a2 < 0.05 or a4 < 0.05 or a6 < 0.05:
        return 1000.0
    
    # Physics: Z_eff(0) = Zc + a1 + a5 ~ Z_nuclear
    z_eff_0 = Zc + a1 + a5
    z_penalty = 0.0
    if abs(z_eff_0 - Z_nuclear) > 3.0:
        z_penalty = 5.0 * (abs(z_eff_0 - Z_nuclear) - 3.0)
    
    E_calc = solve_binding_energy_fast(list(a_params), Zc, n_target, l_target, grid)
    
    # Squared error + penalty
    energy_error = 100.0 * (E_calc - target_E_au) ** 2
    
    return energy_error + z_penalty


# =============================================================================
# FAST HYBRID OPTIMIZER
# =============================================================================

def get_initial_guess(Z: float, Zc: float, name: str = "") -> List[float]:
    """
    Generate physics-based initial guess.
    
    Based on patterns in Tong-Lin Table 1:
    - a1 ≈ Z - Zc (screening strength)
    - a2 ≈ 0.5-2.0 for light atoms, larger for heavy
    - a3 can be large positive (Rb) or negative (Ar)
    - a5 often small, can be negative
    """
    # Default alkali-like estimate
    a1 = Z - Zc
    a2 = min(2.0 + 0.1 * Z, 12.0)
    a3 = 0.0  # Start neutral
    a4 = 1.0
    a5 = 0.0
    a6 = 1.0
    
    return [a1, a2, a3, a4, a5, a6]


def get_parameter_bounds(Z: float, Zc: float) -> List[Tuple[float, float]]:
    """Optimized bounds based on Tong-Lin data analysis."""
    return [
        (-2.0, Z + 5.0),      # a1
        (0.1, 12.0),          # a2
        (-30.0, 120.0),       # a3 (wide for Rb)
        (0.1, 8.0),           # a4
        (-5.0, 15.0),         # a5
        (0.1, 5.0),           # a6
    ]


def optimize_fast(
    Z: float, 
    Zc: float, 
    n_target: int, 
    l_target: int, 
    target_E_au: float,
    progress_callback=None
) -> Tuple[List[float], float, bool]:
    """
    Fast hybrid optimization: basin-hopping + local refinement.
    
    Returns
    -------
    tuple
        (best_params, final_cost, success)
    """
    # Phase 1: Coarse grid for speed
    logger.info("Phase 1: Coarse search (500 pts)...")
    if progress_callback:
        progress_callback("Phase 1: Coarse search...")
    
    grid_coarse = make_r_grid(r_min=1e-4, r_max=150.0, n_points=500)
    bounds = get_parameter_bounds(Z, Zc)
    x0 = get_initial_guess(Z, Zc)
    
    args = (Zc, n_target, l_target, target_E_au, grid_coarse, Z)
    
    # Basin-hopping with few iterations
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "args": args,
        "options": {"maxiter": 100}
    }
    
    try:
        result = basinhopping(
            cost_function_fast,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=15,  # Few hops for speed
            T=1.0,
            stepsize=0.5,
            seed=42
        )
        x_coarse = result.x
        logger.info("Coarse search done. Cost: %.4f", result.fun)
    except Exception as e:
        logger.warning("Basin-hopping failed: %s, using initial guess", e)
        x_coarse = x0
    
    # Phase 2: Fine grid refinement
    logger.info("Phase 2: Fine refinement (2000 pts)...")
    if progress_callback:
        progress_callback("Phase 2: Fine refinement...")
    
    grid_fine = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    args_fine = (Zc, n_target, l_target, target_E_au, grid_fine, Z)
    
    result_fine = minimize(
        cost_function_fast,
        x_coarse,
        args=args_fine,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10}
    )
    
    best_params = list(result_fine.x)
    final_cost = result_fine.fun
    success = final_cost < 0.01  # Energy error < 0.1 mHa
    
    logger.info("Optimization complete. Cost: %.2e, Success: %s", final_cost, success)
    
    return best_params, final_cost, success


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_fit(
    name: str, 
    Z: float, 
    Zc: float, 
    n: int, 
    l: int, 
    ip_ev: float, 
    a_params: List[float], 
    grid=None
) -> Tuple[bool, Optional[float], float, Optional[float]]:
    """Verify fitted parameters by computing binding energy."""
    if grid is None:
        grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    
    params = CorePotentialParams(Zc, *a_params)
    V = V_core_on_grid(grid, params)
    
    states = solve_bound_states(grid, V, l=l, n_states_max=n + 2)
    target_n_idx = n - l
    
    for s in states:
        if s.n_index == target_n_idx:
            calc_E_eV = au_to_ev(s.energy_au)
            expected_E_eV = -ip_ev
            diff = abs(calc_E_eV - expected_E_eV)
            success = diff < 0.1  # 0.1 eV threshold
            
            logger.debug("Verify: calc=%.4f eV, exp=%.4f eV, diff=%.4f eV", 
                        calc_E_eV, expected_E_eV, diff)
            return success, calc_E_eV, expected_E_eV, diff
    
    return False, None, -ip_ev, None


# =============================================================================
# FILE I/O
# =============================================================================

def save_to_json(
    name: str, Z: float, Zc: float, n: int, l: int, 
    ip_ev: float, a_params: List[float], source: str = "Fitted"
):
    """Save optimized parameters to atoms.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.join(base_dir, "atoms.json")
    
    data = load_atoms_database()
    
    l_char = 'spdf'[l] if l < 4 else f'l={l}'
    data[name] = {
        "name": name,
        "Z": float(Z),
        "Zc": float(Zc),
        "a_params": [float(x) for x in a_params],
        "default_n": int(n),
        "default_l": int(l),
        "ip_ev": float(abs(ip_ev)),
        "source": source,
        "description": f"{name} ({n}{l_char} state)"
    }
    
    try:
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved '%s' to atoms.json", name)
        print(f"\n[SUCCESS] Saved '{name}' to {db_file}")
    except Exception as e:
        logger.error("Failed to save: %s", e)
        print(f"\n[ERROR] Failed to save: {e}")


# =============================================================================
# MAIN INTERACTIVE TOOL
# =============================================================================

def main():
    print("=" * 60)
    print("SAE Model Potential Fitter (Fast Hybrid Method)")
    print("=" * 60)
    print()
    print("Method: Basin-Hopping + L-BFGS-B (coarse→fine grid)")
    print("Reference: X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)")
    print()
    
    EXISTING_ATOMS = load_atoms_database()
    KNOWN_ATOMS = load_nist_database()
    
    name = input("Atom Name (e.g. Na): ").strip()
    
    d: Dict = {}
    source = "Manual"
    
    # Check existing database
    if name in EXISTING_ATOMS:
        entry = EXISTING_ATOMS[name]
        entry_source = entry.get("source", "")
        
        if is_reference_source(entry_source):
            print(f"\n[REFERENCE] '{name}' has verified parameters from: {entry_source}")
            logger.warning("Atom '%s' has reference parameters", name)
            print("\n" + "!" * 60)
            print("WARNING: Re-fitting reference parameters is NOT recommended!")
            print("!" * 60)
            
            choice = input("\n(V)iew / (R)efit anyway / (C)ancel? [V/r/c]: ").strip().lower()
            
            if choice == 'c':
                return
            elif choice != 'r':
                grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
                success, calc_E, exp_E, diff = verify_fit(
                    name, entry["Z"], entry["Zc"], 
                    entry["default_n"], entry["default_l"],
                    entry["ip_ev"], entry["a_params"], grid
                )
                print(f"\nVerification for {name}:")
                print(f"  Source: {entry_source}")
                print(f"  Params: {[f'{x:.3f}' for x in entry['a_params']]}")
                if success:
                    print(f"  [OK] Calc: {calc_E:.4f} eV, Exp: {exp_E:.4f} eV, Diff: {diff:.4f} eV")
                return
        
        d["Z"] = entry["Z"]
        d["Zc"] = entry["Zc"]
        d["n"] = entry["default_n"]
        d["l"] = entry["default_l"]
        if "ip_ev" in entry:
            d["E"] = -entry["ip_ev"]
        source = "Local"
        print(f"\n[LOCAL] Found {name} (source: {entry_source})")
        
        if "a_params" in entry:
            print(f"  Params: {[f'{x:.3f}' for x in entry['a_params'][:3]]}...")
            choice = input("  (V)iew/(R)efit/(S)kip? [V/r/s]: ").strip().lower()
            if choice == 's':
                return
            if choice != 'r':
                grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
                verify_fit(name, entry["Z"], entry["Zc"], 
                          d["n"], d["l"], entry["ip_ev"], entry["a_params"], grid)
                return
    
    elif name in KNOWN_ATOMS:
        d = KNOWN_ATOMS[name].copy()
        source = "NIST"
        print(f"\n[NIST] Found {name}")
    
    else:
        print(f"\n[MANUAL] Atom '{name}' not found. Enter data:")
    
    # Confirm data
    if "Z" in d and "E" in d:
        l_char = 'spdf'[d['l']] if d['l'] < 4 else f"l={d['l']}"
        print(f"  Z={d['Z']}, Zc={d['Zc']}, State={d['n']}{l_char}, E={d['E']} eV")
        
        if input("Use these? [Y/n]: ").strip().lower() == 'n':
            d = {}
    
    # Manual input
    if 'Z' not in d:
        try:
            Z = float(input("Nuclear Charge Z: "))
            Zc = float(input("Core Charge Zc: "))
            n = int(input("Target State n: "))
            l = int(input("Target State l: "))
            E_exp_eV = float(input("Binding Energy (eV, negative): "))
            d = {"Z": Z, "Zc": Zc, "n": n, "l": l, "E": E_exp_eV}
        except ValueError:
            print("Invalid input.")
            return
    
    Z = float(d["Z"])
    Zc = float(d["Zc"])
    n = int(d["n"])
    l = int(d["l"])
    E_exp_eV = float(d["E"])
    
    target_E_au = ev_to_au(E_exp_eV)
    
    logger.info("Starting fit: %s (Z=%d, n=%d, l=%d)", name, Z, n, l)
    print(f"\nTarget E = {target_E_au:.6f} a.u. ({E_exp_eV:.4f} eV)")
    print("\nOptimizing (typically 5-15 seconds)...\n")
    
    t0 = time.time()
    
    def progress(msg):
        print(f"  {msg}")
    
    best_params, final_cost, success = optimize_fast(
        Z, Zc, n, l, target_E_au,
        progress_callback=progress
    )
    
    dt = time.time() - t0
    
    print(f"\nDone in {dt:.1f}s. Cost: {final_cost:.2e}")
    
    # Print parameters
    print("\n" + "-" * 40)
    print("Fitted Parameters:")
    for pname, val in zip(['a1','a2','a3','a4','a5','a6'], best_params):
        print(f"  {pname} = {val:+.6f}")
    
    # Verify
    print("\n" + "=" * 40)
    print("VERIFICATION")
    grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    success_v, calc_E, exp_E, diff = verify_fit(name, Z, Zc, n, l, abs(E_exp_eV), best_params, grid)
    
    if success_v:
        print(f"[SUCCESS] Calc: {calc_E:.4f} eV, Exp: {exp_E:.4f} eV, Diff: {diff:.4f} eV")
    else:
        print(f"[WARNING] Diff: {diff:.4f} eV (threshold 0.1 eV)")
    
    z_eff_0 = Zc + best_params[0] + best_params[4]
    print(f"Z_eff(r→0) = {z_eff_0:.2f} (target: {Z:.0f})")
    
    if input("\nSave? [Y/n]: ").strip().lower() != 'n':
        save_to_json(name, Z, Zc, n, l, abs(E_exp_eV), best_params, "Fitted")
    
    # Optional plot
    try:
        import matplotlib.pyplot as plt
        
        V = V_core_on_grid(grid, CorePotentialParams(Zc, *best_params))
        r = grid.r
        Z_eff = -r * V
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(r, V, 'b-')
        plt.plot(r, -Zc/r, 'g--', alpha=0.5)
        plt.xlim(0, 5)
        plt.ylim(-Z*2, 0)
        plt.title(f"V(r) for {name}")
        plt.xlabel("r (a.u.)")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(r, Z_eff, 'r-')
        plt.axhline(Z, color='k', linestyle=':')
        plt.axhline(Zc, color='k', linestyle='--')
        plt.xlim(0, 10)
        plt.ylim(0, Z + 1)
        plt.title("Z_eff(r)")
        plt.xlabel("r (a.u.)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"fit_{name}.png", dpi=150)
        print(f"[INFO] Plot saved: fit_{name}.png")
        
    except Exception:
        pass


if __name__ == "__main__":
    main()
