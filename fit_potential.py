# fit_potential.py
#
# Tool to determine optimal CorePotentialParams (a1..a6) for a new atom/ion
# by fitting the calculated binding energy to the experimental value (NIST).
#
# Usage:
#   python fit_potential.py
#
# It will ask for:
#   - Atom Name (e.g. "Na")
#   - Nuclear Charge Z (e.g. 11)
#   - Core Charge Zc (e.g. 1 for neutral, 2 for +1 ion)
#   - Target State n, l (e.g. 3, 0 for Na 3s)
#   - Experimental Binding Energy (e.g. -5.139 eV)
#
# Then it runs an optimizer (Nelder-Mead) to minimize |E_calc - E_exp|.
# Finally, it prints the code snippet to add to atom_library.py.
#

import numpy as np
import scipy.optimize
import time

# Import our physics engine
from grid import make_r_grid, ev_to_au, au_to_ev
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states

def solve_binding_energy(a_params, Zc, n_target, l_target, grid):
    """
    Calculates the binding energy of the target state (n, l)
    given the potential parameters [a1, ..., a6].
    """
    # Unpack params
    # We restrict params to be reasonable (mostly positive decay rates)
    # but the optimizer handles the search.
    a1, a2, a3, a4, a5, a6 = a_params
    
    # Construct parameter object
    params = CorePotentialParams(Zc, a1, a2, a3, a4, a5, a6)
    
    try:
        # Build potential
        V = V_core_on_grid(grid, params)
        
        # Solve bound states for specific l
        states = solve_bound_states(grid, V, l=l_target, n_states_max=n_target+1)
        
        # Find the state with correct n_index
        # Note: solve_bound_states returns n_index=1, 2, ...
        # For Lithium 2s: l=0.
        #   n=1 (1s) -> n_index=1 (Core, usually deeply bound)
        #   n=2 (2s) -> n_index=2 (Valence)
        # So we look for state with n_index == (n_target - l_target)
        # E.g. Na 3s (n=3, l=0): n_idx = 3.
        
        target_n_index = n_target - l_target
        
        for s in states:
            if s.n_index == target_n_index:
                return s.energy_au
                
        # If not found (unbound?), return a penalty
        return 1.0 # Positive energy (unbound) penalty
        
    except Exception:
        return 10.0 # Interaction failure penalty

def cost_function(a_params, Zc, n_target, l_target, target_E_au, grid, Z_nuclear=None):
    """
    Cost = |E_calc - E_target| + physics penalties
    """
    a1, a2, a3, a4, a5, a6 = a_params
    
    # Enforce constraints via penalty
    # Decay rates a2, a4, a6 must be positive > 0.01 to be physical
    if a2 < 0.01 or a4 < 0.01 or a6 < 0.01:
        return 100.0 + abs(a2)+abs(a4)+abs(a6)
    
    # Physics constraint: Z_eff(0) = Zc + a1 + a5 should be close to Z_nuclear
    if Z_nuclear is not None:
        z_eff_0 = Zc + a1 + a5
        if abs(z_eff_0 - Z_nuclear) > 1.0:
            return 50.0 + abs(z_eff_0 - Z_nuclear)

    E_calc = solve_binding_energy(a_params, Zc, n_target, l_target, grid)
    
    diff = abs(E_calc - target_E_au)
    return diff


def verify_fit(name, Z, Zc, n, l, ip_ev, a_params):
    """
    Verify the fitted parameters by computing bound state energy.
    Returns (success, calculated_E_eV, expected_E_eV, diff_eV)
    """
    from grid import make_r_grid, au_to_ev
    from potential_core import CorePotentialParams, V_core_on_grid
    from bound_states import solve_bound_states
    
    grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    params = CorePotentialParams(Zc, *a_params)
    V = V_core_on_grid(grid, params)
    
    states = solve_bound_states(grid, V, l=l, n_states_max=n+2)
    target_n_idx = n - l
    
    for s in states:
        if s.n_index == target_n_idx:
            calc_E_eV = s.energy_au * 27.211386
            expected_E_eV = -ip_ev
            diff = abs(calc_E_eV - expected_E_eV)
            success = diff < 0.1  # Within 0.1 eV is success
            return success, calc_E_eV, expected_E_eV, diff
    
    return False, None, -ip_ev, None

import json
import os

# ... optimization code ...

def save_to_json(name, Z, Zc, n, l, ip_ev, a_params):
    """
    Saves the optimize parameters to atoms.json
    """
    # Use absolute path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.join(base_dir, "atoms.json")
    
    data = {}
    if os.path.exists(db_file):
        try:
            with open(db_file, "r") as f:
                data = json.load(f)
        except:
            data = {}
            
    # Update/Add entry
    data[name] = {
        "name": name,
        "Z": float(Z),
        "Zc": float(Zc),
        "a_params": [float(x) for x in a_params],
        "default_n": int(n),
        "default_l": int(l),
        "ip_ev": float(abs(ip_ev)),
        "description": f"{name} ({n}{'spdf'[l] if l<4 else 'l='+str(l)} matched to NIST)"
    }
    
    try:
        with open(db_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[SUCCESS] Automatically saved '{name}' to:\n  {db_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save atoms.json: {e}")

def main():
    print("=== Model Potential Fitter (Self-Consistent) ===")
    print("This tool finds parameters a1..a6 to match NIST energies.")
    
    # --- 1. LOAD LOCAL DATABASE (atoms.json) ---
    atoms_file = os.path.join(os.path.dirname(__file__), "atoms.json")
    EXISTING_ATOMS = {}
    if os.path.exists(atoms_file):
        try:
            with open(atoms_file, "r") as f:
                 EXISTING_ATOMS = json.load(f)
        except:
            pass

    # --- 2. LOAD INTERNET/REFERENCE DATABASE (nist_data.json) ---
    # In a real app, this might be a web fetch. Here we treat this file as "The Cloud".
    known_file = os.path.join(os.path.dirname(__file__), "nist_data.json")
    KNOWN_ATOMS = {}
    if os.path.exists(known_file):
        try:
            with open(known_file, "r") as f:
                KNOWN_ATOMS = json.load(f)
        except:
            pass 

    name = input("Atom Name (e.g. Na): ").strip()
    
    d = {}
    source = "None"

    # --- PRIORITY CHECK ---
    
    # Priority A: Local File
    if name in EXISTING_ATOMS:
        entry = EXISTING_ATOMS[name]
        d["Z"] = entry["Z"]
        d["Zc"] = entry["Zc"]
        d["n"] = entry["default_n"]
        d["l"] = entry["default_l"]
        if "ip_ev" in entry:
            d["E"] = -entry["ip_ev"]
        source = "Local File (atoms.json)"
        print(f"\n[SOURCE: {source}] Found data for {name}.")

    # Priority B: Internet/Reference
    elif name in KNOWN_ATOMS:
        d = KNOWN_ATOMS[name].copy()
        source = "Internet/NIST Database"
        print(f"\n[SOURCE: {source}] Found reference data for {name}.")
        print("  (Simulated download from NIST repository...)")

    # Priority C: Not Found
    else:
        print(f"\n[SOURCE: None] Atom '{name}' not found in any database.")
        print("  You will need to enter parameters manually.")
        source = "Manual"


    # --- CONFIRMATION / FALLBACK ---
    if "Z" in d and "E" in d:
        print(f"  Z={d['Z']}, Zc={d['Zc']}, State={d['n']}{'spdf'[d['l']] if d['l']<4 else 'l='+str(d['l'])}")
        print(f"  Binding Energy E = {d['E']} eV")
        
        confirm = input("Use these values? [Y(es)/n(o)/c(ustom)]: ").strip().lower()
        
        if confirm == 'n':
            # User wants to check next source? Or just manual?
            # If we were Local, maybe they want Internet version?
            # For simplicity, if they reject, we go to Manual.
            print("  Switching to Manual Input.")
            d = {} 
        elif confirm == 'c':
             print("  Switching to Manual Input.")
             d = {}
        else:
            # Accepted
            pass
            
    # --- MANUAL INPUT FALLBACK ---
    if 'Z' not in d:
        try:
            def get_in(prompt, default):
                p = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
                val = input(p).strip()
                if not val and default is not None:
                    return default
                return val

            Z = float(get_in("Nuclear Charge Z", KNOWN_ATOMS.get(name, {}).get("Z")))
            Zc = float(get_in("Core Charge Zc", KNOWN_ATOMS.get(name, {}).get("Zc")))
            n = int(get_in("Target State n", KNOWN_ATOMS.get(name, {}).get("n")))
            l = int(get_in("Target State l", KNOWN_ATOMS.get(name, {}).get("l")))
            E_str = get_in("Experimental Binding Energy in eV", KNOWN_ATOMS.get(name, {}).get("E"))
            E_exp_eV = float(E_str)
            
        except ValueError:
            print("Invalid number format.")
            return
            
    else:
        # Unpack confirmed 'd'
        try:
            Z = float(d["Z"])
            Zc = float(d["Zc"])
            n = int(d["n"])
            l = int(d["l"])
            E_exp_eV = float(d["E"])
        except ValueError:
            print("Error in stored data structure.")
            return

    # Grid setup
    print("\nSetting up physics grid...")
    grid = make_r_grid(r_min=1e-4, r_max=200.0, n_points=2000)
    
    target_E_au = ev_to_au(E_exp_eV)
    
    # --- OPTIMIZATION / PLOTTING LOGIC ---
    run_optimization = True
    best_params = None

    if source == "Local File (atoms.json)":
        # Check if we have params
        if "a_params" in entry:
            print(f"  Atom already has fitted parameters: {entry['a_params']}")
            choice = input("  (P)lot/View existing potential or (R)efit? [P/r]: ").strip().lower()
            if choice not in ['r', 'refit']:
                print("  Skipping optimization. Using existing parameters.")
                run_optimization = False
                best_params = entry["a_params"]
                # Need to update E_exp_eV from entry if not already set
                if "ip_ev" in entry: E_exp_eV = float(entry["ip_ev"]) # IP is positive eV
    
    if run_optimization:
        print(f"Target E = {target_E_au:.5f} a.u. ({E_exp_eV:.3f} eV)")
        print("Starting optimization (Nelder-Mead)... please wait.")
        
        # Generic Guess
        # a1=2.0 (screen 1s), a2=Z (decay fast)
        x0 = [Z-1.0, 1.0, 0.0, 1.0, 0.0, 1.0] 
        
        # --- KNOWN PRESETS (Better Initial Guesses) ---
        # Heuristics for common alkali-like systems
        if name in ["Li", "Na", "K", "Rb", "Cs"]:
            # Strong core screening
            x0 = [Z-Zc, 0.8*Z, 0.0, 1.0, 0.0, 1.0]

        t0 = time.time()
        
        # Multi-start optimization for robustness
        best_result = None
        best_cost = float('inf')
        
        # Try multiple initial guesses
        initial_guesses = [
            x0,  # Default guess
            [Z-Zc, Z*0.5, 0.0, 1.0, 0.0, 1.0],  # Alternative 1
            [Z-Zc, Z, 0.001, 0.5, 0.001, 2.0],  # Alternative 2
        ]
        
        for i, guess in enumerate(initial_guesses):
            print(f"  Trying initial guess {i+1}/{len(initial_guesses)}...")
            res = scipy.optimize.minimize(
                cost_function, 
                guess, 
                args=(Zc, n, l, target_E_au, grid, Z),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res
        
        res = best_result
        dt = time.time() - t0
        
        print(f"\nOptimization Finished in {dt:.2f}s.")
        print(f"Success: {res.success}")
        print(f"Final Cost (Error in a.u.): {res.fun:.2e}")
        
        best_params = res.x
        
        # Auto-save only if we optimized
        print("\nSaving results...")
        save_to_json(name, Z, Zc, n, l, abs(E_exp_eV), best_params)

    # --- FINAL CALCULATION ---
    E_final_au = solve_binding_energy(best_params, Zc, n, l, grid)
    E_final_eV = au_to_ev(E_final_au)
    
    # User might track IP (positive) or Binding (negative)
    # We display comparing magnitudes usually
    print(f"\nCalculated Binding E = {E_final_eV:.4f} eV")
    print(f"Reference Binding E  = {E_exp_eV if E_exp_eV < 0 else -E_exp_eV:.4f} eV")
    print(f"Diff                 = {abs(abs(E_final_eV) - abs(E_exp_eV)):.4f} eV")
    
    # --- AUTOMATIC DIAGNOSTIC VERIFICATION ---
    print("\n" + "="*50)
    print("AUTOMATIC VERIFICATION")
    print("="*50)
    success, calc_E, exp_E, diff = verify_fit(name, Z, Zc, n, l, abs(E_exp_eV), best_params)
    
    if success:
        print(f"[SUCCESS] Fitted parameters verified!")
        print(f"  Calculated: {calc_E:.4f} eV")
        print(f"  Expected:   {exp_E:.4f} eV")
        print(f"  Difference: {diff:.4f} eV (< 0.1 eV threshold)")
    else:
        if diff is not None:
            print(f"[WARNING] Fitting not fully successful!")
            print(f"  Calculated: {calc_E:.4f} eV")
            print(f"  Expected:   {exp_E:.4f} eV")
            print(f"  Difference: {diff:.4f} eV (> 0.1 eV threshold)")
            print("\n  Consider:")
            print("  1. Running with more iterations (modify maxiter)")
            print("  2. Trying different initial guesses")
            print("  3. Checking if target state (n,l) is correct")
        else:
            print(f"[ERROR] Could not find target bound state!")
            print(f"  Expected energy: {exp_E:.4f} eV")
    print("="*50)

    
    # --- PHYSICALITY CHECK & PLOTTING ---
    try:
        import matplotlib.pyplot as plt
        
        # Calculate final potential and Z_eff
        V = V_core_on_grid(grid, CorePotentialParams(Zc, *best_params))
        r = grid.r
        Z_eff = -r * V
        
        # Check Z_eff limits
        z_eff_0 = Z_eff[0] # Should be close to Z
        z_eff_inf = Z_eff[-1] # Should be close to Zc
        
        print("\n--- Physicality Check ---")
        print(f"Z_eff(r->0) = {z_eff_0:.2f} (Target: {Z})")
        print(f"Z_eff(r->inf) = {z_eff_inf:.2f} (Target: {Zc})")
        
        if abs(z_eff_0 - Z) > 0.5:
             print("[WARNING] Z_eff at origin deviates significantly from Nuclear Charge Z!")
             
        # Plot
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Potential V(r)
        plt.subplot(1, 2, 1)
        # Compare with pure Coulomb (-Z/r) and asymptotic (-Zc/r)
        plt.plot(r, V, 'b-', label='Effective V(r)')
        plt.plot(r, -Zc/r, 'g--', label=f'Asymptotic -{Zc}/r')
        plt.xlim(0, 5)
        plt.ylim(-Z*2, 0) # Focus near nucleus
        plt.title(f"Potential for {name}")
        plt.xlabel("r (a.u.)")
        plt.ylabel("V (a.u.)")
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Effective Charge Z_eff(r)
        plt.subplot(1, 2, 2)
        plt.plot(r, Z_eff, 'r-', label='Z_eff(r)')
        plt.axhline(Z, color='k', linestyle=':', label='Nuclear Z')
        plt.axhline(Zc, color='k', linestyle='--', label='Core Zc')
        plt.xlim(0, 10)
        plt.ylim(0, Z+1)
        plt.title("Effective Charge Profile")
        plt.xlabel("r (a.u.)")
        plt.ylabel("Z_eff")
        plt.legend()
        plt.grid(True)
        
        plot_file = f"fit_{name}.png"
        plt.savefig(plot_file)
        print(f"[INFO] Fit visualization saved to {plot_file}")
        
    except ImportError:
        print("[INFO] Matplotlib not found, skipping plot.")
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")

if __name__ == "__main__":
    main()
