"""
DW_main.py

Unified Interface for Distorted Wave Born Approximation (DWBA) Calculations and Analysis.

Modes:
1. Excitation Scan (Single Point or Grid)
2. Ionization Scan (Single Point or Grid)
3. Generate Plots (Visualize previously saved results)
4. Quit

Results are saved to 'results_[RUN_NAME]_[TYPE].json'.
"""

import numpy as np
import json
import os
import sys
import glob
from dataclasses import asdict

from driver import (
    compute_total_excitation_cs,
    ExcitationChannelSpec,
    ev_to_au
)
from ionization import (
    compute_ionization_cs,
    IonizationChannelSpec
)
from potential_core import CorePotentialParams
import plotter # Import plotter module

# --- Input Helpers ---

def get_input_float(prompt, default=None):
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    try:
        return float(val)
    except ValueError:
        print("Invalid number.")
        return get_input_float(prompt, default)

def get_input_int(prompt, default=None):
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    try:
        return int(val)
    except ValueError:
        print("Invalid integer.")
        return get_input_int(prompt, default)

def get_energy_list_interactive():
    print("\n--- Energy Selection ---")
    print("1. Single Energy")
    print("2. Linear Grid (Start, End, Step)")
    print("3. Custom List (comma separated)")
    
    choice = input("Select mode [2]: ").strip()
    if not choice: choice = '2'
    
    if choice == '1':
        E = get_input_float("Energy [eV]", 50.0)
        return np.array([E])
        
    elif choice == '2':
        start = get_input_float("  Start [eV]", 10.0)
        end   = get_input_float("  End   [eV]", 200.0)
        step  = get_input_float("  Step  [eV]", 5.0)
        if step <= 0: step = 1.0
        return np.arange(start, end + 0.0001, step)
        
    elif choice == '3':
        raw = input("Enter energies (e.g. 10.0, 15.5, 20): ")
        try:
            vals = [float(x.strip()) for x in raw.split(',')]
            return np.array(vals)
        except ValueError:
            print("Invalid format. Defaulting to 50.0 eV.")
            return np.array([50.0])
    
    print("Invalid choice. Defaulting to Grid 10-200.")
    return np.arange(10.0, 201.0, 5.0)

# --- Data Management ---

def load_results(filename):
    """
    Load existing results from a JSON file.
    Returns an empty dict if file does not exist or is corrupt.
    """
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_results(filename, new_data_dict):
    """
    Update and save results to JSON.
    Merges new_data_dict into existing data (by key) to prevent data loss.
    """
    current = load_results(filename)
    current.update(new_data_dict)
    with open(filename, "w") as f:
        json.dump(current, f, indent=2)
    print(f"\n[INFO] Results saved to {filename}")

def check_file_exists_warning(filename):
    """
    Warn the user if the output file already exists.
    Returns True if user wants to continue (append), False to abort.
    """
    if os.path.exists(filename):
        print(f"\n[WARNING] File '{filename}' already exists!")
        print("New results will be appended/merged into this file.")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    return True

# --- Calculation Routines ---

def run_scan_excitation(run_name):
    """
    Interactive workflow for Excitation Cross Section calculation.
    
    Steps:
    1. Ask user for Nuclear Charge (Z) and quantum numbers (n, l) for Initial/Final states.
    2. Define the energy grid (Single point, Linear, or Custom list).
    3. Loop over energies, calling `compute_total_excitation_cs`.
    4. Save results to `results_{run_name}_exc.json`.
    """
    filename = f"results_{run_name}_exc.json"
    
    # Pre-check if we haven't already
    # But usually we just append. Let's just warn if it's the first write?
    # Actually, let's warn at the start of calculation if file exists.
    if not check_file_exists_warning(filename):
        print("Aborting calculation.")
        return

    print("\n=== EXCITATION CALCULATION ===")
    Z = get_input_float("Nuclear Charge Z", 1.0)
      
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("Final State:")
    nf = get_input_int("  n", 2)
    lf = get_input_int("  l", 0)
    
    energies = get_energy_list_interactive()
    
    # Convert quantum number n to sorting index n_index
    # For a given l, the states are n=l+1, l+2, ...
    # So index = n - l.
    n_idx_i = ni - li
    n_idx_f = nf - lf
    
    if n_idx_i < 1 or n_idx_f < 1:
        print("Error: Invalid n, l combination (n must be > l).")
        return

    spec = ExcitationChannelSpec(
        l_i=li, l_f=lf,
        n_index_i=n_idx_i, n_index_f=n_idx_f,
        N_equiv=1,
        L_max_integrals=15, 
        L_target_i=li,
        L_target_f=lf
    )
    # Default core params (Coulomb only), logic could be extended for Ne+ etc.
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

    key = f"Excitation_Z{Z}_n{ni}l{li}_to_n{nf}l{lf}"
    results = []
    
    print(f"\nStarting calculation for {key} ({len(energies)} points)...")
    
    for E in energies:
        if E <= 0.01: continue
        print(f"E={E:.2f} eV...", end=" ", flush=True)
        try:
            res = compute_total_excitation_cs(E, spec, core_params, r_max=100.0, n_points=3000)
            if res.ok_open_channel:
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2 | C(E)={res.calibration_factor_C:.3f}")
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": res.sigma_mtong_cm2,
                    "calibration_factor_C": res.calibration_factor_C,
                    "Threshold_eV": res.E_excitation_eV
                })
            else:
                print(f"Closed (Thr={res.E_excitation_eV:.2f})")
                results.append({
                    "energy_eV": E,
                    "sigma_au": 0.0,
                    "sigma_cm2": 0.0,
                    "sigma_mtong_cm2": 0.0,
                    "Threshold_eV": res.E_excitation_eV
                })
        except Exception as e:
            print(f"Error: {e}")

    save_results(filename, {key: results})
    print("Calculation complete.")


def run_scan_ionization(run_name):
    """
    Interactive workflow for Ionization Cross Section calculation.
    
    Steps:
    1. Ask user for Target parameters.
    2. Define energy grid.
    3. Loop over incident energies.
    4. Provide M-Tong scaling (BE-scaling) output for comparison.
    5. Save results to `results_{run_name}_ion.json`.
    """
    filename = f"results_{run_name}_ion.json"
    
    if not check_file_exists_warning(filename):
        print("Aborting calculation.")
        return

    print("\n=== IONIZATION CALCULATION ===")
    Z = get_input_float("Nuclear Charge Z", 1.0)
    
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    energies = get_energy_list_interactive()
    
    n_idx_i = ni - li
    if n_idx_i < 1:
        print("Error: Invalid n, l combination (n must be > l).")
        return
        
    spec = IonizationChannelSpec(
        l_i=li,
        n_index_i=n_idx_i,
        N_equiv=1,
        l_eject_max=3,
        L_max=8,
        L_i_total=li
    )
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

    key = f"Ionization_Z{Z}_n{ni}l{li}"
    results = []

    print(f"\nStarting calculation for {key} ({len(energies)} points)...")

    for E in energies:
        print(f"E={E:.2f} eV...", end=" ", flush=True)
        try:
            res = compute_ionization_cs(E, spec, core_params, r_max=100.0, n_points=3000)
            
            if res.sigma_total_cm2 > 0:
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2")
                ip = res.IP_eV
                scale = E / (E + ip) if (E + ip) > 0 else 0.0
                mt = res.sigma_total_cm2 * scale
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": mt,
                    "calibration_factor_C": 1.0,
                    "Threshold_eV": ip,
                    "IP_eV": ip
                })
            else:
                 print("Closed/Zero")
        except Exception as e:
            print(f"Error: {e}")

    save_results(filename, {key: results})
    print("Calculation complete.")

# --- Visualization ---

def run_visualization():
    print("\n=== PLOT GENERATION ===")
    
    # List JSON files
    files = glob.glob("*.json")
    if not files:
        print("No .json result files found in current directory.")
        return
        
    print("Available Result Files:")
    for idx, f in enumerate(files):
        print(f"{idx+1}. {f}")
        
    print("0. Cancel")
    
    try:
        choice_idx = int(input("Select file number: ")) - 1
    except ValueError:
        return
        
    if choice_idx < 0 or choice_idx >= len(files):
        return
        
    selected_file = files[choice_idx]
    
    print("\nSelect Style:")
    print("1. Standard (eV / cm^2)")
    print("2. Atomic (Ha / a0^2)")
    print("3. Article (E/Thr / pi*a0^2)")
    print("4. Mixed (eV / a.u.)")
    
    choice = input("Choice [1]: ").strip()
    
    # Map to plotter style args
    style_map = {
        '1': 'std',
        '2': 'atomic',
        '3': 'article',
        '4': 'ev_au'
    }
    style = style_map.get(choice, 'std')
    
    print(f"Generating plot from '{selected_file}' with style '{style}'...")
    
    # Pass arguments to plotter
    old_argv = sys.argv
    sys.argv = ["plotter.py", style, selected_file]
    try:
        plotter.main()
    except SystemExit:
        pass 
    except Exception as e:
        print(f"Plotter Error: {e}")
    finally:
        sys.argv = old_argv

# --- Main Loop ---

def main():
    print("\n===========================================")
    print("   DWBA CALCULATION SUITE - UNIFIED V2")
    print("===========================================")
    
    run_name = input("Enter Simulation Name (e.g. 'run1'): ").strip()
    if not run_name:
        run_name = "default"
        print("Using name: 'default'")
        
    print(f"Active Run Name: {run_name}")
    print(f"Outputs will be: results_{run_name}_exc.json / results_{run_name}_ion.json")

    while True:
        print("\n--- Main Menu ---")
        print(f"Current Run: {run_name}")
        print("1. Calculate Excitation Cross Sections")
        print("2. Calculate Ionization Cross Sections")
        print("3. Generate Plots (Load any file)")
        print("4. Change Run Name")
        print("q. Quit")
        
        choice = input("\nSelect Mode: ").strip().lower()
        
        if choice == '1':
            run_scan_excitation(run_name)
        elif choice == '2':
            run_scan_ionization(run_name)
        elif choice == '3':
            run_visualization()
        elif choice == '4':
            new_name = input("Enter new Simulation Name: ").strip()
            if new_name:
                run_name = new_name
                print(f"Run Name changed to: {run_name}")
        elif choice == 'q':
            print("Goodbye.")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
