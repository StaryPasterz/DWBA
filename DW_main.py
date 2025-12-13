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
from calibration import TongModel


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

        return get_input_int(prompt, default)

def select_target():
    print("\n--- Target Selection ---")
    print("1. Hydrogen (H)    [Z=1]")
    print("2. Helium Ion (He+) [Z=2]")
    print("3. Custom Nucleus")
    
    c = input("Choice [1]: ").strip()
    if c == '2': 
        return 2.0
    elif c == '3':
        return get_input_float("Enter Nuclear Charge Z")
    else:
        return 1.0

def generate_flexible_energy_grid(start_eV: float, end_eV: float, density_factor: float = 1.0) -> np.ndarray:
    """
    Generates a grid that is dense at low energies (start) and sparse at high energies (end).
    Logic: Uses geometric progression for 'excess energy' (E - start).
    """
    epsilon_min = 0.05 # 50 meV above threshold
    if end_eV <= start_eV + epsilon_min:
        return np.array([start_eV + epsilon_min])
        
    epsilon_max = end_eV - start_eV
    
    # Estimate decent number of points
    # Log range is np.log10(epsilon_max / epsilon_min)
    # E.g. 1000/0.05 = 20000 -> 4.3 decades.
    # Base density = 15 points/decade.
    
    decades = np.log10(epsilon_max / epsilon_min)
    n_points = int(decades * 15.0 * density_factor)
    n_points = max(n_points, 10) # Minimum safety
    
    # Generate excess
    excess = np.geomspace(epsilon_min, epsilon_max, n_points)
    
    energies = start_eV + excess
    
    # Ensure they are unique and sorted (geomspace should be, but safety first)
    return np.unique(np.round(energies, 3))


def get_energy_list_interactive():
    print("\n--- Energy Selection ---")
    print("1. Single Energy")
    print("2. Linear Grid (Start, End, Step)")
    print("3. Custom List (comma separated)")
    print("4. Flexible/Log Grid (Dense at low, Sparse at high)")
    
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

    elif choice == '4':
        start = get_input_float("  Threshold/Start [eV]", 10.0)
        end   = get_input_float("  End [eV]", 1000.0)
        dens  = get_input_float("  Density Factor (1.0=Normal, 2.0=Dense)", 1.0)
        return generate_flexible_energy_grid(start, end, dens)
    
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
    Z = select_target()
    print(f"Selected Z = {Z}")
      
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("Final State:")
    nf = get_input_int("  n", 2)
    lf = get_input_int("  l", 0)
    
    print("\n--- Model Selection ---")
    print("1. Static Only (Standard DWBA) - Matches Article (Default)")
    print("2. Static + Exchange (DWSE) - Improved Physics")
    print("3. Static + Exchange + Polarization (SEP) - Advanced")
    choice = input("Select Model [1-3] (default=1): ").strip()
    
    use_ex = False # Default 1 is Static Only
    use_pol = False
    
    if choice == '2':
        use_ex = True
    elif choice == '3':
        use_ex = True
        use_pol = True
    elif choice == '1':
        pass # Default
    else:
        # Default -> 1
        pass
    
    # Sub-selection for Exchange
    ex_method = 'fumc'
    if use_ex:
        print("\n  [Exchange Potential Options]")
        print("  1. Furness-McCarthy (Article Standard, Default)")
        print("  2. Slater (Free Electron Gas Approx)")
        ex_choice = input("  Select Exchange Method [1-2] (default=1): ").strip()
        if ex_choice == '2':
            ex_method = 'slater'
    
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
    
    # --- Calibration Pilot Run ---
    print("\n[Calibration] Running Pilot Calculation at 1000 eV for Alpha Matching...")
    pilot_E = 1000.0
    
    # Determine n-p vs n-s (approximate check)
    # Generalized: if final state is S (l=0), use s-s params. Else use s-p params.
    # Note: Article only explicitly covers starting from 1s. We assume this generalizes.
    t_type = "s-p"
    if lf == 0:
        t_type = "s-s"
    
    # We need threshold energy to init model.
    # We can get it from a quick bound state check or run one calc.
    # compute_total_excitation_cs runs bound state solve internally each time (inefficient but safe).
    # We will run it once.
    
    try:
        # Article uses large box (200 au)
        res_pilot = compute_total_excitation_cs(pilot_E, spec, core_params, r_max=200.0, n_points=3000, use_exchange_potential=use_ex, use_polarization_potential=use_pol, exchange_method=ex_method)
    except Exception as e:
        print(f"[Calibration] Pilot failed: {e}. Defaulting Alpha=1.0")
        res_pilot = None

    # Init model
    # If pilot failed, we guess threshold (approx for H-like).
    dE_thr = res_pilot.E_excitation_eV if res_pilot else 10.2 * Z**2 
    epsilon_exc_au = -0.5 * (Z**2) / (nf**2) # Approximate if pilot failed, but usually correct from res
    
    # If res_pilot worked, use exact epsilon? 
    # res doesn't allow easy access to epsilon_exc_au directly (it returns E_exc_eV).
    # E_incident = E_final + dE. dE = E_i - E_f.
    # We trust the model init.
    
    tong_model = TongModel(dE_thr, epsilon_exc_au, transition_type=t_type)
    
    if res_pilot and res_pilot.ok_open_channel:
        alpha = tong_model.calibrate_alpha(pilot_E, res_pilot.sigma_total_cm2)
        print(f"[Calibration] Alpha determined: {alpha:.4f} (Matched to DWBA at 1000 eV)")
    else:
        print("[Calibration] Pilot could not determine Alpha. Using default=1.0")

    
    print(f"\nStarting calculation for {key} ({len(energies)} points)...")
    
    try:
        for E in energies:
            if E <= 0.01: continue
            print(f"E={E:.2f} eV...", end=" ", flush=True)
            try:
                # Article uses large box (200 au)
                res = compute_total_excitation_cs(E, spec, core_params, r_max=200.0, n_points=3000, use_exchange_potential=use_ex, use_polarization_potential=use_pol, exchange_method=ex_method)
                
                if res.ok_open_channel:
                    # Calculate Calibration
                    sigma_raw = res.sigma_total_cm2
                    sigma_cal = tong_model.calculate_sigma_cm2(E)
                    factor_C = tong_model.get_calibration_factor(E, sigma_raw)
                    
                    print(f"OK. σ_dwba={sigma_raw:.2e} | σ_cal={sigma_cal:.2e} | C={factor_C:.3f}")
                    
                    # Print Dominant Partial Waves (Top 3)
                    if res.partial_waves:
                        sorted_pw = sorted(res.partial_waves.items(), key=lambda x: x[1], reverse=True)[:3]
                        pw_str = ", ".join([f"{k}:{v:.1e}" for k, v in sorted_pw])
                        print(f"    Dominant L: {pw_str}")

                    results.append({
                        "energy_eV": E,
                        "sigma_au": res.sigma_total_au,
                        "sigma_cm2": sigma_raw,
                        "sigma_mtong_cm2": sigma_cal,
                        "calibration_factor_C": factor_C,
                        "Threshold_eV": res.E_excitation_eV,
                        "partial_waves": res.partial_waves
                    })
                else:
                    print(f"Closed (Thr={res.E_excitation_eV:.2f})")
                    results.append({
                        "energy_eV": E,
                        "sigma_au": 0.0,
                        "sigma_cm2": 0.0,
                        "sigma_mtong_cm2": 0.0,
                        "calibration_factor_C": 1.0, 
                        "Threshold_eV": res.E_excitation_eV
                    })
                
                # Incremental Save
                save_results(filename, {key: results})

            except Exception as e:
                print(f"Error ({E} eV): {e}")

    except KeyboardInterrupt:
        print("\n\n[STOP] Calculation interrupted by user (Ctrl+C).")
        print(f"[INFO] Saving {len(results)} data points collected so far...")
        save_results(filename, {key: results})
        print("[INFO] Partial results saved successfully.")
        return

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
    Z = select_target()
    print(f"Selected Z = {Z}")
    
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("\n--- Model Selection ---")
    print("1. Static Only (Standard DWBA) - Matches Article (Default)")
    print("2. Static + Exchange (DWSE) - Improved Incident Physics")
    print("3. Static + Exchange + Polarization (SEP) - Advanced")
    choice = input("Select Model [1-3] (default=1): ").strip()

    use_ex = False 
    use_pol = False
    
    if choice == '2':
        use_ex = True
    elif choice == '3':
        use_ex = True
        use_pol = True
    elif choice == '1':
        pass 
    else:
        pass
    
    # Sub-selection for Exchange
    ex_method = 'fumc'
    if use_ex:
        print("\n  [Exchange Potential Options]")
        print("  1. Furness-McCarthy (Article Standard, Default)")
        print("  2. Slater (Free Electron Gas Approx)")
        ex_choice = input("  Select Exchange Method [1-2] (default=1): ").strip()
        if ex_choice == '2':
            ex_method = 'slater'

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

    try:
        for E in energies:
            print(f"E={E:.2f} eV...", end=" ", flush=True)
            try:
                res = compute_ionization_cs(
                    E, spec, core_params, 
                    r_max=200.0, n_points=3000, 
                    use_exchange=use_ex, 
                    use_polarization=use_pol,
                    exchange_method=ex_method
                )
                
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
                
                # Incremental Save
                save_results(filename, {key: results})

            except Exception as e:
                print(f"Error ({E} eV): {e}")

    except KeyboardInterrupt:
        print("\n\n[STOP] Calculation interrupted by user (Ctrl+C).")
        print(f"[INFO] Saving {len(results)} data points collected so far...")
        save_results(filename, {key: results})
        print("[INFO] Partial results saved successfully.")
        return

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
