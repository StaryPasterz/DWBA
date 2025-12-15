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
import atom_library
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

def select_target():
    print("\n--- Target Selection ---")
    atoms = atom_library.get_atom_list()
    
    # Display available atoms
    for i, name in enumerate(atoms):
        entry = atom_library.get_atom(name)
        print(f"{i+1}. {entry.name:<5} - {entry.description}")
    
    print(f"{len(atoms)+1}. Custom (Manual Z)")
    
    raw = input(f"Select Target [1-{len(atoms)+1}] (default=1): ").strip()
    if not raw: raw = "1"
    
    try:
        idx = int(raw) - 1
    except ValueError:
        idx = 0 # default
        
    if 0 <= idx < len(atoms):
        name = atoms[idx]
        return atom_library.get_atom(name)
    else:
        # manual custom
        z_in = get_input_float("Enter Nuclear Charge Z")
        # For custom, we assume simple Coulomb for now (Zc=Z)
        return atom_library.AtomEntry(
            name=f"Z={z_in}",
            Z=z_in,
            core_params=CorePotentialParams(Zc=z_in, a1=0, a2=1, a3=0, a4=1, a5=0, a6=1),
            default_n=1,
            default_l=0,
            ip_ev=13.6*z_in**2, # approx
            description="Custom Hydrogen-like"
        )

def generate_flexible_energy_grid(start_eV: float, end_eV: float, density_factor: float = 1.0) -> np.ndarray:
    """
    Generates a grid that is dense at low energies (start) and sparse at high energies (end).
    Logic: Uses geometric progression for 'excess energy' (E - start).
    """
    epsilon_min = 0.5 # 500 meV above threshold (Relaxed from 0.05)
    if end_eV <= start_eV + epsilon_min:
        return np.array([start_eV + epsilon_min])
        
    epsilon_max = end_eV - start_eV
    
    # Estimate decent number of points
    # Log range is np.log10(epsilon_max / epsilon_min)
    # E.g. 1000/0.5 = 2000 -> 3.3 decades.
    # Base density = 10 points/decade (Relaxed from 15).
    
    decades = np.log10(epsilon_max / epsilon_min)
    n_points = int(decades * 10.0 * density_factor)
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
    """
    filename = f"results_{run_name}_exc.json"
    
    if not check_file_exists_warning(filename):
        print("Aborting calculation.")
        return

    print("\n=== EXCITATION CALCULATION ===")
    atom_entry = select_target()
    Z = atom_entry.Z
    print(f"Selected Target: {atom_entry.name} (Z={Z})")
      
    print("Initial State:")
    ni = get_input_int("  n", atom_entry.default_n)
    li = get_input_int("  l", atom_entry.default_l)
    
    print("Final State:")
    nf = get_input_int("  n", ni + 1)
    lf = get_input_int("  l", li + 1 if li==0 else li-1)
    
    # --- VALIDATION ---
    if li >= ni:
        print(f"\n[ERROR] Initial state impossible: l({li}) >= n({ni}).")
        return
    if lf >= nf:
        print(f"\n[ERROR] Final state impossible: l({lf}) >= n({nf}).")
        return
    if ni == nf and li == lf:
        print(f"\n[ERROR] Initial and Final states are identical (Elastic Scattering).")
        print("This module is for Excitation (inelastic). Aborting.")
        return
    if nf < ni:
        print(f"\n[WARNING] n_final ({nf}) < n_initial ({ni}). This is De-excitation.")
        confirm = input("Are you sure? [y/N]: ").strip().lower()
        if confirm != 'y': return
    if abs(lf - li) > 3:
        print(f"\n[WARNING] Large angular momentum change (Delta L = {abs(lf-li)}).")
        print("Cross sections might be extremely small.")
        # Just a warning, proceed.
    
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
        N_equiv=1, # SAE approximation
        L_max_integrals=15, 
        L_target_i=li,
        L_target_f=lf
    )
    
    # Use parameters from library
    core_params = atom_entry.core_params

    key = f"Excitation_Z{Z}_{atom_entry.name}_n{ni}l{li}_to_n{nf}l{lf}"
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
        res_pilot = compute_total_excitation_cs(
            pilot_E, spec, core_params, 
            r_max=200.0, n_points=3000, 
            use_polarization_potential=use_pol,
            exchange_method=ex_method
        )
    except Exception as e:
        print(f"[Calibration] Pilot failed: {e}. Defaulting Alpha=1.0")
        res_pilot = None

    # Init model
    # If pilot failed, we guess threshold.
    dE_thr = res_pilot.E_excitation_eV if res_pilot else atom_entry.ip_ev # rough guess
    
    # Calculate final state binding energy (epsilon in Eq 493)
    # E_final = E_initial + dE_thr
    # E_initial approx -IP
    # So epsilon = (-IP + dE_thr) in a.u.
    if res_pilot:
        # res_pilot doesn't store E_final_bound directly, but we can deduce
        # We need epsilon_exc_au.
        # Let's approximate from IP.
        E_init_eV = -atom_entry.ip_ev
        E_final_eV = E_init_eV + dE_thr
        # Check consistency: if dE_thr > IP, then E_final > 0 (Ionization). 
        # But we are in Excitation. Usually dE < IP.
        epsilon_exc_au = ev_to_au(E_final_eV)
    else:
        # Fallback for H-like
        epsilon_exc_au = -0.5 * (Z**2) / (nf**2) 

    tong_model = TongModel(dE_thr, epsilon_exc_au, transition_type=t_type)
    alpha = 1.0
    
    if res_pilot and res_pilot.sigma_total_cm2 > 0:
        sigma_calc = res_pilot.sigma_total_cm2
        alpha = tong_model.calibrate_alpha(pilot_E, sigma_calc)
        print(f"[Calibration] Pilot Sigma={sigma_calc:.2e} cm2")
        print(f"[Calibration] Tong Model Ref={tong_model.calculate_sigma_cm2(pilot_E):.2e} cm2")
        print(f"[Calibration] ALPHA = {alpha:.4f}")
    else:
        print("[Calibration] Pilot result invalid. Using Alpha=1.0")

    # --- Smart Grid Adjustment ---
    # User request: If grid starts below threshold, automatically adjust to start from threshold.
    print(f"[Smart Grid Debug] Threshold dE_thr = {dE_thr:.3f} eV")
    if dE_thr > 0:
        valid_indices = energies > dE_thr
        # Check if we need to filter
        if not np.all(valid_indices):
            print(f"\n[Smart Grid] Some energies in grid (min={np.min(energies):.2f}) are below/at threshold ({dE_thr:.2f} eV). Correcting...")
            energies = energies[valid_indices]
            
            # Ensure we have a starting point near threshold
            start_epsilon = 0.5 
            new_start = dE_thr + start_epsilon
            
            if len(energies) == 0 or energies[0] > new_start + 0.1:
                energies = np.insert(energies, 0, new_start)
                
            energies = np.unique(np.round(energies, 3))
            print(f"[Smart Grid] New start: {energies[0]:.2f} eV. Grid size: {len(energies)}")
            
    # --- Main Loop ---
    
    # Pre-calculate static target properties (Optimization)
    print("\n[Optimization] Pre-calculating static target properties...")
    from driver import prepare_target, compute_excitation_cs_precalc
    
    prep = prepare_target(
        chan=spec,
        core_params=core_params,
        use_exchange=use_ex,
        use_polarization=use_pol,
        exchange_method=ex_method,
        r_max=200.0,
        n_points=3000
    )
    print("[Optimization] Ready.")
    
    print(f"\nStarting calculation for {key} ({len(energies)} points)...")
    
    try:
        for E in energies:
            if E <= 0.01: continue
            print(f"E={E:.2f} eV...", end=" ", flush=True)
            try:
                # Optimized call
                res = compute_excitation_cs_precalc(E, prep)
                
                # Print dominant partial waves
                top_pws = []
                if res.partial_waves:
                    # Sort by contribution
                    sorted_pws = sorted(res.partial_waves.items(), key=lambda x: x[1], reverse=True)
                    # Take top 3
                    top_pws = [f"{k}={v:.1e}" for k,v in sorted_pws[:3]]
                
                pw_info = f" [Top: {', '.join(top_pws)}]" if top_pws else ""
                
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2{pw_info}")
                
                # Robust Calibration Handling
                try:
                    tong_sigma = tong_model.calculate_sigma_cm2(E)
                    scaled_sigma = res.sigma_total_cm2 * alpha
                except Exception as cal_err:
                    # If calibration model fails, default to unscaled
                    print(f"  [Warn] Calibration calc failed: {cal_err}")
                    tong_sigma = 0.0
                    scaled_sigma = res.sigma_total_cm2
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": tong_sigma,
                    "sigma_scaled_cm2": scaled_sigma,
                    "calibration_alpha": alpha,
                    "partial_waves": res.partial_waves
                })
                
            except Exception as e:
                print(f"Failed: {e}")

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
    """
    filename = f"results_{run_name}_ion.json"
    
    if not check_file_exists_warning(filename):
        print("Aborting calculation.")
        return

    print("\n=== IONIZATION CALCULATION ===")
    atom_entry = select_target()
    Z = atom_entry.Z
    print(f"Selected Target: {atom_entry.name} (Z={Z})")
    
    print("Initial State:")
    ni = get_input_int("  n", atom_entry.default_n)
    li = get_input_int("  l", atom_entry.default_l)
    
    if li >= ni:
        print(f"\n[ERROR] Initial state impossible: l({li}) >= n({ni}).")
        return
    
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

    # --- Smart Grid Adjustment (Ionization) ---
    ion_thr = atom_entry.ip_ev
    if np.any(energies <= ion_thr):
        print(f"\n[Smart Grid] Some energies are below/at IP ({ion_thr:.2f} eV). Correcting...")
        energies = energies[energies > ion_thr]
        
        start_epsilon = 0.5
        new_start = ion_thr + start_epsilon
        
        if len(energies) == 0 or energies[0] > new_start + 0.1:
             energies = np.insert(energies, 0, new_start)
             
        energies = np.unique(np.round(energies, 3))
        print(f"[Smart Grid] New start: {energies[0]:.2f} eV")
    
    n_idx_i = ni - li
    if n_idx_i < 1:
        print("Error: Invalid n, l combination (n must be > l).")
        return
        
    spec = IonizationChannelSpec(
        l_i=li,
        n_index_i=n_idx_i,
        N_equiv=1,
        l_eject_max=3,  # Sum up to F waves? 
        L_max=15, 
        L_i_total=li
    )
    
    # Use parameters from library
    core_params = atom_entry.core_params
    
    key = f"Ionization_Z{Z}_{atom_entry.name}_n{ni}l{li}"
    # --- Main Loop ---
    
    # Pre-calculate (Optimization)
    print("\n[Optimization] Pre-calculating static target properties...")
    from driver import prepare_target, ExcitationChannelSpec
    
    # Adapter: prepare_target expects ExcitationChannelSpec but Ionization needs IonizationChannelSpec.
    # We can fake the ExcitationChannelSpec just to get grid, V_core, and orb_i.
    # We set l_f to something valid (l_i + 1) just to pass the solver checks, though we won't use orb_f.
    
    tmp_chan = ExcitationChannelSpec(
        l_i=li, l_f=li+1, n_index_i=n_idx_i, n_index_f=n_idx_i+1, # dummy final
        N_equiv=1, L_max_integrals=10, L_target_i=li, L_target_f=li+1
    )
    
    prep = prepare_target(
        chan=tmp_chan,
        core_params=core_params,
        use_exchange=use_ex, 
        use_polarization=use_pol,
        exchange_method=ex_method,
        r_max=200.0,
        n_points=3000
    )
    print("[Optimization] Ready.")

    print(f"\nStarting calculation for {key} ({len(energies)} points)...")
    
    try:
        for E in energies:
            print(f"E={E:.2f} eV...", end=" ", flush=True)
            try:
                res = compute_ionization_cs(
                    E, spec, 
                    core_params=core_params, # Ignored for grid/V_core generation if _precalc used
                    r_max=200.0, n_points=3000,
                    use_exchange=use_ex, 
                    use_polarization=use_pol, 
                    exchange_method=ex_method
                )
                
                if res.sigma_total_cm2 > 0:
                    print(f"OK. σ={res.sigma_total_cm2:.2e} cm2")
                else:
                    print(f"Below Threshold (IP={res.IP_eV:.2f} eV)")
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "IP_eV": res.IP_eV,
                    "sdcs": res.sdcs_data,
                    "partial_waves": res.partial_waves
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed: {e}")

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
        print("3. Generate Plots (Cross Sections)")
        print("4. Partial Wave Analysis (Detailed Plots)")
        print("5. Fit Potential (New Atom Tool)")
        print("6. Change Run Name")
        print("q. Quit")
        
        choice = input("\nSelect Mode: ").strip().lower()
        
        if choice == '1':
            run_scan_excitation(run_name)
        elif choice == '2':
            run_scan_ionization(run_name)
        elif choice == '3':
            run_visualization()
        elif choice == '4':
            # Run Partial Wave Analysis
            try:
                import partial_wave_plotter
                print("\n=== PARTIAL WAVE ANALYSIS ===")
                # It expects a file argument normally, but we can wrap it or call main() if it allows.
                # Inspecting partial_wave_plotter.py suggests it has a main() that asks for file.
                # Let's try calling its main logic.
                partial_wave_plotter.main()
            except Exception as e:
                print(f"Error running plotter: {e}")
        elif choice == '5':
            # Run Potential Fitter
            try:
                import fit_potential
                fit_potential.main()
            except Exception as e:
                print(f"Error running fitter: {e}")
        elif choice == '6':
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
