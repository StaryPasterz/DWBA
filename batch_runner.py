# batch_runner.py
#
# Flexible DWBA Scan Runner.
# Allows user to select predefined sets or define custom scans interactively.
#

import numpy as np
import json
import os
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

RESULTS_FILE = "scan_results.json"

def get_input_float(prompt, default=None):
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    return float(val)

def get_input_int(prompt, default=None):
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    return int(val)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_results(new_data_dict):
    current = load_results()
    current.update(new_data_dict)
    with open(RESULTS_FILE, "w") as f:
        json.dump(current, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

def run_scan_excitation():
    print("\n--- Custom Excitation Scan ---")
    Z = get_input_float("Nuclear Charge Z", 1.0)
    
    # Init params
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("Final State:")
    nf = get_input_int("  n", 2)
    lf = get_input_int("  l", 0) # e.g. 2s
    
    # Energy grid
    print("Incident Energy Grid (eV):")
    e_start = get_input_float("  Start", 10.0)
    e_end   = get_input_float("  End", 200.0)
    e_step  = get_input_float("  Step", 5.0)
    
    energies = np.arange(e_start, e_end + 0.001, e_step)
    
    # Build Spec
    # Note: n_index is usually principal quantum number n for H-like.
    spec = ExcitationChannelSpec(
        l_i=li, l_f=lf,
        n_index_i=ni, n_index_f=nf,
        N_equiv=1,
        L_max=10, # default high quality
        L_i_total=li
    )
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

    key = f"Excitation_Z{Z}_n{ni}l{li}_to_n{nf}l{lf}"
    results = []
    
    print(f"Running scan for {key} ({len(energies)} points)...")
    
    for E in energies:
        if E <= 0.1: continue
        print(f"E={E:.2f} eV...", end=" ", flush=True)
        
        try:
            res = compute_total_excitation_cs(E, spec, core_params, r_max=100.0, n_points=3000)
            
            if res.ok_open_channel:
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2")
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": res.sigma_mtong_cm2,
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

    save_results({key: results})

def run_scan_ionization():
    print("\n--- Custom Ionization Scan ---")
    Z = get_input_float("Nuclear Charge Z", 1.0)
    
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("Incident Energy Grid (eV):")
    e_start = get_input_float("  Start", 15.0)
    e_end   = get_input_float("  End", 200.0)
    e_step  = get_input_float("  Step", 10.0)
    
    energies = np.arange(e_start, e_end + 0.001, e_step)
    
    spec = IonizationChannelSpec(
        l_i=li,
        n_index_i=ni,
        N_equiv=1,
        l_eject_max=3,
        L_max=8,
        L_i_total=li
    )
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

    key = f"Ionization_Z{Z}_n{ni}l{li}"
    results = []

    print(f"Running scan for {key} ({len(energies)} points)...")

    for E in energies:
        print(f"E={E:.2f} eV...", end=" ", flush=True)
        try:
            res = compute_ionization_cs(E, spec, core_params, r_max=100.0, n_points=3000)
            
            # Manual Check
            if res.sigma_total_cm2 > 0:
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2")
                # Calc MTong manually
                ip = res.IP_eV
                scale = E / (E + ip) if (E + ip) > 0 else 0.0
                mt = res.sigma_total_cm2 * scale
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": mt,
                    "Threshold_eV": ip, # For Ionization Threshold is IP
                    "IP_eV": ip
                })
            else:
                 print("Closed/Zero")
        except Exception as e:
            print(f"Error: {e}")

    save_results({key: results})

def main():
    while True:
        print("\n=== DWBA Batch Scanner ===")
        print("1. Custom Excitation Scan")
        print("2. Custom Ionization Scan")
        print("3. Run Standard Suite (H 1s->2p, H Ion, He+ Ion)")
        print("q. Quit")
        
        choice = input("Select: ").strip().lower()
        if choice == '1':
            run_scan_excitation()
        elif choice == '2':
            run_scan_ionization()
        elif choice == '3':
            # Implement standard suite call or call legacy functions
            print("Not implemented in this flexible version yet. Use custom modes.")
        elif choice == 'q':
            break
        else:
            print("Invalid.")

if __name__ == "__main__":
    main()
