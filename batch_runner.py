# batch_runner.py
#
# Flexible DWBA Scan Runner (v2).
# Supports Single/Grid/Custom Energy inputs.
#

import numpy as np
import json
import os
import sys
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

def load_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
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
    
    print("Initial State:")
    ni = get_input_int("  n", 1)
    li = get_input_int("  l", 0)
    
    print("Final State:")
    nf = get_input_int("  n", 2)
    lf = get_input_int("  l", 0)
    
    energies = get_energy_list_interactive()
    
    spec = ExcitationChannelSpec(
        l_i=li, l_f=lf,
        n_index_i=ni, n_index_f=nf,
        N_equiv=1,
        L_max=10, 
        L_i_total=li
    )
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)

    key = f"Excitation_Z{Z}_n{ni}l{li}_to_n{nf}l{lf}"
    results = []
    
    print(f"Running scan for {key} ({len(energies)} points)...")
    
    for E in energies:
        if E <= 0.01: continue
        print(f"E={E:.2f} eV...", end=" ", flush=True)
        try:
            res = compute_total_excitation_cs(E, spec, core_params, r_max=100.0, n_points=3000)
            if res.ok_open_channel:
                print(f"OK. σ={res.sigma_total_cm2:.2e} cm2 ({res.sigma_total_au:.2e} au)")
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
    
    energies = get_energy_list_interactive()
    
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
                    "Threshold_eV": ip,
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
        print("1. Custom Excitation Scan (Interactive Energy)")
        print("2. Custom Ionization Scan (Interactive Energy)")
        print("3. Quit")
        
        choice = input("Select [1]: ").strip()
        if not choice: choice = '1'
        
        if choice == '1':
            run_scan_excitation()
        elif choice == '2':
            run_scan_ionization()
        elif choice == '3' or choice.lower() == 'q':
            break
        else:
            print("Invalid.")

if __name__ == "__main__":
    main()
