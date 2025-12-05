# batch_runner.py
#
# Automatisches uruchamianie obliczeń DWBA dla serii energii.
# Generuje plik JSON z wynikami do późniejszej wizualizacji.
#

import numpy as np
import json
import time
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


def run_excitation_scan(target_name, Z, energy_list, channel_spec: ExcitationChannelSpec):
    print(f"\n--- Starting Excitation Scan for {target_name} (Z={Z}) ---")
    print(f"Channel: l_i={channel_spec.l_i} -> l_f={channel_spec.l_f}")
    
    # Parametry dlad czystego Coulomba (H-like)
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    
    results = []
    
    for E in energy_list:
        print(f"  Calculating E = {E:.2f} eV ...", end=" ", flush=True)
        try:
            res = compute_total_excitation_cs(
                E_incident_eV=E,
                chan=channel_spec,
                core_params=core_params,
                r_max=100.0, # Nieco mniejsze r_max dla szybkości w batchu
                n_points=2000
            )
            
            if res.ok_open_channel:
                print(f"OK. Sigma={res.sigma_total_cm2:.2e} cm2")
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": res.sigma_mtong_cm2
                })
            else:
                print("Closed channel.")
        except Exception as e:
            print(f"Error: {e}")
            
    return results

def run_ionization_scan(target_name, Z, energy_list, channel_spec: IonizationChannelSpec):
    print(f"\n--- Starting Ionization Scan for {target_name} (Z={Z}) ---")
    
    # Parametry dlad czystego Coulomba (H-like)
    core_params = CorePotentialParams(Zc=Z, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0)
    
    results = []
    
    for E in energy_list:
        print(f"  Calculating E = {E:.2f} eV ...", end=" ", flush=True)
        try:
            # Używamy nieco mniejszego l_eject_max dla szybkości skanu
            # W driverze domyślnie jest high quality, tu możemy nadpisać jeśli funkcja pozwala,
            # ale compute_ionization_cs bierze l_eject_max z channel_spec.
            
            res = compute_ionization_cs(
                E_incident_eV=E,
                chan=channel_spec,
                core_params=core_params,
                r_max=100.0,
                n_points=2000
            )
            
            if res.sigma_total_cm2 > 0.0:
                print(f"OK. Sigma={res.sigma_total_cm2:.2e} cm2")
                # M-Tong scaling manual calculation for ionization
                # sigma_scaled = sigma * E / (E + IP)
                ip = res.IP_eV
                scale = E / (E + ip) if (E + ip) > 0 else 0.0
                sigma_mtong = res.sigma_total_cm2 * scale
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": sigma_mtong,
                    "IP_eV": ip
                })
            else:
                 print("Closed channel.")
        except Exception as e:
             print(f"Error: {e}")
             import traceback
             traceback.print_exc()

    return results

def main():
    all_data = {}
    
    # --- 1. Hydrogen Excitation 1s -> 2p ---
    # Threshold ~10.2 eV. Scan range 12..100 eV. few points.
    e_list_ex = [12.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    
    spec_h_1s_2p = ExcitationChannelSpec(
        l_i=0, l_f=1,          # s -> p
        n_index_i=1, n_index_f=1, # 1s (lowest s), 2p (lowest p)
        N_equiv=1,
        L_max=5,
        L_i_total=0
    )
    
    data_h_ex = run_excitation_scan("H", 1.0, e_list_ex, spec_h_1s_2p)
    all_data["H_Excitation_1s_2p"] = data_h_ex
    
    # --- 2. Hydrogen Ionization 1s -> cont ---
    # IP ~13.6 eV. Scan range 15..100 eV.
    e_list_ion = [15.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0]
    
    spec_h_ion = IonizationChannelSpec(
        l_i=0, 
        n_index_i=1, 
        N_equiv=1, 
        l_eject_max=3,
        L_max=5,
        L_i_total=0
    )
    
    data_h_ion = run_ionization_scan("H", 1.0, e_list_ion, spec_h_ion)
    all_data["H_Ionization_1s"] = data_h_ion

    # --- 3. He+ Ionization 1s -> cont ---
    # IP ~54.4 eV. Scan range 60..300 eV.
    e_list_he = [60.0, 70.0, 90.0, 120.0, 180.0, 250.0, 350.0]
    
    # Same spec, but Z=2 in the runner call
    data_he_ion = run_ionization_scan("He+", 2.0, e_list_he, spec_h_ion)
    all_data["He_Plus_Ionization_1s"] = data_he_ion

    # Save
    with open("results_scan.json", "w") as f:
        json.dump(all_data, f, indent=2)
        
    print("\nBatch scan complete. Results saved to results_scan.json")

if __name__ == "__main__":
    main()
