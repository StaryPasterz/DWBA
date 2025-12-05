
"""
DW_main.py

Główny punkt wejścia do obliczeń DWBA (Wzbudzenie i Jonizacja).
Obsługuje interakcję z użytkownikiem i uruchamia odpowiednie solvery.
"""

import sys
import numpy as np
from potential_core import CorePotentialParams
from driver import (
    ExcitationChannelSpec,
    compute_total_excitation_cs,
)
from ionization import (
    IonizationChannelSpec,
    compute_ionization_cs,
)

def get_user_input_float(prompt: str, default: float = None) -> float:
    try:
        if default is not None:
            raw = input(f"{prompt} [{default}]: ")
            if not raw.strip():
                return default
        else:
            raw = input(f"{prompt}: ")
        return float(raw)
    except ValueError:
        print("Invalid number.")
        return get_user_input_float(prompt, default)

def get_user_input_int(prompt: str, default: int = None) -> int:
    try:
        if default is not None:
            raw = input(f"{prompt} [{default}]: ")
            if not raw.strip():
                return default
        else:
            raw = input(f"{prompt}: ")
        return int(raw)
    except ValueError:
        print("Invalid integer.")
        return get_user_input_int(prompt, default)

def main():
    print("=== DWBA Electron Impact Cross Section Calculator ===")
    print("1. Excitation (Bound -> Bound)")
    print("2. Ionization (Bound -> Continuum)")
    
    mode = get_user_input_int("Select mode", 1)
    
    # Common params
    print("\n--- Core Potential Parameters (H-like / generic) ---")
    Z = get_user_input_float("Nuclear charge Z", 1.0)
    # Prosty potencjał kulombowski -Z/r + krótkozasięgowe?
    # W tym kodzie V_coreParams ma: Z, alpha1, alpha2...
    # Użyjmy domyślnych zer dla alph, czyli czysty Coulomb.
    # Chyba że użytkownik chce podać parametry.
    # Uproszczenie: tylko Z.
    core_params = CorePotentialParams(Zc=Z, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, a6=0.0) # Gołe jądro Z (lub efektywny)

    print("\n--- Incident Energy ---")
    E_inc = get_user_input_float("Incident Electron Energy [eV]", 50.0)

    if mode == 1:
        # EXCITATION
        print("\n--- Excitation Channel ---")
        l_i = get_user_input_int("Initial l_i", 0)
        n_i = get_user_input_int("Initial n_index (1=ground)", 1)
        
        l_f = get_user_input_int("Final l_f", 1)
        n_f = get_user_input_int("Final n_index", 1)
        
        N_equiv = get_user_input_int("Number of equivalent electrons", 1)
        L_max = get_user_input_int("Max Multipole L", 5)
        L_i_tot = get_user_input_int("Target Total L_i", l_i)
        
        spec = ExcitationChannelSpec(
            l_i=l_i, l_f=l_f, 
            n_index_i=n_i, n_index_f=n_f,
            N_equiv=N_equiv, L_max=L_max, L_i_total=L_i_tot
        )
        
        print(f"\nComputing Excitation σ for E={E_inc} eV...")
        try:
            res = compute_total_excitation_cs(E_inc, spec, core_params)
            if res.ok_open_channel:
                print(f"\nRESULT:")
                print(f"  Excitation Energy: {res.E_excitation_eV:.4f} eV")
                print(f"  Sigma DWBA:        {res.sigma_total_cm2:.4e} cm^2")
                print(f"  Sigma M-Tong:      {res.sigma_mtong_cm2:.4e} cm^2")
            else:
                print(f"\nChannel closed! (Threshold {res.E_excitation_eV:.4f} eV > Incident {E_inc:.4f} eV)")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    elif mode == 2:
        # IONIZATION
        print("\n--- Ionization Channel ---")
        l_i = get_user_input_int("Initial l_i", 0)
        n_i = get_user_input_int("Initial n_index (1=ground)", 1)
        N_equiv = get_user_input_int("Number of equivalent electrons", 1)
        
        l_eject_max = get_user_input_int("Max ejected electron l (sum up to)", 3)
        L_max = get_user_input_int("Max Multipole L (for interaction)", 5)
        L_i_tot = get_user_input_int("Target Total L_i", l_i)
        
        spec = IonizationChannelSpec(
            l_i=l_i, 
            n_index_i=n_i, 
            N_equiv=N_equiv,
            l_eject_max=l_eject_max,
            L_max=L_max,
            L_i_total=L_i_tot
        )
        
        print(f"\nComputing Ionization σ for E={E_inc} eV...")
        print("Integration over ejected energy takes time...")
        try:
            res = compute_ionization_cs(E_inc, spec, core_params)
            print(f"\nRESULT:")
            print(f"  Ionization Potential: {res.IP_eV:.4f} eV")
            print(f"  Total Ionization CS:  {res.sigma_total_cm2:.4e} cm^2")
        except Exception as e:
             print(f"Error: {e}")
             import traceback
             traceback.print_exc()

    else:
        print("Unknown mode.")

if __name__ == "__main__":
    main()
