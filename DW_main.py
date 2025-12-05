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
    ev_to_au,
)
from ionization import (
    IonizationChannelSpec,
    compute_ionization_cs,
)

AU_TO_EV = 27.211386245988 
A0_SQ_CM2 = 2.8002852e-17
PI_A0_SQ_CM2 = np.pi * A0_SQ_CM2

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

def format_sigma(sigma_cm2: float, unit: str) -> str:
    if unit == "cm2":
        return f"{sigma_cm2:.4e} cm^2"
    elif unit == "a02":
        val = sigma_cm2 / A0_SQ_CM2
        return f"{val:.4e} a0^2"
    elif unit == "pia02":
        val = sigma_cm2 / PI_A0_SQ_CM2
        return f"{val:.4e} pi*a0^2"
    return f"{sigma_cm2:.4e} ?"

def format_energy(E_eV: float, unit: str) -> str:
    if unit == "eV":
        return f"{E_eV:.4f} eV"
    elif unit == "Ha":
        return f"{E_eV / AU_TO_EV:.4f} Ha"
    return f"{E_eV:.4f} ?"

def main():
    print("=== DWBA Electron Impact Cross Section Calculator ===")
    print("1. Excitation (Bound -> Bound)")
    print("2. Ionization (Bound -> Continuum)")
    
    mode = get_user_input_int("Select mode", 1)
    
    # Units selection
    print("\n--- Unit Selection ---")
    print("Select Energy unit:")
    print("1. eV (Default)")
    print("2. Hartree")
    u_e = get_user_input_int("Choice", 1)
    unit_E = "Ha" if u_e == 2 else "eV"
    
    print("Select Cross Section unit:")
    print("1. cm^2 (Default)")
    print("2. a0^2 (Bohr radius squared)")
    print("3. pi * a0^2")
    u_s = get_user_input_int("Choice", 1)
    if u_s == 2:
        unit_Sig = "a02"
    elif u_s == 3:
        unit_Sig = "pia02"
    else:
        unit_Sig = "cm2"

    # Common params
    print("\n--- Core Potential Parameters (H-like / generic) ---")
    Z = get_user_input_float("Nuclear charge Z", 1.0)
    core_params = CorePotentialParams(Zc=Z, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, a6=0.0)

    print("\n--- Incident Energy ---")
    
    if unit_E == "Ha":
        E_inc_au = get_user_input_float("Incident Electron Energy [Ha]", 2.0)
        E_inc_eV = E_inc_au * AU_TO_EV
    else:
        E_inc_eV = get_user_input_float("Incident Electron Energy [eV]", 50.0)

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
        
        print(f"\nComputing Excitation σ for E={format_energy(E_inc_eV, unit_E)}...")
        try:
            res = compute_total_excitation_cs(E_inc_eV, spec, core_params)
            
            if res.ok_open_channel:
                print(f"\nRESULT:")
                print(f"  Excitation Energy: {format_energy(res.E_excitation_eV, unit_E)}")
                print(f"  Sigma DWBA:        {format_sigma(res.sigma_total_cm2, unit_Sig)}")
                print(f"  Sigma M-Tong:      {format_sigma(res.sigma_mtong_cm2, unit_Sig)}")
            else:
                thr = res.E_excitation_eV
                print(f"\nChannel closed! (Threshold {format_energy(thr, unit_E)} > Incident {format_energy(E_inc_eV, unit_E)})")
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
        
        print(f"\nComputing Ionization σ for E={format_energy(E_inc_eV, unit_E)}...")
        print("Integration over ejected energy takes time...")
        try:
            res = compute_ionization_cs(E_inc_eV, spec, core_params)
            
            print(f"\nRESULT:")
            print(f"  Ionization Potential: {format_energy(res.IP_eV, unit_E)}")
            print(f"  Total Ionization CS:  {format_sigma(res.sigma_total_cm2, unit_Sig)}")
                 
        except Exception as e:
             print(f"Error: {e}")
             import traceback
             traceback.print_exc()

    else:
        print("Unknown mode.")

if __name__ == "__main__":
    main()
