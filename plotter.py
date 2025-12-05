# plotter.py
#
# Wizualizacja wyników z batch_runner.py
# Usage: python plotter.py [output_filename] [--E-unit UNIT] [--Sig-unit UNIT]
#   UNITs for E: eV, Ha, u (Threshold)
#   UNITs for Sig: cm2, a02, pia02
#

import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import argparse

# Stałe
AU_TO_EV = 27.211386
A0_SQ_CM2 = 2.8002852e-17
PI_A0_SQ_CM2 = np.pi * A0_SQ_CM2

# Hardcoded thresholds for known keys (since not currently in JSON for excitation)
THRESHOLDS_EV = {
    "H_Excitation_1s_2p": 10.1988, # 13.6 * (1 - 1/4)
    # For Ionization, we will try to read IP_eV from JSON data points
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot DWBA cross sections.")
    parser.add_argument("output", nargs='?', default="cross_sections_plot_custom.png", help="Output filename")
    parser.add_argument("--E-unit", default="eV", choices=["eV", "Ha", "u"], help="Energy unit")
    parser.add_argument("--Sig-unit", default="cm2", choices=["cm2", "a02", "pia02"], help="Cross section unit")
    return parser.parse_args()

def convert_E(E_ev, unit, threshold_ev=None):
    if unit == "eV":
        return E_ev, "Incident Energy [eV]"
    elif unit == "Ha":
        return E_ev / AU_TO_EV, "Incident Energy [Ha]"
    elif unit == "u":
        if threshold_ev is None:
            return E_ev, "Incident Energy [eV] (No Thr)"
        return E_ev / threshold_ev, "Incident Energy [E/I]"
    return E_ev, "Energy"

def convert_Sig(Sig_cm2, unit):
    if unit == "cm2":
        return Sig_cm2, "Cross Section [cm^2]"
    elif unit == "a02":
        return Sig_cm2 / A0_SQ_CM2, "Cross Section [a0^2]"
    elif unit == "pia02":
        return Sig_cm2 / PI_A0_SQ_CM2, "Cross Section [pi a0^2]"
    return Sig_cm2, "Cross Section"

def plot_results():
    args = parse_arguments()
    
    try:
        with open("results_scan.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Brak pliku results_scan.json.")
        return

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Helper to process one dataset
    def process_and_plot(ax, key, title, color_base):
        if key not in data or not data[key]:
            ax.text(0.5, 0.5, "No Data", ha='center')
            return

        pts = data[key]
        E_ev = np.array([p["energy_eV"] for p in pts])
        Sig_cm2 = np.array([p["sigma_cm2"] for p in pts])
        SigMT_cm2 = np.array([p.get("sigma_mtong_cm2", 0.0) for p in pts])
        
        # Determine Threshold
        thr = THRESHOLDS_EV.get(key)
        if thr is None:
            # Try finding IP in points
            if "IP_eV" in pts[0]:
                thr = pts[0]["IP_eV"]
            else:
                thr = 1.0 # fallback
        
        # Convert
        X, xlab = convert_E(E_ev, args.E_unit, thr)
        Y, ylab = convert_Sig(Sig_cm2, args.Sig_unit)
        Y_MT, _ = convert_Sig(SigMT_cm2, args.Sig_unit)
        
        # Plot
        ax.plot(X, Y, 'o-', color=color_base, label='DWBA')
        ax.plot(X, Y_MT, 's--', color=adjust_lightness(color_base, 0.7), label='DWBA + M-Tong')
        
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        
        # Add Threshold marker if relevant (u=1, or E=Thr)
        if args.E_unit == 'u':
             ax.axvline(x=1.0, color='k', linestyle=':', label='Threshold')
        elif args.E_unit == 'eV':
             ax.axvline(x=thr, color='k', linestyle=':', label=f'Thr={thr:.1f}eV')
        elif args.E_unit == 'Ha':
             thr_au = thr / AU_TO_EV
             ax.axvline(x=thr_au, color='k', linestyle=':', label=f'Thr={thr_au:.2f}Ha')

        ax.legend()
        ax.grid(True)

    # 1. H Excitation
    process_and_plot(axes[0], "H_Excitation_1s_2p", "H(1s->2p) Excitation", "blue")

    # 2. H Ionization
    process_and_plot(axes[1], "H_Ionization_1s", "H(1s) Ionization", "red")

    # 3. He+ Ionization
    process_and_plot(axes[2], "He_Plus_Ionization_1s", "He+(1s) Ionization", "green")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

if __name__ == "__main__":
    plot_results()
