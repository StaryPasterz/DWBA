# plotter.py
#
# Universal DWBA Results Plotter (v2).
# Reads scan_results.json and generates high-quality plots.
#
# Features:
#   - Styles (std/atomic/article)
#   - Left Axis: Cross Sections (DWBA & DWBA+Tong)
#   - Right Axis: Calibration/Normalization Factor (Twin Axis)
#

import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import math
import os

RESULTS_FILE = "scan_results.json"

# Constants
AU_TO_EV = 27.211386245988
A0_SQ_CM2 = 2.8002852e-17
PI_A0_SQ_CM2 = np.pi * A0_SQ_CM2

def load_data():
    if not os.path.exists(RESULTS_FILE):
        print(f"File {RESULTS_FILE} not found.")
        sys.exit(1)
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def get_style_config(style_name):
    # Returns (conv_E, conv_S, xlabel, ylabel, suffix)
    
    if style_name == 'atomic':
        ev_conv = lambda e, thr: e / AU_TO_EV
        sig_conv = lambda s: s / A0_SQ_CM2
        return ev_conv, sig_conv, "Incident Energy [Ha]", r"Cross Section [$a_0^2$]", "_atomic"
        
    elif style_name == 'article':
        ev_conv = lambda e, thr: e / thr if thr > 1e-6 else e
        sig_conv = lambda s: s / A0_SQ_CM2
        return ev_conv, sig_conv, "Energy [$E/E_{thr}$]", r"Cross Section [a.u.]", "_article"
        
    elif style_name == 'ev_au':
        ev_conv = lambda e, thr: e 
        sig_conv = lambda s: s / A0_SQ_CM2
        return ev_conv, sig_conv, "Incident Energy [eV]", r"Cross Section [a.u.]", "_ev_au"

    else: # std
        ev_conv = lambda e, thr: e
        sig_conv = lambda s: s
        return ev_conv, sig_conv, "Incident Energy [eV]", r"Cross Section [$cm^2$]", "_std"

def main():
    style = 'std'
    if len(sys.argv) > 1:
        style = sys.argv[1]
    
    print(f"Generating plots with style: {style}")
    data = load_data()
    if not data:
        print("No data found in JSON.")
        return

    conv_E, conv_S, xlab, ylab, suffix = get_style_config(style)
    
    keys = list(data.keys())
    n = len(keys)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if n == 1: axes = [axes]
    else: axes = axes.flatten()
    
    for idx, key in enumerate(keys):
        ax = axes[idx]
        pts = data[key]
        if not pts: continue
        
        E_raw = np.array([p["energy_eV"] for p in pts])
        Sig_raw = np.array([p["sigma_cm2"] for p in pts])
        SigMT_raw = np.array([p.get("sigma_mtong_cm2", 0.0) for p in pts])
        
        # Calculate Factor (mtong / raw)
        # Handle zeros safely
        Factor = np.zeros_like(E_raw)
        mask = Sig_raw > 1e-50 # Avoid div by zero
        Factor[mask] = SigMT_raw[mask] / Sig_raw[mask]
        
        # For Ionization or Excitation threshold behavior
        thr = 1.0 
        if "Threshold_eV" in pts[0]: thr = pts[0]["Threshold_eV"]
        elif "IP_eV" in pts[0]: thr = pts[0]["IP_eV"]
        
        X = [conv_E(e, thr) for e in E_raw]
        Y = [conv_S(s) for s in Sig_raw]
        Y_MT = [conv_S(s) for s in SigMT_raw]
        
        # --- Left Axis (Cross Sections) ---
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        
        l1, = ax.plot(X, Y, 'o-', linewidth=2, color=color1, label='DWBA')
        l2, = ax.plot(X, Y_MT, 's--', linewidth=2, color=color2, label='DWBA + Calibration')
        
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_title(key.replace("_", " "), fontsize=12, fontweight='bold')
        
        # --- Right Axis (Calibration Factor) ---
        ax2 = ax.twinx()
        color3 = 'tab:green'
        l3, = ax2.plot(X, Factor, ':', linewidth=1.5, color=color3, label='Norm. Factor')
        ax2.set_ylabel("Calibration Factor", fontsize=11, color=color3)
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.set_ylim(0, 1.1) # Factor usually 0..1
        
        # Legend (Merge handles)
        lns = [l1, l2, l3]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='best', fontsize=9)
        
        # Threshold Mark
        if style == 'article':
            ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
        elif style == 'std':
             pass 

    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    out_name = f"plot_combined{suffix}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
