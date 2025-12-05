# plotter.py
#
# Universal DWBA Results Plotter.
# Reads scan_results.json and generates high-quality plots for all datasets found.
#
# Usage: python plotter.py [style]
# Styles:
#   std     -> E[eV] vs Sigma[cm^2]
#   atomic  -> E[Ha] vs Sigma[a0^2]
#   article -> E[u]  vs Sigma[pi a0^2]  (u = E/Threshold)
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
    # Returns (convert_E_func, convert_Sig_func, xlabel, ylabel, file_suffix)
    
    if style_name == 'atomic':
        ev_conv = lambda e, thr: e / AU_TO_EV
        sig_conv = lambda s: s / A0_SQ_CM2
        return ev_conv, sig_conv, "Incident Energy [Ha]", r"Cross Section [$a_0^2$]", "_atomic"
        
    elif style_name == 'article':
        ev_conv = lambda e, thr: e / thr if thr > 1e-6 else e
        sig_conv = lambda s: s / PI_A0_SQ_CM2
        return ev_conv, sig_conv, "Energy [$E/E_{thr}$]", r"Cross Section [$\pi a_0^2$]", "_article"
        
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
    
    # 1. Generate Combined Grid Plot
    keys = list(data.keys())
    n = len(keys)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))
    if n == 1: axes = [axes]
    else: axes = axes.flatten()
    
    for idx, key in enumerate(keys):
        ax = axes[idx]
        pts = data[key]
        if not pts: continue
        
        # Prepare arrays
        # Filter zero energy or negative? No, keep all but handle threshold.
        
        E_raw = np.array([p["energy_eV"] for p in pts])
        Sig_raw = np.array([p["sigma_cm2"] for p in pts])
        SigMT_raw = np.array([p.get("sigma_mtong_cm2", 0.0) for p in pts])
        
        # Threshold
        thr = 1.0 # fallback
        if "Threshold_eV" in pts[0]: thr = pts[0]["Threshold_eV"]
        
        # Convert
        X = [conv_E(e, thr) for e in E_raw]
        Y = [conv_S(s) for s in Sig_raw]
        Y_MT = [conv_S(s) for s in SigMT_raw]
        
        # Plot
        ax.plot(X, Y, 'o-', linewidth=2, label='DWBA')
        ax.plot(X, Y_MT, 's--', linewidth=2, label='DWBA + M-Tong')
        
        # Formatting
        readable_title = key.replace("_", " ")
        ax.set_title(readable_title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Mark Threshold if applicable
        if style == 'article':
            ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
        elif style == 'std':
            ax.axvline(thr, color='red', linestyle='--', alpha=0.5, label=f'Thr={thr:.1f}eV')

    # Remove empty plots
    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    out_name = f"plot_combined{suffix}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")
    
    # Optional: Generate individual plots? Use subplots is usually better for overview.

if __name__ == "__main__":
    main()
