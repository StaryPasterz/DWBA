# plotter.py
#
# Universal DWBA Results Plotter.
# Reads results JSON from results/ directory and generates high-quality plots.
#
# Features:
#   - Styles (std/atomic/article)
#   - Left Axis: Cross Sections (DWBA & DWBA+Tong)
#   - Right Axis: Calibration/Normalization Factor (Twin Axis)
#   - All plots saved to results/ directory
#
# Usage:
#   python plotter.py [style] [input_file]
#   
#   Examples:
#     python plotter.py std results/results_H2s_exc.json
#     python plotter.py dcs results/results_He+2p_exc.json
#

import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import math
import os
from pathlib import Path

from output_utils import get_output_path, get_results_dir, find_result_files

# Constants
AU_TO_EV = 27.211386245988
A0_SQ_CM2 = 2.8002852e-17
PI_A0_SQ_CM2 = np.pi * A0_SQ_CM2


def load_data(filename):
    """Load data from JSON file, checking results/ directory first."""
    filepath = Path(filename)
    
    # If file doesn't exist, try results/ directory
    if not filepath.exists():
        results_path = get_results_dir() / filepath.name
        if results_path.exists():
            filepath = results_path
        else:
            print(f"File {filename} not found.")
            print(f"Also checked: {results_path}")
            return {}
    
    with open(filepath, "r") as f:
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
    
    elif style_name == 'dcs':
        ev_conv = lambda e, thr: e
        sig_conv = lambda s: s  # DCS often plotted in a.u.; caller can rescale
        return ev_conv, sig_conv, "Scattering Angle [deg]", r"DCS [$a_0^2$/sr]", "_dcs"

    else: # std
        ev_conv = lambda e, thr: e
        sig_conv = lambda s: s
        return ev_conv, sig_conv, "Incident Energy [eV]", r"Cross Section [$cm^2$]", "_std"


def main():
    style = 'std'
    input_file = None
    
    # Parse arguments: [script, style, input_file]
    if len(sys.argv) > 1:
        style = sys.argv[1]
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    
    # Auto-discover input file if not specified
    if input_file is None:
        result_files = find_result_files("results_*.json")
        if result_files:
            input_file = str(result_files[-1])  # Use most recent
            print(f"Auto-selected: {input_file}")
        else:
            print("No results files found in results/ directory.")
            print("Usage: python plotter.py [style] [input_file]")
            return
    
    print(f"Generating plots with style: {style}")
    print(f"Reading from: {input_file}")
    
    data = load_data(input_file)
    if not data:
        print("No data found in JSON.")
        return

    conv_E, conv_S, xlab, ylab, suffix = get_style_config(style)
    
    keys = list(data.keys())
    n = len(keys)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    
    # DCS mode uses different layout per energy; handled separately below
    if style == 'dcs':
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n == 1: axes = [axes]
        else: axes = axes.flatten()
    
    for idx, key in enumerate(keys):
        ax = axes[idx]
        pts = data[key]
        if not pts: continue
        
        # --- DCS Plotting Mode ---
        if style == 'dcs':
            # Filter points that contain angular data
            dcs_entries = [p for p in pts if p.get("theta_deg") and p.get("dcs_au_raw")]
            if not dcs_entries:
                ax.axis('off')
                continue
            ax.set_yscale("log")
            
            # Filter by selected energies if specified
            selected_e_str = os.environ.get('DCS_ENERGIES', '')
            if selected_e_str:
                try:
                    selected_energies = [float(x.strip()) for x in selected_e_str.split(',')]
                    # Find closest matches (within 1 eV tolerance)
                    filtered = []
                    for entry in dcs_entries:
                        e = entry["energy_eV"]
                        if any(abs(e - se) < 1.0 for se in selected_energies):
                            filtered.append(entry)
                    if filtered:
                        dcs_entries = filtered
                except ValueError:
                    pass  # Use all if parsing fails
            
            # Use colormap for different energies
            import matplotlib.cm as cm
            from matplotlib.lines import Line2D
            n_entries = len(dcs_entries)
            colors = cm.viridis(np.linspace(0, 0.9, n_entries))
            
            for i, p in enumerate(dcs_entries):
                E = p["energy_eV"]
                theta = np.array(p["theta_deg"])
                dcs_raw = np.array(p["dcs_au_raw"]) if p.get("dcs_au_raw") else None
                dcs_cal = np.array(p["dcs_au_calibrated"]) if p.get("dcs_au_calibrated") else None
                if dcs_raw is None: continue
                
                # Plot raw (dashed) and calibrated (solid)
                color = colors[i]
                ax.semilogy(theta, dcs_raw, '--', color=color, alpha=0.4, linewidth=1)
                if dcs_cal is not None:
                    ax.semilogy(theta, dcs_cal, '-', color=color, alpha=0.9, linewidth=1.5, label=f"{E:.0f} eV")
                else:
                    ax.semilogy(theta, dcs_raw, '-', color=color, alpha=0.9, linewidth=1.5, label=f"{E:.0f} eV")
            
            ax.set_xlabel(xlab, fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
            ax.set_xlim(0, 180)
            
            # Explicitly enable minor ticks for log scale (required for minor grid)
            from matplotlib.ticker import LogLocator
            ax.yaxis.set_minor_locator(LogLocator(subs='auto', numticks=12))
            
            # Draw gridlines
            ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5, zorder=0)
            ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3, zorder=0)
            ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.5, zorder=0)
            
            ax.set_title(f"{key} (DCS)", fontsize=12, fontweight='bold')
            
            # Create custom legend with style info
            handles, labels = ax.get_legend_handles_labels()
            # Add style explanation entries
            style_handles = [
                Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5, label='Calibrated'),
                Line2D([0], [0], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Raw DWBA')
            ]
            ax.legend(handles=style_handles + handles, fontsize=7, ncol=3, loc='upper right')
            continue

        
        E_raw = np.array([p["energy_eV"] for p in pts])
        Sig_raw = np.array([p["sigma_cm2"] for p in pts])
        is_ionization = ("IP_eV" in pts[0]) and ("sigma_mtong_cm2" not in pts[0])

        if is_ionization:
            thr = pts[0].get("IP_eV", 1.0)
            X = [conv_E(e, thr) for e in E_raw]
            Y = [conv_S(s) for s in Sig_raw]

            if style == 'article':
                ax.plot(X, Y, 'k:', linewidth=2, label='DWBA (Ionization)')
            else:
                ax.plot(X, Y, 'o--', linewidth=2, color='tab:blue', label='DWBA (Ionization)')

            ax.set_xlabel(xlab, fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_title(key.replace("_", " "), fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            continue

        Sig_tong = np.array([p.get("sigma_mtong_cm2") for p in pts])
        Factor = np.array([p.get("calibration_factor", 1.0) for p in pts])
        
        # Check if calibration is available (sigma_mtong_cm2 is not None for all points)
        has_calibration = all(s is not None for s in Sig_tong)
        if has_calibration:
            Sig_tong = np.array([s if s is not None else 0.0 for s in Sig_tong])
        else:
            Sig_tong = None  # No calibration data
            Factor = np.ones_like(E_raw)  # Default to 1.0
        
        # For Ionization or Excitation threshold behavior
        thr = 1.0 
        if "Threshold_eV" in pts[0]: thr = pts[0]["Threshold_eV"]
        elif "IP_eV" in pts[0]: thr = pts[0]["IP_eV"]
        
        X = [conv_E(e, thr) for e in E_raw]
        Y = [conv_S(s) for s in Sig_raw]
        
        # --- Left Axis (Cross Sections) ---
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        
        if has_calibration:
            Y_tong = [conv_S(s) for s in Sig_tong]
            # Article Style: Dotted for DWBA (Uncalibrated), Solid for Calibrated
            if style == 'article':
                l1, = ax.plot(X, Y, 'k:', linewidth=2, label='DWBA (Uncalibrated)')
                l2, = ax.plot(X, Y_tong, 'k-', linewidth=2, label='Tong model')
            else:
                l1, = ax.plot(X, Y, 'o--', linewidth=2, color=color1, label='DWBA')
                l2, = ax.plot(X, Y_tong, 's-', linewidth=2, color=color2, label='Tong model')
        else:
            # No calibration - plot only DWBA curve
            if style == 'article':
                l1, = ax.plot(X, Y, 'k-', linewidth=2, label='DWBA')
            else:
                l1, = ax.plot(X, Y, 'o-', linewidth=2, color=color1, label='DWBA')
            l2 = None
        
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_title(key.replace("_", " "), fontsize=12, fontweight='bold')
        
        # --- Right Axis (Calibration Factor) - only if calibration available ---
        if has_calibration:
            ax2 = ax.twinx()
            color3 = 'tab:green'
            l3, = ax2.plot(X, Factor, ':', linewidth=1.5, color=color3, label='Norm. Factor')
            ax2.set_ylabel("Calibration Factor C(E)", fontsize=11, color=color3)
            ax2.tick_params(axis='y', labelcolor=color3)
            # Dynamic limits for factor
            if np.max(Factor) > 0:
                ax2.set_ylim(0, max(1.1, np.max(Factor)*1.1))
            
            # Legend (merge handles)
            lns = [l1, l2, l3]
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='best', fontsize=9)
        else:
            # Simple legend without calibration
            ax.legend(loc='best', fontsize=9)
        
        # Threshold Mark
        if style == 'article':
            ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
        elif style == 'std':
             pass 

    # Handle empty subplots
    if len(axes) > n:
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
    plt.tight_layout()
    
    # Derive output filename and save to results/ directory
    # e.g. results_run1_exc.json -> plot_run1_exc_std.png
    base = os.path.splitext(os.path.basename(input_file))[0]
    
    # Heuristic for nicer names
    if base.startswith("results_"):
        base = base.replace("results_", "plot_", 1)
    else:
        base = f"plot_{base}"
        
    out_name = f"{base}{suffix}.png"
    out_path = get_output_path(out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
