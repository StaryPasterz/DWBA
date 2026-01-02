# partial_wave_plotter.py
#
# Interactive tool for visualizing Partial Wave Analysis data from DWBA results.
# All plots are saved to the results/ directory.
#
# Features:
#   - Interactive file selection from results/ directory
#   - Run key selection (for files with multiple transitions)
#   - Configurable L_max for energy dependence plots
#   - Convergence analysis with L_90% indicator
#   - Summary statistics
#
# Usage:
#   python partial_wave_plotter.py [results_file.json]
#

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

from output_utils import get_output_path, get_results_dir, find_result_files
from logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# UI HELPERS
# =============================================================================

def print_header(title: str, width: int = 50):
    """Print a prominent section header."""
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def print_menu(title: str, options: list, allow_cancel: bool = True) -> int:
    """Display a numbered menu and return selected index (0-based) or -1 for cancel."""
    print(f"\n{title}:")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx}. {opt}")
    if allow_cancel:
        print("  0. Cancel")
    
    try:
        choice = int(input("\n  Select: ").strip())
        if allow_cancel and choice == 0:
            return -1
        if 1 <= choice <= len(options):
            return choice - 1
        print("  Invalid selection.")
        return -1
    except ValueError:
        print("  Invalid input.")
        return -1


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filename):
    """Load data from JSON, checking results/ directory."""
    filepath = Path(filename)
    
    if not filepath.exists():
        # Try results/ directory
        results_path = get_results_dir() / filepath.name
        if results_path.exists():
            filepath = results_path
        else:
            print(f"  Error: File '{filename}' not found.")
            logger.warning("File not found: %s", filename)
            return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_partial_waves(run_data):
    """
    Extracts partial wave data from run results.
    Returns:
        energies (np.ndarray): Array of incident energies.
        pw_data (dict): Mapping E_eV -> { 'L0': val, 'L1': val, ... }
        max_L (int): Maximum L value found in data.
    """
    energies = []
    pw_data = {}
    max_L = 0
    
    for point in run_data:
        E = point.get('energy_eV')
        pws = point.get('partial_waves')
        
        if E is not None and pws:
            energies.append(E)
            pw_data[E] = pws
            
            # Find max L in this point
            for key in pws.keys():
                if key.startswith("L"):
                    try:
                        L_val = int(key[1:])
                        max_L = max(max_L, L_val)
                    except ValueError:
                        pass
            
    return np.array(energies), pw_data, max_L


def get_convergence_L(pws: dict, threshold: float = 0.9) -> int:
    """
    Find the L value at which partial wave sum reaches threshold of total.
    E.g., threshold=0.9 returns L_90%.
    """
    l_vals = []
    sigmas = []
    for key, val in pws.items():
        if key.startswith("L"):
            try:
                l_vals.append(int(key[1:]))
                sigmas.append(val)
            except ValueError:
                pass
    
    if not l_vals:
        return 0
    
    combined = sorted(zip(l_vals, sigmas))
    sig_sorted = [x[1] for x in combined]
    
    total = sum(sig_sorted)
    if total <= 0:
        return 0
    
    cumsum = np.cumsum(sig_sorted)
    for i, (L, cs) in enumerate(zip([x[0] for x in combined], cumsum)):
        if cs / total >= threshold:
            return L
    
    return combined[-1][0]


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_distribution_at_energy(energies, pw_data, target_E, run_name: str = ""):
    """Plots Bar Chart of Sigma_L vs L for the energy closest to target_E."""
    # Find closest energy
    idx = (np.abs(energies - target_E)).argmin()
    E_closest = energies[idx]
    
    pws = pw_data[E_closest]
    
    # Sort by L index
    l_vals = []
    sigmas = []
    
    for key, val in pws.items():
        if key.startswith("L"):
            try:
                l = int(key[1:])
                l_vals.append(l)
                sigmas.append(val)
            except ValueError:
                pass
    
    if not l_vals:
        print("  No partial wave data at this energy.")
        return
            
    combined = sorted(zip(l_vals, sigmas))
    l_vals = [x[0] for x in combined]
    sigmas = [x[1] for x in combined]
    
    # Find L_90
    total = sum(sigmas)
    cumsum = np.cumsum(sigmas)
    L_90 = l_vals[-1]
    for i, cs in enumerate(cumsum):
        if total > 0 and cs / total >= 0.9:
            L_90 = l_vals[i]
            break
    
    plt.figure(figsize=(12, 5))
    
    # Left: Bar chart
    plt.subplot(1, 2, 1)
    colors = ['skyblue' if l <= L_90 else 'lightgray' for l in l_vals]
    plt.bar(l_vals, sigmas, color=colors, edgecolor='navy', alpha=0.8)
    plt.axvline(L_90, color='red', linestyle='--', linewidth=2, label=f'$L_{{90\\%}}$ = {L_90}')
    plt.xlabel(r"Projectile Partial Wave $L$", fontsize=12)
    plt.ylabel(r"Cross Section Contribution (a.u.)", fontsize=12)
    plt.title(f"Partial Wave Distribution at E = {E_closest:.2f} eV", fontsize=13)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right: Cumulative
    plt.subplot(1, 2, 2)
    plt.plot(l_vals, cumsum, 'b-o', linewidth=2, markersize=4)
    plt.axhline(total * 0.9, color='red', linestyle='--', alpha=0.7, label='90% of total')
    plt.axhline(total, color='green', linestyle='-', alpha=0.7, label='100% (total)')
    plt.axvline(L_90, color='red', linestyle=':', alpha=0.5)
    plt.xlabel(r"$L_{max}$", fontsize=12)
    plt.ylabel("Cumulative Cross Section", fontsize=12)
    plt.title("Convergence", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    safe_name = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = get_output_path(f"plot_pw_dist_{safe_name}_E{E_closest:.0f}.png")
    plt.savefig(out_path, dpi=150)
    print(f"  ✓ Saved {out_path}")
    plt.close()


def plot_convergence_all_energies(energies, pw_data, run_name: str = ""):
    """Plots Cumulative Sum vs L for selected energies to show convergence rate."""
    n = len(energies)
    if n > 5:
        indices = np.linspace(0, n-1, 5, dtype=int)
    else:
        indices = list(range(n))
        
    selected_E = energies[indices]
    
    plt.figure(figsize=(10, 6))
    
    L_90_values = []
    
    for E in selected_E:
        pws = pw_data[E]
        l_vals = []
        sigmas = []
        for key, val in pws.items():
            if key.startswith("L"):
                try:
                    l_vals.append(int(key[1:]))
                    sigmas.append(val)
                except ValueError:
                    pass
        
        if not l_vals:
            continue
            
        combined = sorted(zip(l_vals, sigmas))
        l_sorted = [x[0] for x in combined]
        sig_sorted = [x[1] for x in combined]
        cumsum = np.cumsum(sig_sorted)
        
        # Find L_90
        total = cumsum[-1] if len(cumsum) > 0 else 0
        L_90 = l_sorted[-1] if l_sorted else 0
        for i, cs in enumerate(cumsum):
            if total > 0 and cs / total >= 0.9:
                L_90 = l_sorted[i]
                break
        L_90_values.append((E, L_90))
        
        plt.plot(l_sorted, cumsum, '-o', label=f"E={E:.1f} eV (L₉₀={L_90})", markersize=4)
        
    plt.xlabel(r"$L_{max}$ (Summation Limit)", fontsize=12)
    plt.ylabel(r"Total Cross Section $\sum_{l=0}^{L_{max}} \sigma_l$", fontsize=12)
    plt.title(f"Partial Wave Convergence — {run_name}", fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    safe_name = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = get_output_path(f"plot_convergence_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"  ✓ Saved {out_path}")
    plt.close()
    
    return L_90_values


def plot_energy_dependence_of_L(energies, pw_data, max_L_to_plot: int = 5, run_name: str = ""):
    """Plots Sigma_L(E) for partial waves L=0 to max_L_to_plot."""
    l_traces = {l: [] for l in range(max_L_to_plot + 1)}
    e_axis = []
    
    for E in energies:
        pws = pw_data[E]
        e_axis.append(E)
        for l in range(max_L_to_plot + 1):
            key = f"L{l}"
            val = pws.get(key, 0.0)
            l_traces[l].append(val)
            
    plt.figure(figsize=(10, 6))
    
    # Use distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, max_L_to_plot + 1))
    
    for l in range(max_L_to_plot + 1):
        plt.plot(e_axis, l_traces[l], '-o', label=f"L={l}", 
                linewidth=2, markersize=4, color=colors[l])
        
    plt.xlabel("Incident Energy (eV)", fontsize=12)
    plt.ylabel("Partial Cross Section (a.u.)", fontsize=12)
    plt.title(f"Energy Dependence of Partial Waves (L=0..{max_L_to_plot}) — {run_name}", fontsize=14)
    plt.yscale('log')
    plt.legend(ncol=2)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    safe_name = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = get_output_path(f"plot_L_energy_dep_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"  ✓ Saved {out_path}")
    plt.close()


def plot_L90_vs_energy(L_90_values, run_name: str = ""):
    """Plot L_90% as a function of energy."""
    if not L_90_values:
        return
        
    energies = [x[0] for x in L_90_values]
    L90s = [x[1] for x in L_90_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(energies, L90s, 'ro-', linewidth=2, markersize=8)
    plt.xlabel("Incident Energy (eV)", fontsize=12)
    plt.ylabel(r"$L_{90\%}$ (90% convergence)", fontsize=12)
    plt.title(f"Convergence Requirement vs Energy — {run_name}", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    safe_name = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = get_output_path(f"plot_L90_vs_E_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"  ✓ Saved {out_path}")
    plt.close()


def print_summary(energies, pw_data, max_L, run_name: str):
    """Print summary statistics."""
    print("\n" + "─" * 50)
    print(f"  SUMMARY: {run_name}")
    print("─" * 50)
    print(f"  Energy range: {min(energies):.1f} - {max(energies):.1f} eV ({len(energies)} points)")
    print(f"  Max L in data: {max_L}")
    
    # Get L_90 at min and max energy
    L90_min = get_convergence_L(pw_data[energies[0]])
    L90_max = get_convergence_L(pw_data[energies[-1]])
    print(f"  L_90% at E_min: {L90_min}")
    print(f"  L_90% at E_max: {L90_max}")
    
    # Total cross section at first and last energy
    total_first = sum(pw_data[energies[0]].get(f"L{l}", 0) for l in range(max_L + 1))
    total_last = sum(pw_data[energies[-1]].get(f"L{l}", 0) for l in range(max_L + 1))
    print(f"  σ_total at E_min: {total_first:.3e} a.u.")
    print(f"  σ_total at E_max: {total_last:.3e} a.u.")
    print("─" * 50)


# =============================================================================
# INTERACTIVE MAIN
# =============================================================================

def select_file() -> str:
    """Interactive file selection from results/ directory."""
    files = find_result_files("results_*.json")
    
    if not files:
        print("  No result files found in results/ directory.")
        return None
    
    file_names = [str(f) for f in files]
    idx = print_menu("Available Result Files", file_names)
    
    if idx < 0:
        return None
    
    return file_names[idx]


def select_run_key(data: dict) -> str:
    """Select which run/transition to analyze if multiple are present."""
    keys = list(data.keys())
    
    if len(keys) == 1:
        print(f"  Selected: {keys[0]}")
        return keys[0]
    
    idx = print_menu("Available Runs/Transitions", keys, allow_cancel=False)
    return keys[max(0, idx)]


def interactive_analysis(filename: str, run_key: str, energies, pw_data, max_L):
    """Interactive menu for analysis options."""
    
    while True:
        print("\n" + "─" * 40)
        print("  ANALYSIS OPTIONS")
        print("─" * 40)
        print("  1. Generate ALL plots (recommended)")
        print("  2. Convergence Overview")
        print("  3. Energy Dependence (configurable L)")
        print("  4. Distribution at Specific Energy")
        print("  5. L_90% vs Energy")
        print("  6. Summary Statistics")
        print("  0. Exit")
        
        try:
            choice = input("\n  Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if choice == '0' or choice.lower() == 'q':
            break
            
        elif choice == '1':
            print("\n  Generating all plots...")
            L_90_data = plot_convergence_all_energies(energies, pw_data, run_key)
            plot_energy_dependence_of_L(energies, pw_data, min(5, max_L), run_key)
            plot_L90_vs_energy(L_90_data, run_key)
            
            # Distribution at mid-energy
            mid_E = energies[len(energies) // 2]
            plot_distribution_at_energy(energies, pw_data, mid_E, run_key)
            print_summary(energies, pw_data, max_L, run_key)
            
        elif choice == '2':
            L_90_data = plot_convergence_all_energies(energies, pw_data, run_key)
            
        elif choice == '3':
            try:
                L_plot = int(input(f"  Max L to plot [0-{max_L}, default=5]: ").strip() or "5")
                L_plot = min(max(0, L_plot), max_L)
            except ValueError:
                L_plot = min(5, max_L)
            plot_energy_dependence_of_L(energies, pw_data, L_plot, run_key)
            
        elif choice == '4':
            print(f"\n  Available energies: {min(energies):.1f} - {max(energies):.1f} eV")
            try:
                e_in = float(input("  Enter energy (eV): ").strip())
                plot_distribution_at_energy(energies, pw_data, e_in, run_key)
            except ValueError:
                print("  Invalid energy.")
                
        elif choice == '5':
            L_90_data = plot_convergence_all_energies(energies, pw_data, run_key)
            plot_L90_vs_energy(L_90_data, run_key)
            
        elif choice == '6':
            print_summary(energies, pw_data, max_L, run_key)


def main():
    print_header("PARTIAL WAVE ANALYSIS TOOL")
    
    # Get filename
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        print(f"  Using: {filename}")
    else:
        filename = select_file()
        if not filename:
            return
    
    # Load data
    data = load_data(filename)
    if not data:
        return
    
    # Select run key
    run_key = select_run_key(data)
    run_data = data[run_key]
    
    print(f"\n  Analyzing: {run_key}")
    print(f"  Data points: {len(run_data)}")
    
    # Parse partial wave data
    energies, pw_data, max_L = parse_partial_waves(run_data)
    
    if not pw_data:
        print("\n  ✗ No partial wave data found in this file.")
        print("    (Partial wave data is only available for excitation calculations)")
        return
    
    print(f"  Found {len(energies)} energies with partial wave data")
    print(f"  Max L in data: {max_L}")
    
    # Interactive analysis
    interactive_analysis(filename, run_key, energies, pw_data, max_L)
    
    print("\n  Done.")


if __name__ == "__main__":
    main()
