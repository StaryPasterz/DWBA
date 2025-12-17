# partial_wave_plotter.py
#
# Tool for visualizing Partial Wave Analysis data from DWBA results.
#

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return {}
    with open(filename, 'r') as f:
        return json.load(f)

def parse_partial_waves(run_data):
    """
    Extracts partial wave data from run results.
    Returns:
        energies (list): List of incident energies.
        pw_data (dict): Mapping E_eV -> { 'L0': val, 'L1': val, ... }
    """
    energies = []
    pw_data = {}
    
    for point in run_data:
        E = point.get('energy_eV')
        pws = point.get('partial_waves')
        
        if E is not None and pws:
            energies.append(E)
            # pws is expected to be {'L0': val, 'L1': val ...}
            # Ensure keys are sorted for processing
            pw_data[E] = pws
            
    return np.array(energies), pw_data

def plot_distribution_at_energy(energies, pw_data, target_E):
    """
    Plots Bar Chart of Sigma_L vs L for the energy closest to target_E.
    """
    # Find closest energy
    idx = (np.abs(energies - target_E)).argmin()
    E_closest = energies[idx]
    
    pws = pw_data[E_closest]
    
    # Sort by L index (L0, L1, L2 -> 0, 1, 2)
    l_vals = []
    sigmas = []
    
    for key, val in pws.items():
        if key.startswith("L"):
            try:
                l = int(key[1:])
                l_vals.append(l)
                sigmas.append(val)
            except: pass
            
    # Sort
    combined = sorted(zip(l_vals, sigmas))
    l_vals = [x[0] for x in combined]
    sigmas = [x[1] for x in combined]
    
    plt.figure(figsize=(10, 6))
    plt.bar(l_vals, sigmas, color='skyblue', edgecolor='navy')
    plt.xlabel(r"Projectile Partial Wave $L$", fontsize=12)
    plt.ylabel(r"Cross Section Contribution (a.u.)", fontsize=12)
    plt.title(f"Partial Wave Distribution at E = {E_closest:.2f} eV", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accumulation line?
    ax2 = plt.gca().twinx()
    cumsum = np.cumsum(sigmas)
    ax2.plot(l_vals, cumsum, 'r-o', linewidth=2, label='Cumulative')
    ax2.set_ylabel("Cumulative Cross Section", color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    out = f"plot_distribution_E{E_closest:.0f}.png"
    plt.savefig(out, dpi=150)
    print(f"   Saved {out}")
    plt.close()

def plot_convergence_all_energies(energies, pw_data):
    """
    Plots Cumulative Sum vs L for a few selected energies to show convergence rate.
    """
    # Select a few energies: Low, Mid, High
    n = len(energies)
    indices = [0, n//2, n-1]
    if n > 5:
        # Pick 5 points evenly
        indices = np.linspace(0, n-1, 5, dtype=int)
        
    selected_E = energies[indices]
    
    plt.figure(figsize=(10, 6))
    
    max_l_overall = 0
    
    for E in selected_E:
        pws = pw_data[E]
        l_vals = []
        sigmas = []
        for key, val in pws.items():
            if key.startswith("L"):
                l_vals.append(int(key[1:]))
                sigmas.append(val)
        
        combined = sorted(zip(l_vals, sigmas))
        l_sorted = [x[0] for x in combined]
        if not l_sorted: continue
        
        sig_sorted = [x[1] for x in combined]
        cumsum = np.cumsum(sig_sorted)
        max_l_overall = max(max_l_overall, max(l_sorted))
        
        # Normalize? No, absolute is better to see magnitude diffs.
        plt.plot(l_sorted, cumsum, '-o', label=f"E={E:.1f} eV", markersize=4)
        
    plt.xlabel(r"$L_{max}$ (Summation Limit)", fontsize=12)
    plt.ylabel(r"Total Cross Section $\sum_{l=0}^{L_{max}} \sigma_l$", fontsize=12)
    plt.title("Partial Wave Convergence", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    out = "plot_convergence.png"
    plt.savefig(out, dpi=150)
    print(f"   Saved {out}")
    plt.close()

def plot_energy_dependence_of_L(energies, pw_data):
    """
    Plots Sigma_L(E) for the first few L modes (0, 1, 2, 3...)
    """
    max_L_to_plot = 5
    
    # Organise data: L -> [sigma(E0), sigma(E1)...]
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
    for l in range(max_L_to_plot + 1):
        plt.plot(e_axis, l_traces[l], '-o', label=f"L={l}", linewidth=2, markersize=4)
        
    plt.xlabel("Incident Energy (eV)", fontsize=12)
    plt.ylabel("Partial Cross Section (a.u.)", fontsize=12)
    plt.title(f"Energy Dependence of Dominant Partial Waves (L=0..{max_L_to_plot})", fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out = "plot_L_energy_dep.png"
    plt.savefig(out, dpi=150)
    print(f"   Saved {out}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python partial_wave_plotter.py <results_file.json>")
        
        # Try finding default
        files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.json')]
        if files:
            print(f"\nFound potential files: {files}")
            print(f"Defaulting to: {files[-1]}")
            filename = files[-1]
        else:
            return
    else:
        filename = sys.argv[1]
        
    data = load_data(filename)
    if not data: return
    
    # Data is usually { 'Key_Name': [ {run_point}, ... ] }
    # Let's extract the first run found
    keys = list(data.keys())
    print(f"Found runs: {keys}")
    
    run_key = keys[0] # Pick first
    run_data = data[run_key]
    
    print(f"Analyzing run: {run_key} ({len(run_data)} points)")
    
    energies, pw_data = parse_partial_waves(run_data)
    
    if not pw_data:
        print("No Partial Wave data found in this file.")
        return

    print("\n--- Generating Plots ---")
    print("1. Convergence Overview (All Energies)")
    plot_convergence_all_energies(energies, pw_data)
    
    print("2. Energy Dependence (L=0..5 vs Energy)")
    plot_energy_dependence_of_L(energies, pw_data)
    
    # Interactive single point
    print("3. Distribution at Specific Energy")
    try:
        e_in = float(input(f"   Enter energy to inspect (Available: {min(energies):.1f} - {max(energies):.1f} eV): "))
        plot_distribution_at_energy(energies, pw_data, e_in)
    except ValueError:
        print("   Skipping single energy plot.")

if __name__ == "__main__":
    main()
