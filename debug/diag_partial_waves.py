#!/usr/bin/env python
"""
Quick Diagnostic: Analyze partial waves from existing results JSON
"""
import json
import numpy as np

# Load results
with open('results/results_H2s_exc.json') as f:
    data = json.load(f)['H_1s-2s']

print("=" * 80)
print("PARTIAL WAVE ANALYSIS - H 1s->2s")
print("=" * 80)
print(f"{'E (eV)':>8} | {'sigma_cm2':>12} | {'C(E)':>6} | L0/Total | Partial Wave Distribution")
print("-" * 80)

for entry in data:
    E = entry['energy_eV']
    sigma = entry['sigma_cm2']
    calib = entry.get('calibration_factor', 0)
    pw = entry.get('partial_waves', {})
    
    # Calculate L0/total ratio
    L0 = pw.get('L0', 0)
    total_pw = sum(v for k, v in pw.items() if k.startswith('L') and k != 'born_topup')
    ratio = L0 / total_pw if total_pw > 0 else 0
    
    # Show first 4 partial waves
    pw_str = ", ".join([f"{k}={v:.1e}" for k, v in list(pw.items())[:4]])
    
    print(f"{E:>8.2f} | {sigma:>12.3e} | {calib:>6.3f} | {ratio:.4f}   | {pw_str}")

print("\n" + "=" * 80)
print("ANOMALY DETECTION")
print("=" * 80)

# Check for anomalies
sigmas = [d['sigma_cm2'] for d in data]
energies = [d['energy_eV'] for d in data]

for i in range(1, len(data)):
    prev_sigma = data[i-1]['sigma_cm2']
    curr_sigma = data[i]['sigma_cm2']
    ratio = curr_sigma / prev_sigma if prev_sigma > 0 else 1
    
    if ratio < 0.3:  # More than 3x drop
        print(f"LARGE DROP at {data[i]['energy_eV']:.2f} eV: {prev_sigma:.2e} -> {curr_sigma:.2e} (ratio={ratio:.2f})")
    elif ratio > 3:  # More than 3x rise
        print(f"LARGE RISE at {data[i]['energy_eV']:.2f} eV: {prev_sigma:.2e} -> {curr_sigma:.2e} (ratio={ratio:.2f})")

# Check partial wave distribution
print("\n" + "=" * 80)
print("L0 DOMINANCE CHECK")
print("=" * 80)

for entry in data:
    pw = entry.get('partial_waves', {})
    L0 = pw.get('L0', 0)
    L1 = pw.get('L1', 0)
    
    if L0 > 0 and L1 > 0:
        ratio = L0 / L1
        if ratio > 1e6:
            print(f"E={entry['energy_eV']:.2f} eV: L0/L1 = {ratio:.1e} (extreme dominance!)")
