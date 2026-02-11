"""Check partial wave convergence in stored results more carefully."""
import json

data = json.load(open('results/results_H_exc.json', encoding='utf-8'))
print("=== H 1s->2s: Detailed Partial Wave Analysis ===")
for r in data['H_1s-2s']:
    E = r['energy_eV']
    s = r['sigma_au']
    pw_data = r.get('partial_waves', [])
    if isinstance(pw_data, list):
        n_pw = len(pw_data)
    elif isinstance(pw_data, dict):
        n_pw = len(pw_data)
    else:
        n_pw = 0
    
    # Look for max l_i in partial wave data
    max_li = 0
    if isinstance(pw_data, list):
        for pw in pw_data:
            if isinstance(pw, dict):
                li = pw.get('l_i', 0)
                if li > max_li:
                    max_li = li
    
    # Calculate expected L_max for this energy
    import numpy as np
    k = np.sqrt(2 * E / 27.211)
    L_max_safe = int(k * (200 / 2.5) - 0.5)
    
    stop = r.get('stop_reason', 'N/A')
    print(f"E={E:7.1f}  sigma={s:.4e}  n_pw={n_pw:3d}  max_li={max_li:3d}  L_safe={L_max_safe:3d}  stop={stop}")

# Also check 2p
print("\n=== H 1s->2p ===")
for r in data.get('H_1s-2p', []):
    E = r['energy_eV']
    s = r['sigma_au']
    pw_data = r.get('partial_waves', [])
    n_pw = len(pw_data) if isinstance(pw_data, (list, dict)) else 0
    
    max_li = 0
    if isinstance(pw_data, list):
        for pw in pw_data:
            if isinstance(pw, dict):
                li = pw.get('l_i', 0)
                if li > max_li:
                    max_li = li
    
    k = np.sqrt(2 * E / 27.211)
    L_max_safe = int(k * (200 / 2.5) - 0.5)
    stop = r.get('stop_reason', 'N/A')
    print(f"E={E:7.1f}  sigma={s:.4e}  n_pw={n_pw:3d}  max_li={max_li:3d}  L_safe={L_max_safe:3d}  stop={stop}")
