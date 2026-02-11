"""Compare computed DWBA results against reference data from dwba_reference_points.md"""
import json
import numpy as np

# Reference data from article (digitized from plots)
# Format: E_eV, sigma_cal(Tong), sigma_DWBA, N = cal/DWBA  (all in a.u.)
ref_H_2s = {
    50:  {'sigma_cal': 0.2529, 'sigma_DWBA': 0.3706, 'N': 0.6825},
    100: {'sigma_cal': 0.1382, 'sigma_DWBA': 0.2029, 'N': 0.6812},
    200: {'sigma_cal': 0.07647,'sigma_DWBA': 0.1000, 'N': 0.7647},
    500: {'sigma_cal': 0.03529,'sigma_DWBA': 0.03824,'N': 0.9231},
}

ref_H_2p = {
    100: {'sigma_cal': 1.6618, 'sigma_DWBA': 2.2059, 'N': 0.7533},
    200: {'sigma_cal': 1.0735, 'sigma_DWBA': 1.5294, 'N': 0.7019},
    500: {'sigma_cal': 0.5735, 'sigma_DWBA': 0.7206, 'N': 0.7959},
}

def find_closest(results, target_eV, tolerance=15):
    best = None
    best_dist = float('inf')
    for r in results:
        d = abs(r['energy_eV'] - target_eV)
        if d < best_dist:
            best = r
            best_dist = d
    if best_dist > tolerance:
        return None
    return best

def compare(label, results, references):
    print("\n" + "=" * 70)
    print(" " + label)
    print("=" * 70)
    print("{:>8s} {:>15s} {:>16s} {:>8s} {:>8s}".format(
        "E(eV)", "sigma_ref", "sigma_code", "ratio", "error%"))
    print("-" * 70)
    
    for E_ref, ref in sorted(references.items()):
        r = find_closest(results, E_ref)
        if r is None:
            print("{:>8.0f} {:>15.4f} {:>16s} {:>8s} {:>8s}".format(
                E_ref, ref['sigma_DWBA'], "(no data)", "N/A", "N/A"))
            continue
        
        E_actual = r['energy_eV']
        sigma_code = r.get('sigma_au', 0)
        sigma_ref = ref['sigma_DWBA']
        
        ratio = sigma_code / sigma_ref if sigma_ref > 0 else float('inf')
        error_pct = (ratio - 1) * 100
        
        print("{:>8.1f} {:>15.4f} {:>16.4f} {:>8.3f} {:>+8.1f}%".format(
            E_actual, sigma_ref, sigma_code, ratio, error_pct))

# Load data
for fname, key2s, key2p in [
    ('results/results_H_exc.json', 'H_1s-2s', 'H_1s-2p'),
    ('results/results_H2s_exc.json', 'H_1s-2s', None),
    ('results/results_H2p_exc.json', None, 'H_1s-2p'),
]:
    try:
        data = json.load(open(fname, encoding='utf-8'))
        if key2s and key2s in data:
            compare("H 1s->2s from %s" % fname, data[key2s], ref_H_2s)
        if key2p and key2p in data:
            compare("H 1s->2p from %s" % fname, data[key2p], ref_H_2p)
    except Exception as e:
        print("Error loading %s: %s" % (fname, e))

# He+ results
print("\n\n=== He+ Results ===")
for fname, key in [
    ('results/results_He+2s_exc.json', 'He+_1s-2s'),
    ('results/results_He+2p_exc.json', 'He+_1s-2p'),
]:
    try:
        data = json.load(open(fname, encoding='utf-8'))
        results = data.get(key, [])
        print("\n%s: %d energy points" % (key, len(results)))
        for r in results:
            e = r['energy_eV']
            sa = r.get('sigma_au', 'N/A')
            sc = r.get('sigma_cm2', 'N/A')
            cal = r.get('calibration', {})
            print("  E=%7.1f: sigma_au=%.6e  sigma_cm2=%.6e  cal_factor=%.4f" % (
                e, sa, sc if isinstance(sc, float) else 0, 
                cal.get('calibration_factor', 0) if isinstance(cal, dict) else 0))
    except Exception as e:
        print("Error loading %s: %s" % (fname, e))
