"""
Diagnostic script to verify angular coupling coefficients for 1s->2p transition.

This tests the CG and Racah coefficients that appear in Eq. 412 for the specific
case of L_i=0 (1s) -> L_f=1 (2p).
"""

import numpy as np
from dwba_coupling import clebsch_gordan, racah_W, wigner_6j_num

print("=" * 60)
print("Angular Coefficient Diagnostics for 1s -> 2p (L_i=0, L_f=1)")
print("=" * 60)

L_i = 0  # Initial target (1s)
L_f = 1  # Final target (2p)

# For 1s -> 2p, the constraint C(l_T, L_i, L_f; 0,0,0) = C(l_T, 0, 1; 0,0,0)
# requires l_T = 1 (triangle rule + parity)
print("\n--- CG(l_T, L_i=0, L_f=1; 0,0,0) for various l_T ---")
for l_T in range(4):
    cg = clebsch_gordan(l_T, L_i, L_f, 0, 0, 0)
    print(f"  l_T={l_T}: CG = {cg:.6f}")

print("\n--- Testing Eq. 412 coefficients for l_i=0, l_f=1, l_T=1 ---")
l_i = 0
l_f = 1
l_T = 1

# CG1: C(l_f, l_i, l_T; 0,0,0) = C(1, 0, 1; 0,0,0)
cg1 = clebsch_gordan(l_f, l_i, l_T, 0, 0, 0)
print(f"  CG(l_f={l_f}, l_i={l_i}, l_T={l_T}; 0,0,0) = {cg1:.6f}")

# CG2: C(l_T, L_i, L_f; 0,0,0) = C(1, 0, 1; 0,0,0)
cg2 = clebsch_gordan(l_T, L_i, L_f, 0, 0, 0)
print(f"  CG(l_T={l_T}, L_i={L_i}, L_f={L_f}; 0,0,0) = {cg2:.6f}")

# Racah W(l_f, l_i, L_f, L_i; l_T, g) for various g
print(f"\n--- Racah W(l_f={l_f}, l_i={l_i}, L_f={L_f}, L_i={L_i}; l_T={l_T}, g) ---")
print(f"    g bounds: max(|l_i-L_i|, |l_f-L_f|) to min(l_i+L_i, l_f+L_f)")
g_min = max(abs(l_i - L_i), abs(l_f - L_f))
g_max = min(l_i + L_i, l_f + L_f)
print(f"    g_min={g_min}, g_max={g_max}")

for g in range(g_min, g_max + 1):
    w = racah_W(l_f, l_i, L_f, L_i, l_T, g)
    print(f"  g={g}: W = {w:.6f}")

# Now test for higher l_i, l_f with l_T=1 (the only contributing multipole)
print("\n--- Testing various (l_i, l_f) combinations with l_T=1 ---")
for l_i_test in range(5):
    for l_f_test in range(5):
        # Parity check for CG1: l_f + l_i + l_T must be even
        if (l_f_test + l_i_test + l_T) % 2 != 0:
            continue
        cg1_test = clebsch_gordan(l_f_test, l_i_test, l_T, 0, 0, 0)
        if abs(cg1_test) < 1e-9:
            continue
        
        # Check if this contributes for the 1s->2p transition
        # CG2 is always C(1, 0, 1; 0,0,0) = constant for L_i=0, L_f=1, l_T=1
        
        # e factors
        e_lf = np.sqrt(2*l_f_test + 1)
        e_li = np.sqrt(2*l_i_test + 1)
        e_Li = np.sqrt(2*L_i + 1)  # sqrt(1) = 1
        e_lT = np.sqrt(2*l_T + 1)  # sqrt(3)
        
        pre_R = (e_lf * e_li * e_Li) / e_lT
        
        print(f"\n  (l_i={l_i_test}, l_f={l_f_test}):")
        print(f"    CG1 = {cg1_test:.6f}")
        print(f"    pre_R = {pre_R:.6f}")
        
        # Sum over g
        g_min_t = max(abs(l_i_test - L_i), abs(l_f_test - L_f))
        g_max_t = min(l_i_test + L_i, l_f_test + L_f)
        
        sum_g = 0.0
        for g in range(g_min_t, g_max_t + 1):
            w = racah_W(l_f_test, l_i_test, L_f, L_i, l_T, g)
            # For M_i = 0, mu_i = 0, mu_f = 0 - M_f
            # CG_A = C(l_i, L_i, g; 0, 0, 0) 
            cg_A = clebsch_gordan(l_i_test, L_i, g, 0, 0, 0)
            # For M_f = 0: mu_f = 0, CG_B = C(l_f, L_f, g; 0, 0, 0)
            cg_B = clebsch_gordan(l_f_test, L_f, g, 0, 0, 0)
            
            sum_g += w * cg_A * cg_B
            if abs(w * cg_A * cg_B) > 1e-9:
                print(f"    g={g}: W={w:.4f}, CG_A={cg_A:.4f}, CG_B={cg_B:.4f}, product={w*cg_A*cg_B:.6f}")
        
        total_term = pre_R * cg1_test * cg2 * sum_g
        print(f"    Total angular factor (excl. radial): {total_term:.6f}")

print("\n" + "=" * 60)
print("Compare |angular factor|^2 ratios between channels")
print("=" * 60)
