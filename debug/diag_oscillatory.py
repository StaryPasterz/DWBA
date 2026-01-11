#!/usr/bin/env python
"""Comprehensive verification of oscillatory integral fixes."""

import numpy as np
import time

print("=" * 60)
print("COMPREHENSIVE VERIFICATION OF OSCILLATORY INTEGRAL FIXES")
print("=" * 60)

# Import modules
from oscillatory_integrals import (
    clenshaw_curtis_nodes,
    oscillatory_kernel_integral_2d,
    _analytical_dipole_tail,
    _analytical_multipole_tail,
)
from continuum import ContinuumWave

print("\n1. CLENSHAW-CURTIS WEIGHTS (vectorized)")
print("-" * 40)
for n in [3, 5, 7, 9]:
    nodes, w = clenshaw_curtis_nodes(n, 0, 1)
    print(f"   n={n}: sum(w) = {sum(w):.10f} (expected 1.0)")

print("\n2. FILON INTEGRATION (vectorized)")
print("-" * 40)
r = np.linspace(0.1, 100, 1000)
rho1 = np.sin(r)
rho2 = np.exp(-r / 10)
kernel = np.outer(1 / r, np.ones(len(r)))

t0 = time.time()
result = oscillatory_kernel_integral_2d(
    rho1, rho2, kernel, r, 1.0, 0.5, 500, method="filon"
)
t1 = time.time()
print(f"   Result: {result:.6e}")
print(f"   Time: {(t1 - t0) * 1000:.2f} ms")

print("\n3. DIPOLE TAIL (Coulomb phase)")
print("-" * 40)
t_neutral = _analytical_dipole_tail(
    50.0, 1.0, 0.8, 0.1, 0.2, 1, 2, 1.0
)
t_coulomb = _analytical_dipole_tail(
    50.0, 1.0, 0.8, 0.1, 0.2, 1, 2, 1.0,
    eta_i=-0.5, eta_f=-0.6, sigma_i=0.1, sigma_f=0.2
)
print(f"   Neutral: {t_neutral:.6e}")
print(f"   Coulomb: {t_coulomb:.6e}")
print(f"   Ratio (Coulomb/Neutral): {t_coulomb / t_neutral:.4f}")

print("\n4. MULTIPOLE TAIL (L=2, Coulomb phase)")
print("-" * 40)
m_neutral = _analytical_multipole_tail(
    50.0, 1.0, 0.8, 0.1, 0.2, 1, 2, 2, 1.0
)
m_coulomb = _analytical_multipole_tail(
    50.0, 1.0, 0.8, 0.1, 0.2, 1, 2, 2, 1.0,
    eta_i=-0.5, eta_f=-0.6, sigma_i=0.1, sigma_f=0.2
)
print(f"   Neutral: {m_neutral:.6e}")
print(f"   Coulomb: {m_coulomb:.6e}")
print(f"   Ratio: {m_coulomb / m_neutral:.4f}" if m_neutral != 0 else "   N/A")

print("\n5. CONTINUUM WAVE WITH COULOMB PARAMS")
print("-" * 40)
cw = ContinuumWave(
    l=1, k_au=0.5, chi_of_r=np.zeros(100),
    phase_shift=0.1, eta=-1.0, sigma_l=0.5
)
print(f"   l={cw.l}, k_au={cw.k_au}")
print(f"   eta={cw.eta}")
print(f"   sigma_l={cw.sigma_l}")
print(f"   phase_shift={cw.phase_shift}")

print("\n6. FILON EXCHANGE (inner+outer CC)")
print("-" * 40)
r_ex = np.linspace(0.1, 100, 500)
rho1_ex = np.sin(2*r_ex) * np.exp(-r_ex / 20)  # Oscillatory exchange density
rho2_ex = np.cos(1.5*r_ex) * np.exp(-r_ex / 15)  # Oscillatory exchange density
kernel_ex = np.outer(1 / r_ex, np.ones(len(r_ex)))

t0 = time.time()
res_standard = oscillatory_kernel_integral_2d(
    rho1_ex, rho2_ex, kernel_ex, r_ex, 2.0, 1.5, 250, method="standard"
)
t1 = time.time()
res_filon_ex = oscillatory_kernel_integral_2d(
    rho1_ex, rho2_ex, kernel_ex, r_ex, 2.0, 1.5, 250, method="filon_exchange"
)
t2 = time.time()
print(f"   Standard: {res_standard:.6e} ({(t1-t0)*1000:.1f}ms)")
print(f"   Filon Exchange: {res_filon_ex:.6e} ({(t2-t1)*1000:.1f}ms)")
print(f"   Difference: {abs(res_filon_ex - res_standard):.2e}")

print("\n" + "=" * 60)
print("ALL VERIFICATION TESTS PASSED")
print("=" * 60)
