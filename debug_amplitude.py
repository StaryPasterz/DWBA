"""
Diagnostic script to check amplitude contributions for 1s->2p.
Tests whether the different M_f channels contribute correctly.
"""
import numpy as np
from dwba_coupling import calculate_amplitude_contribution, clebsch_gordan, racah_W
from scipy.special import sph_harm

print("=" * 60)
print("Testing amplitude calculation for 1s -> 2p transition")
print("L_i = 0 (1s), L_f = 1 (2p)")
print("=" * 60)

# Setup
theta_grid = np.array([0.0, np.pi/4, np.pi/2, np.pi])  # Just a few points
L_i, L_f = 0, 1
l_i, l_f = 0, 1  # Projectile partial waves (simplest case)
ki, kf = 1.0, 0.9  # Arbitrary k values

# Mock radial integrals (l_T = 1 is the only contributing multipole for L_i=0, L_f=1)
I_L_dir = {1: 1.0}  # Only l_T = 1
I_L_exc = {1: 0.5}

print("\nTest parameters:")
print(f"  L_i = {L_i}, L_f = {L_f}")
print(f"  l_i = {l_i}, l_f = {l_f}")
print(f"  I_L_dir = {I_L_dir}")
print(f"  I_L_exc = {I_L_exc}")

print("\n" + "=" * 60)
print("Testing each (M_i, M_f) channel:")
print("=" * 60)

for M_i in range(-L_i, L_i+1):  # Only M_i = 0 for L_i = 0
    for M_f in range(-L_f, L_f+1):  # M_f = -1, 0, +1 for L_f = 1
        mu_f_direct = M_f - M_i  # = M_f for M_i = 0
        mu_f_exchange = M_i - M_f  # = -M_f for M_i = 0
        
        print(f"\n--- M_i = {M_i}, M_f = {M_f} ---")
        print(f"  mu_f_direct = {mu_f_direct}, mu_f_exchange = {mu_f_exchange}")
        
        # Check selection rules
        valid_direct = abs(mu_f_direct) <= l_f
        valid_exchange = abs(mu_f_exchange) <= l_f
        print(f"  Direct valid: {valid_direct}, Exchange valid: {valid_exchange}")
        
        # Calculate Y_lf values
        phi = 0.0
        for theta_val in [0.0, np.pi/2]:
            if valid_direct:
                Y_direct = sph_harm(-mu_f_direct, l_f, phi, theta_val)
            else:
                Y_direct = 0.0
            if valid_exchange:
                Y_exc_base = sph_harm(-mu_f_exchange, l_f, phi, theta_val)
                Y_exchange = ((-1.0)**mu_f_exchange) * Y_exc_base
            else:
                Y_exchange = 0.0
            print(f"  theta={np.degrees(theta_val):.0f}°: Y_direct={Y_direct:.4f}, Y_exchange={Y_exchange:.4f}")
        
        # Call the actual function
        amps = calculate_amplitude_contribution(
            theta_grid, I_L_dir, I_L_exc,
            l_i, l_f, ki, kf,
            L_i, L_f, M_i, M_f
        )
        
        print(f"  |f(0°)|² = {np.abs(amps.f_theta[0])**2:.6e}")
        print(f"  |g(0°)|² = {np.abs(amps.g_theta[0])**2:.6e}")
        print(f"  |f(90°)|² = {np.abs(amps.f_theta[2])**2:.6e}")

print("\n" + "=" * 60)
print("Summing all channels:")
print("=" * 60)

# Sum amplitudes as in the DCS calculation
total_f2 = 0.0
total_g2 = 0.0
for M_i in range(-L_i, L_i+1):
    for M_f in range(-L_f, L_f+1):
        amps = calculate_amplitude_contribution(
            theta_grid, I_L_dir, I_L_exc,
            l_i, l_f, ki, kf,
            L_i, L_f, M_i, M_f
        )
        total_f2 += np.abs(amps.f_theta[0])**2
        total_g2 += np.abs(amps.g_theta[0])**2

print(f"Sum |f(0°)|² over all M: {total_f2:.6e}")
print(f"Sum |g(0°)|² over all M: {total_g2:.6e}")
