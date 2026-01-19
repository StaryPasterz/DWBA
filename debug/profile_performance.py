"""
Performance Profiling Script - Identify bottlenecks in DWBA calculations.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
from functools import wraps

# Profiling decorator
timing_stats = {}

def profile(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if name not in timing_stats:
                timing_stats[name] = {'count': 0, 'total': 0.0}
            timing_stats[name]['count'] += 1
            timing_stats[name]['total'] += elapsed
            return result
        return wrapper
    return decorator

def print_timing_stats():
    print("\n" + "="*70)
    print("PERFORMANCE BREAKDOWN")
    print("="*70)
    total = sum(s['total'] for s in timing_stats.values())
    sorted_stats = sorted(timing_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for name, stats in sorted_stats:
        pct = 100 * stats['total'] / total if total > 0 else 0
        avg = stats['total'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{name:40s}: {stats['total']:8.2f}s ({pct:5.1f}%) | {stats['count']:4d} calls | {avg*1000:6.1f}ms avg")
    print("="*70)
    print(f"{'TOTAL':40s}: {total:8.2f}s")

# Import after defining profiler
from grid import make_r_grid, k_from_E_eV
from potential_core import CorePotentialParams, V_core_on_grid
from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from dwba_matrix_elements import radial_ME_all_L_gpu
from dwba_coupling import calculate_amplitude_contribution
from sigma_total import dcs_dwba, integrate_dcs_over_angles

print("="*70)
print("DWBA PERFORMANCE PROFILER")
print("="*70)

# Setup - measure each component
t0 = time.perf_counter()
grid = make_r_grid(1e-5, 200.0, 3000)
t_grid = time.perf_counter() - t0
print(f"Grid creation: {t_grid*1000:.1f} ms (n={len(grid.r)} points)")

t0 = time.perf_counter()
core = CorePotentialParams(Zc=1.0, a1=0, a2=1, a3=0, a4=1, a5=0, a6=1)
V = V_core_on_grid(grid, core)
t_potential = time.perf_counter() - t0
print(f"Core potential: {t_potential*1000:.1f} ms")

t0 = time.perf_counter()
states = solve_bound_states(grid, V, 0, 3)
t_bound = time.perf_counter() - t0
print(f"Bound states: {t_bound*1000:.1f} ms (3 states)")

orb_i, orb_f = states[0], states[1]

# Test at 300 eV (high energy case)
E_eV = 300.0
dE = (orb_f.energy_au - orb_i.energy_au) * 27.211
E_final = E_eV - dE
ki = float(k_from_E_eV(E_eV))
kf = float(k_from_E_eV(E_final))

print(f"\nTest case: E={E_eV} eV, k_i={ki:.3f}, k_f={kf:.3f} a.u.")

# Distorting potentials
t0 = time.perf_counter()
U_i, U_f = build_distorting_potentials(grid, V, orb_i, orb_f, ki, kf)
t_dist = time.perf_counter() - t0
print(f"Distorting potentials: {t_dist*1000:.1f} ms")

theta = np.linspace(0, np.pi, 50)

# Profile continuum wave solving
print("\n--- Continuum Wave Solving (bottleneck?) ---")
chi_times = []
for l in range(0, 15):
    t0 = time.perf_counter()
    chi = solve_continuum_wave(grid, U_i, l, E_eV, 0.0)
    elapsed = time.perf_counter() - t0
    chi_times.append(elapsed)
    if l < 5 or l == 14:
        print(f"  chi_i[L={l:2d}]: {elapsed*1000:6.1f} ms")

avg_chi = np.mean(chi_times)
print(f"  Average chi solve: {avg_chi*1000:.1f} ms per wave")
print(f"  For L_max=30: {30 * avg_chi:.1f}s for chi_i + {45 * avg_chi:.1f}s for chi_f ≈ {75*avg_chi:.0f}s total")

# Profile radial integrals (GPU)
print("\n--- Radial Integrals GPU ---")
chi_i = solve_continuum_wave(grid, U_i, 5, E_eV, 0.0)
chi_f = solve_continuum_wave(grid, U_f, 5, E_final, 0.0)

integral_times = []
for _ in range(5):
    t0 = time.perf_counter()
    integrals = radial_ME_all_L_gpu(grid, V, U_i.U_of_r, orb_i, orb_f,
                                     chi_i, chi_f, 10, U_f_array=U_f.U_of_r)
    elapsed = time.perf_counter() - t0
    integral_times.append(elapsed)

avg_int = np.mean(integral_times[1:])  # Skip first (warmup)
print(f"  Radial integrals (GPU): {avg_int*1000:.1f} ms per (chi_i, chi_f) pair")

# Profile amplitude calculation (CPU)
print("\n--- Amplitude Calculation (CPU) ---")
amp_times = []
for _ in range(10):
    t0 = time.perf_counter()
    amps = calculate_amplitude_contribution(
        theta, integrals.I_L_direct, integrals.I_L_exchange,
        5, 5, ki, kf, 0, 0, 0, 0
    )
    elapsed = time.perf_counter() - t0
    amp_times.append(elapsed)

avg_amp = np.mean(amp_times)
print(f"  Amplitude calculation: {avg_amp*1000:.2f} ms per (l_i, l_f)")

# Profile DCS calculation
print("\n--- DCS/TCS Calculation ---")
dcs_times = []
for _ in range(10):
    t0 = time.perf_counter()
    dcs = dcs_dwba(theta, amps.f_theta, amps.g_theta, ki, kf, 0, 1)
    sigma = integrate_dcs_over_angles(theta, dcs)
    elapsed = time.perf_counter() - t0
    dcs_times.append(elapsed)

avg_dcs = np.mean(dcs_times)
print(f"  DCS + integrate: {avg_dcs*1000:.2f} ms")

# Summary
print("\n" + "="*70)
print("BOTTLENECK ANALYSIS")
print("="*70)

# Estimate breakdown for one l_i iteration (L_max_proj=30)
n_lf = 15  # average l_f per l_i
chi_cost = avg_chi * (1 + n_lf)  # chi_i + chi_f's
integral_cost = avg_int * n_lf
amp_cost = avg_amp * n_lf
dcs_cost = avg_dcs

total_per_li = chi_cost + integral_cost + amp_cost + dcs_cost

print(f"Per l_i iteration (estimated):")
print(f"  Continuum waves: {chi_cost*1000:7.1f} ms ({100*chi_cost/total_per_li:5.1f}%)")
print(f"  Radial integrals: {integral_cost*1000:7.1f} ms ({100*integral_cost/total_per_li:5.1f}%)")
print(f"  Amplitudes:       {amp_cost*1000:7.1f} ms ({100*amp_cost/total_per_li:5.1f}%)")
print(f"  DCS/TCS:          {dcs_cost*1000:7.1f} ms ({100*dcs_cost/total_per_li:5.1f}%)")
print(f"  ---")
print(f"  TOTAL per l_i:    {total_per_li*1000:7.1f} ms")
print(f"  Est. for L=30:    {30 * total_per_li:.1f} s")
print()
print(f"CONCLUSION: {'Continuum waves' if chi_cost > integral_cost else 'Radial integrals'} is the bottleneck")
print(f"GPU utilization is {'LOW' if chi_cost > integral_cost*2 else 'REASONABLE'} because ODE solving is CPU-bound")
