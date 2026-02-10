#!/usr/bin/env python
"""
Micro-benchmark for GPU ratio policy in radial_ME_all_L_gpu.

Compares `DWBA_GPU_RATIO_POLICY=off/auto/on` on one synthetic DWBA setup
without running a full production scan.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, ".")

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from bound_states import solve_bound_states
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from dwba_matrix_elements import GPUCache, check_cupy_runtime, radial_ME_all_L_gpu
from grid import k_from_E_eV, make_r_grid
from potential_core import CorePotentialParams, V_core_on_grid


@dataclass
class CaseData:
    grid: object
    V: np.ndarray
    U_i: object
    U_f: object
    orb_i: object
    orb_f: object
    chi_i: object
    chi_f: object


def _gpu_snapshot() -> dict[str, float]:
    if cp is None:
        return {}
    free_mem, total_mem = cp.cuda.Device().mem_info
    pool = cp.get_default_memory_pool()
    return {
        "free_gb": free_mem / 1e9,
        "total_gb": total_mem / 1e9,
        "pool_total_gb": pool.total_bytes() / 1e9,
        "pool_used_gb": pool.used_bytes() / 1e9,
    }


def _build_case(n_grid: int, energy_eV: float, l_wave: int) -> CaseData:
    grid = make_r_grid(1e-5, 200.0, n_grid)
    core = CorePotentialParams(Zc=1.0, a1=0.0, a2=1.0, a3=0.0, a4=1.0, a5=0.0, a6=1.0)
    V = V_core_on_grid(grid, core)

    # H-like synthetic case (1s -> 2s) for repeatable timing.
    states = solve_bound_states(grid, V, 0, 2)
    orb_i = states[0]
    orb_f = states[1]

    delta_e_eV = (orb_f.energy_au - orb_i.energy_au) * 27.211386245988
    out_energy_eV = max(energy_eV - delta_e_eV, 5.0)

    k_i = float(k_from_E_eV(energy_eV))
    k_f = float(k_from_E_eV(out_energy_eV))
    U_i, U_f = build_distorting_potentials(grid, V, orb_i, orb_f, k_i, k_f)

    chi_i = solve_continuum_wave(grid, U_i, l_wave, energy_eV, 0.0)
    chi_f = solve_continuum_wave(grid, U_f, l_wave, out_energy_eV, 0.0)

    return CaseData(
        grid=grid,
        V=V,
        U_i=U_i,
        U_f=U_f,
        orb_i=orb_i,
        orb_f=orb_f,
        chi_i=chi_i,
        chi_f=chi_f,
    )


def _run_once(case: CaseData, L_max: int, gpu_memory_mode: str, gpu_memory_threshold: float) -> None:
    cache = GPUCache.from_grid(case.grid, max_chi_cached=12)
    try:
        _ = radial_ME_all_L_gpu(
            case.grid,
            case.V,
            case.U_i.U_of_r,
            case.orb_i,
            case.orb_f,
            case.chi_i,
            case.chi_f,
            L_max=L_max,
            use_oscillatory_quadrature=True,
            oscillatory_method="advanced",
            gpu_memory_mode=gpu_memory_mode,
            gpu_memory_threshold=gpu_memory_threshold,
            gpu_block_size=0,
            gpu_cache=cache,
            U_f_array=case.U_f.U_of_r,
        )
        if cp is not None:
            cp.cuda.Stream.null.synchronize()
    finally:
        cache.clear()


def _benchmark_policy(
    case: CaseData,
    policy: str,
    L_max: int,
    repeats: int,
    gpu_memory_mode: str,
    gpu_memory_threshold: float,
) -> dict[str, float]:
    os.environ["DWBA_GPU_RATIO_POLICY"] = policy

    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    _run_once(case, min(4, L_max), gpu_memory_mode, gpu_memory_threshold)  # warmup

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_once(case, L_max, gpu_memory_mode, gpu_memory_threshold)
        times.append(time.perf_counter() - t0)

    snap = _gpu_snapshot()
    return {
        "mean_s": float(np.mean(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "free_gb": float(snap.get("free_gb", 0.0)),
        "pool_total_gb": float(snap.get("pool_total_gb", 0.0)),
        "pool_used_gb": float(snap.get("pool_used_gb", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DWBA GPU ratio policy.")
    parser.add_argument("--n-grid", type=int, default=3200, help="Radial grid points (default: 3200)")
    parser.add_argument("--energy-ev", type=float, default=250.0, help="Incident energy in eV")
    parser.add_argument("--l-wave", type=int, default=6, help="Continuum partial wave l for benchmark")
    parser.add_argument("--L-max", type=int, default=12, help="Multipole cutoff for each benchmark run")
    parser.add_argument("--repeats", type=int, default=2, help="Measured repeats per policy")
    parser.add_argument(
        "--policies",
        type=str,
        default="off,auto,on",
        help="Comma-separated DWBA_GPU_RATIO_POLICY values",
    )
    parser.add_argument(
        "--gpu-memory-mode",
        type=str,
        default="auto",
        choices=("auto", "full", "block"),
        help="radial_ME_all_L_gpu memory mode",
    )
    parser.add_argument(
        "--gpu-memory-threshold",
        type=float,
        default=0.8,
        help="GPU memory threshold for auto mode",
    )
    args = parser.parse_args()

    if not check_cupy_runtime():
        print("CuPy runtime is unavailable. Benchmark skipped.")
        return 0

    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    valid = {"auto", "on", "off"}
    bad = [p for p in policies if p not in valid]
    if bad:
        print(f"Invalid policy values: {bad}. Allowed: auto,on,off")
        return 2

    print("Preparing synthetic benchmark case...")
    case = _build_case(args.n_grid, args.energy_ev, args.l_wave)
    print(
        "Case ready | "
        f"n_grid={args.n_grid}, E={args.energy_ev:.1f} eV, l={args.l_wave}, "
        f"L_max={args.L_max}, repeats={args.repeats}"
    )
    print(f"Initial GPU snapshot: {_gpu_snapshot()}")
    print()

    results = {}
    for policy in policies:
        print(f"[policy={policy}] running...")
        try:
            results[policy] = _benchmark_policy(
                case=case,
                policy=policy,
                L_max=args.L_max,
                repeats=args.repeats,
                gpu_memory_mode=args.gpu_memory_mode,
                gpu_memory_threshold=args.gpu_memory_threshold,
            )
        except Exception as exc:
            print(f"[policy={policy}] failed: {exc}")
            results[policy] = {"mean_s": float("nan"), "min_s": float("nan"), "max_s": float("nan")}

    print("\nResults:")
    print("policy | mean_s | min_s | max_s | free_gb | pool_total_gb | pool_used_gb")
    print("-" * 78)
    for policy in policies:
        r = results.get(policy, {})
        print(
            f"{policy:>6} | "
            f"{r.get('mean_s', float('nan')):6.3f} | "
            f"{r.get('min_s', float('nan')):5.3f} | "
            f"{r.get('max_s', float('nan')):5.3f} | "
            f"{r.get('free_gb', float('nan')):7.3f} | "
            f"{r.get('pool_total_gb', float('nan')):13.3f} | "
            f"{r.get('pool_used_gb', float('nan')):12.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
