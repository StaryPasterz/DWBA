#!/usr/bin/env python
"""
Lightweight CPU/GPU parity check for radial DWBA integrals.

The script is intentionally small:
- uses one H 1s->2s case,
- one (l_i, l_f) pair,
- moderate grid and L_max,
- exits with code 0 on SKIP when GPU runtime is unavailable.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Allow running as `python debug/test_cpu_gpu_parity.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atom_library import get_atom
from continuum import solve_continuum_wave
from distorting_potential import build_distorting_potentials
from driver import ExcitationChannelSpec, prepare_target
from dwba_matrix_elements import (
    HAS_CUPY,
    check_cupy_runtime,
    radial_ME_all_L,
    radial_ME_all_L_gpu,
)


def _max_rel_err(values_cpu: dict[int, float], values_gpu: dict[int, float]) -> tuple[float, int]:
    max_rel = 0.0
    worst_l = -1
    for L in sorted(values_cpu.keys()):
        a = float(values_cpu[L])
        b = float(values_gpu[L])
        den = max(abs(a), abs(b), 1e-12)
        rel = abs(a - b) / den
        if rel > max_rel:
            max_rel = rel
            worst_l = L
    return max_rel, worst_l


def main() -> int:
    if not HAS_CUPY or not check_cupy_runtime():
        print("SKIP: CuPy/GPU runtime unavailable.")
        return 0

    atom = get_atom("H")
    spec = ExcitationChannelSpec(
        l_i=0,
        l_f=0,
        n_index_i=1,
        n_index_f=2,
        N_equiv=1,
        L_max_integrals=6,
        L_target_i=0,
        L_target_f=0,
        L_max_projectile=4,
    )

    prep = prepare_target(
        spec,
        atom.core_params,
        r_max=160.0,
        n_points=900,
        use_polarization=False,
    )

    E_inc = 80.0
    E_fin = E_inc - prep.dE_target_eV
    if E_fin <= 0.0:
        print("FAIL: closed final channel in synthetic parity setup.")
        return 1

    k_i = np.sqrt(2.0 * (E_inc / 27.211386245988))
    k_f = np.sqrt(2.0 * (E_fin / 27.211386245988))
    z_ion = atom.core_params.Zc - 1.0

    U_i, U_f = build_distorting_potentials(
        prep.grid,
        prep.V_core,
        prep.orb_i,
        prep.orb_f,
        k_i_au=k_i,
        k_f_au=k_f,
        use_polarization=False,
    )

    cont_i = solve_continuum_wave(prep.grid, U_i, l=0, E_eV=E_inc, z_ion=z_ion)
    cont_f = solve_continuum_wave(prep.grid, U_f, l=0, E_eV=E_fin, z_ion=z_ion)
    if cont_i is None or cont_f is None:
        print("FAIL: continuum solve failed in parity setup.")
        return 1

    kwargs = dict(
        grid=prep.grid,
        V_core_array=prep.V_core,
        U_i_array=U_i.U_of_r,
        bound_i=prep.orb_i,
        bound_f=prep.orb_f,
        cont_i=cont_i,
        cont_f=cont_f,
        L_max=spec.L_max_integrals,
        use_oscillatory_quadrature=True,
        oscillatory_method="advanced",
        CC_nodes=5,
        phase_increment=np.pi / 2,
        min_grid_fraction=0.1,
        k_threshold=0.5,
        U_f_array=U_f.U_of_r,
    )

    cpu = radial_ME_all_L(**kwargs)
    gpu = radial_ME_all_L_gpu(
        **kwargs,
        gpu_block_size="auto",
        gpu_memory_mode="auto",
        gpu_memory_threshold=0.8,
        gpu_cache=None,
    )

    rel_dir, worst_dir = _max_rel_err(cpu.I_L_direct, gpu.I_L_direct)
    rel_exc, worst_exc = _max_rel_err(cpu.I_L_exchange, gpu.I_L_exchange)
    tol = 5e-2

    print(f"CPU/GPU parity (direct):   max_rel={rel_dir:.3e} at L={worst_dir}")
    print(f"CPU/GPU parity (exchange): max_rel={rel_exc:.3e} at L={worst_exc}")
    if rel_dir > tol or rel_exc > tol:
        print(f"FAIL: parity tolerance exceeded (tol={tol:.3e})")
        return 1

    print(f"PASS: parity within tolerance (tol={tol:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
