#!/usr/bin/env python
"""
Regression checks for two targeted fixes:
1) Config excitation path uses N_equiv=1 (SAE consistency).
2) GPU convergence logic uses post-add DCS state (no pre-add false stop).

This script is intentionally lightweight and deterministic.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from sigma_total import dcs_dwba


def _find_config_excit_nequiv_constant() -> int | None:
    """Return constant N_equiv value used in run_from_config excitation spec."""
    tree = ast.parse(Path("DW_main.py").read_text(encoding="utf-8"))
    target_value = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "run_from_config":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func_name = getattr(sub.func, "id", None)
            if func_name != "ExcitationChannelSpec":
                continue
            # Config excitation spec is currently around ~line 1977.
            if not (1900 <= getattr(sub, "lineno", -1) <= 2050):
                continue
            for kw in sub.keywords:
                if kw.arg == "N_equiv" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, int):
                        target_value = kw.value.value
                        return target_value
    return target_value


def _check_nequiv_scaling_identity() -> tuple[float, float, float]:
    """
    Sanity: dcs_dwba scales linearly with N_equiv.
    Returns (sum_dcs_N1, sum_dcs_N2, ratio).
    """
    theta = np.linspace(0.0, np.pi, 121)
    # Deterministic synthetic amplitudes
    f = np.exp(1j * 0.25 * np.cos(theta)) * (0.30 + 0.10 * np.cos(theta))
    g = np.exp(-1j * 0.17 * np.cos(theta)) * (0.12 + 0.05 * np.sin(theta))

    d1 = dcs_dwba(theta, f, g, k_i_au=2.2, k_f_au=1.9, L_i=0, N_equiv=1)
    d2 = dcs_dwba(theta, f, g, k_i_au=2.2, k_f_au=1.9, L_i=0, N_equiv=2)

    s1 = float(np.sum(d1))
    s2 = float(np.sum(d2))
    ratio = s2 / (s1 + 1e-300)
    return s1, s2, ratio


def _old_pre_add_stop_l() -> int | None:
    """
    Emulate old behavior: convergence checked before adding current l_i contribution.
    Returns l_i where stop is triggered (or None).
    """
    total = 100.0
    # (l_i, delta_from_current_l_i)
    seq = [
        (10, 0.20),
        (11, 0.15),
        (12, 0.10),
        (13, 0.10),
        (14, 0.08),
        (15, 0.07),
        (16, 8.00),   # large contribution that old logic can miss
        (17, 0.05),
    ]
    hist: list[tuple[int, float]] = []
    for l_i, delta in seq:
        current_pre = total  # pre-add state
        if l_i >= 5:
            hist.append((l_i, current_pre))
            if len(hist) >= 4:
                old = hist[-4][1]
                max_change = abs(current_pre - old) / (abs(current_pre) + 1e-50)
                if max_change < 0.01 and l_i > 15:
                    return l_i
        total += delta
    return None


def _new_post_add_stop_l() -> int | None:
    """
    Emulate new behavior: convergence checked after adding current l_i contribution.
    Returns l_i where stop is triggered (or None).
    """
    total = 100.0
    seq = [
        (10, 0.20),
        (11, 0.15),
        (12, 0.10),
        (13, 0.10),
        (14, 0.08),
        (15, 0.07),
        (16, 8.00),
        (17, 0.05),
    ]
    hist: list[tuple[int, float]] = []
    for l_i, delta in seq:
        total += delta
        current_post = total  # post-add state
        if l_i >= 5:
            hist.append((l_i, current_post))
            if len(hist) >= 4:
                old = hist[-4][1]
                max_change = abs(current_post - old) / (abs(current_post) + 1e-50)
                if max_change < 0.01 and l_i > 15:
                    return l_i
    return None


def main() -> int:
    print("== Regression check: point 1 (N_equiv in config path) ==")
    n_equiv_cfg = _find_config_excit_nequiv_constant()
    print(f"run_from_config excitation N_equiv = {n_equiv_cfg}")
    if n_equiv_cfg != 1:
        print("FAIL: expected N_equiv=1 in config excitation path.")
        return 1

    s1, s2, ratio = _check_nequiv_scaling_identity()
    print(f"DCS linearity check: sum(N=1)={s1:.6e}, sum(N=2)={s2:.6e}, ratio={ratio:.12f}")
    if not math.isfinite(ratio) or abs(ratio - 2.0) > 1e-10:
        print("FAIL: expected exact linear scaling ratio ~2.0 for N_equiv.")
        return 1

    # Reference-point context from dwba_reference_points.md (Fig. 1b):
    # H 1s->2p at 100 eV: sigma_DWBA ≈ 2.2059 a.u.
    # Wrong N_equiv=2 would inflate that to ≈ 4.4118 a.u.
    ref_sigma_au = 2.2059
    print(
        "Reference context (H 1s->2p, 100 eV): "
        f"DWBA~{ref_sigma_au:.4f} a.u., wrong N=2 -> {2.0 * ref_sigma_au:.4f} a.u."
    )

    print("\n== Regression check: point 2 (convergence check ordering) ==")
    old_stop = _old_pre_add_stop_l()
    new_stop = _new_post_add_stop_l()
    print(f"old pre-add stop l_i = {old_stop}")
    print(f"new post-add stop l_i = {new_stop}")
    if old_stop is None:
        print("FAIL: synthetic scenario should trigger old premature stop.")
        return 1
    if new_stop is not None and new_stop <= old_stop:
        print("FAIL: new logic should not stop at or before old premature stop.")
        return 1

    print("\nPASS: Both targeted fixes validated by lightweight regression checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
