# potential_core.py
#
# Effective single-active-electron (SAE) core potential V_A+(r)
# as used in the DWBA formulation in the article.
#
# This potential is intended to represent the "ion core" felt by the
# active electron in the target. It is central (depends only on r)
# and is tuned via parameters (a1..a6) so that the bound-state spectrum
# reproduces known binding energies / excitation thresholds.
#
# Form used (Tong/Lin-style screening form, consistent with the paper):
#
#   V_A+(r) = -1/r + [ a1*exp(-a2*r) + a3*r*exp(-a4*r) + a5*exp(-a6*r) ] / r
#
# Notes:
# - r in atomic units (bohr)
# - V in atomic units (hartree)
# - As r -> infinity, V_A+(r) ~ -1/r (correct for singly-charged target)
# - As r -> 0, V_A+(r) ~ -(1 - a1 - ...)/r + finite terms
#
# These parameters are target-dependent and must be provided as input
# for each ion / atom you want to model.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from grid import RadialGrid
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)


@dataclass(frozen=True)
class CorePotentialParams:
    """
    Parameters defining the SAE effective core potential V_A+(r).

    The functional form is:
        V(r) = -1/r + [a1*exp(-a2*r) + a3*r*exp(-a4*r) + a5*exp(-a6*r)] / r

    These parameters (a1..a6) encode short-range screening behavior
    and effective nuclear charge structure. They are chosen so
    that the resulting bound-state energies of the single-active-electron
    Hamiltonian match known level energies of the ion.

    Physically:
    - a1,a2     : very short-range screening near the nucleus
    - a3,a4     : intermediate-range correction ~ r * exp(-a4 r)
    - a5,a6     : additional exponential tail correction

    Different targets (H, He+, Ne+, ...) will have different sets
    of these parameters.

    Attributes
    ----------
    a1, a2, a3, a4, a5, a6 : float
        Screening parameters in atomic units.
        Typically a2,a4,a6 > 0 (decay rates); magnitudes of a1,a3,a5
        control the strength of each screening term.

    # Zeff_large should be approx Zc (e.g. ~2 for Ne+),
    # not ~1. If this is not the case, parameters are incorrect.

    """
    Zc: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float


def _safe_inv_r(r: np.ndarray, r_floor: float = 1e-12) -> np.ndarray:
    """
    Return 1/r but avoid an actual division by zero if r contains 0.0.

    In our pipeline we generate r grids with r_min > 0, so normally
    we shouldn't hit r=0. This is just a guardrail in case someone
    later builds a grid starting exactly at r=0.

    Parameters
    ----------
    r : np.ndarray
        Radii in bohr.
    r_floor : float
        Minimum radius allowed in denominator. Values of r smaller than
        r_floor are internally clamped to r_floor.

    Returns
    -------
    inv_r : np.ndarray
        1 / max(r, r_floor), elementwise.
    """
    r_eff = np.maximum(r, r_floor)
    return 1.0 / r_eff


def V_core_on_grid(grid: RadialGrid, params: CorePotentialParams) -> np.ndarray:
    """
    Evaluate the SAE core potential V_A+(r) on a given radial grid.

    The functional form matches the model used in the article:
        V_A+(r) = -1/r
                  + [ a1*exp(-a2*r)
                      + a3*r*exp(-a4*r)
                      + a5*exp(-a6*r) ] / r

    All quantities are in atomic units:
    - r in bohr
    - V in hartree

    Parameters
    ----------
    grid : RadialGrid
        Radial grid (see grid.make_r_grid). We assume r>0.
    params : CorePotentialParams
        Parameters (a1..a6) defining the short-range screening.

    Returns
    -------
    V : np.ndarray, shape (N,)
        Core potential values at each r point in the grid.
        Units: hartree.

    Raises
    ------
    ValueError
        If any numerical issue (NaN/inf) appears.
    """
    r = grid.r  # bohr
    inv_r = _safe_inv_r(r)

    # Build screening numerator S(r):
    # S(r) = a1*exp(-a2*r) + a3*r*exp(-a4*r) + a5*exp(-a6*r)
    # All exponentials are dimensionless in a.u.
    S = (
        params.a1 * np.exp(-params.a2 * r)
        + params.a3 * r * np.exp(-params.a4 * r)
        + params.a5 * np.exp(-params.a6 * r)
    )

    V = -(params.Zc + S) * inv_r

    # Sanity checks
    if not np.all(np.isfinite(V)):
        raise ValueError("V_core_on_grid produced non-finite values (inf or NaN). "
                         "Check radial grid or parameters.")

    # Physical sanity:
    # - at large r, expect V ~ -1/r -> goes to 0- from below
    # - at very small r, magnitude should blow up ~ -Z_eff(0)/r
    #   where Z_eff(0) = 1 - a1 - ... (depends on params)
    # We'll not enforce this here, but you can check with inspect_core_potential().

    return V


def effective_charge_profile(r: np.ndarray, V_core: np.ndarray) -> np.ndarray:
    """
    Compute the effective charge profile Z_eff(r) defined by

        Z_eff(r) = - r * V_core(r)

    Interpretation:
    - If V_core(r) were exactly a pure Coulomb potential -Z/r,
      then Z_eff(r) would be identically Z.
    - Here, Z_eff(r) generally depends on r: near the nucleus it
      should approach something close to the bare nuclear charge,
      and for large r it should approach 1 (for a singly-charged ion),
      matching the asymptotic -1/r tail in the article.

    This is extremely useful for debugging / fitting params:
    - You can quickly see if you're really modeling (say) Ne⁺
      or something unphysical.

    Parameters
    ----------
    r : np.ndarray
        Radii in bohr.
    V_core : np.ndarray
        Potential values in hartree, same shape as r.

    Returns
    -------
    Z_eff : np.ndarray
        Dimensionless effective charge profile.
    """
    if r.shape != V_core.shape:
        raise ValueError("effective_charge_profile: r and V_core must match shape.")
    return -r * V_core


def inspect_core_potential(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    n_preview: int = 5
) -> Tuple[float, float, float, float]:
    """
    Quick diagnostic helper.

    Gives you a feeling for whether your chosen CorePotentialParams
    look physical for a given target:
    - Check short-range behavior (first few points),
    - Check long-range tail (last few points),
    - Estimate effective charge in the inner and outer region.

    This is not used in production calculations, but it's handy when
    tuning a1..a6 for a new ion.

    Parameters
    ----------
    grid : RadialGrid
        The radial grid used.
    V_core_array : np.ndarray
        Output of V_core_on_grid(grid, params).
    n_preview : int
        How many points from each end to sample for summary.

    Returns
    -------
    (V_small, Zeff_small, V_large, Zeff_large) : tuple of floats
        Representative values:
        - V_small:  potential at the smallest radius r[0]
        - Zeff_small: Z_eff at r[0]
        - V_large:  potential at the largest radius r[-1]
        - Zeff_large: Z_eff at r[-1]

    Notes
    -----
    Typical expectations:
    - Zeff_large ~ 1  (so V ~ -1/r asymptotically).
    - Zeff_small should be close to the actual nuclear charge (e.g. ~10 for Ne),
      because screening vanishes near the nucleus.
      If this is completely off, parameters a1..a6 are incorrect.
    """
    r = grid.r
    if V_core_array.shape != r.shape:
        raise ValueError("inspect_core_potential: grid/V_core mismatch.")

    # Clip n_preview to avoid issues on tiny grids
    n_preview = max(1, min(n_preview, r.size))

    r_small = r[0]
    r_large = r[-1]

    V_small = float(V_core_array[0])
    V_large = float(V_core_array[-1])

    Zeff = effective_charge_profile(r, V_core_array)
    Zeff_small = float(Zeff[0])
    Zeff_large = float(Zeff[-1])

    return V_small, Zeff_small, V_large, Zeff_large
