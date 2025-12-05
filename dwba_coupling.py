# dwba_coupling.py
#
# Angular + spin coupling layer for DWBA amplitudes.
#
# This file converts radial DWBA integrals I_L into angular / spin
# multipole coefficients F_L and G_L so that
#
#   f(θ) = Σ_L F_L P_L(cosθ)
#   g(θ) = Σ_L G_L P_L(cosθ)
#
# and those go into the cross section.
#
# NOTE ABOUT WIGNER SYMBOLS:
# --------------------------
# SciPy >= ~1.11 on CPython <=3.12 provides scipy.special.wigner_3j / wigner_6j.
# On your setup (Python 3.13), SciPy doesn't expose them yet.
# Sympy DOES expose them (sympy.physics.wigner).
#
# We handle both:
#   - try SciPy first (fast, vectorized)
#   - if ImportError, fall back to Sympy
#
# Everything else in the physics stays the same.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable

from sigma_total import DWBAAngularCoeffs

# --- WIGNER SYMBOL BACKEND SELECTION ---------------------------------------

def _build_wigner_backend() -> tuple[Callable, Callable]:
    """
    Return two callables: (w3j, w6j)
    Each callable must accept floats/ints and return a float.
    We attempt SciPy first; if unavailable, fall back to Sympy.
    """

    # Try SciPy first
    try:
        from scipy.special import wigner_3j as _scipy_w3j, wigner_6j as _scipy_w6j  # type: ignore
        def _w3j(l1,l2,l3,m1,m2,m3):
            val = _scipy_w3j(l1,l2,l3,m1,m2,m3)
            return float(val) if np.isfinite(val) else 0.0
        def _w6j(j1,j2,j3,j4,j5,j6):
            val = _scipy_w6j(j1,j2,j3,j4,j5,j6)
            return float(val) if np.isfinite(val) else 0.0
        return _w3j, _w6j
    except Exception:
        pass

    # Fallback: Sympy
    # Sympy returns exact rationals / sqrt expressions. We'll cast to float.
    try:
        from sympy.physics.wigner import wigner_3j as _sympy_w3j, wigner_6j as _sympy_w6j  # type: ignore
        import sympy as sp

        def _w3j(l1,l2,l3,m1,m2,m3):
            val = _sympy_w3j(sp.Rational(l1), sp.Rational(l2), sp.Rational(l3),
                             sp.Rational(m1), sp.Rational(m2), sp.Rational(m3))
            # sympy may return sympy object; cast to float
            try:
                return float(val.evalf())
            except Exception:
                return 0.0

        def _w6j(j1,j2,j3,j4,j5,j6):
            val = _sympy_w6j(sp.Rational(j1), sp.Rational(j2), sp.Rational(j3),
                             sp.Rational(j4), sp.Rational(j5), sp.Rational(j6))
            try:
                return float(val.evalf())
            except Exception:
                return 0.0

        return _w3j, _w6j
    except Exception:
        pass

    # If neither SciPy nor Sympy is available, we cannot continue physically.
    # We return dummy lambdas that raise immediately so the user sees it fast.
    def _nope_3j(*args, **kwargs):
        raise RuntimeError(
            "No Wigner 3j implementation available. Install scipy>=1.11 (with wigner_3j) "
            "or sympy (sympy.physics.wigner)."
        )
    def _nope_6j(*args, **kwargs):
        raise RuntimeError(
            "No Wigner 6j implementation available. Install scipy>=1.11 or sympy."
        )
    return _nope_3j, _nope_6j


_W3J, _W6J = _build_wigner_backend()

# --- DATA CLASSES ----------------------------------------------------------

@dataclass(frozen=True)
class ChannelAngularInfo:
    """
    Quantum numbers describing the excitation channel.

    l_i, l_f : single-electron orbital angular momenta (integers)
    S_i, S_f : total spin of initial / final target term (can be 0, 1/2, 1, ...)
    L_i, L_f : total orbital angular momentum of initial / final term
    J_i, J_f : total J (can be half-integers). If you work purely in LS,
               you can just mirror L_i -> J_i etc. as placeholders.

    Note:
    We store them all, because the full DWBA prefactors in the article
    will generally depend on combinations of these via Wigner 3j/6j and
    (-1)^{...} phases. Even if we don't yet *use* them all in the scaffold,
    they're here so you can plug the final formulas in one place.
    """
    l_i: int
    l_f: int
    S_i: float
    S_f: float
    L_i: float
    L_f: float
    J_i: float
    J_f: float


# --- INTERNAL HELPERS ------------------------------------------------------

def _direct_prefactor(L: int, I_L: float, chan: ChannelAngularInfo) -> complex:
    """
    Proto-direct amplitude coefficient for multipole rank L.
    This enforces basic angular selection rules using a 3j symbol.

    Physics scaffold (NOT final full article formula):
    F_L  ~ (-1)^{l_f} * sqrt[(2 l_f + 1)(2 l_i + 1)]
           * ( l_f   L   l_i
               0     0    0 ) * I_L

    Missing pieces that you'll paste from the article later:
    - spin recoupling (S_i, S_f),
    - coupling from single-electron l_i,l_f to total term L_i,L_f and J_i,J_f,
      which typically introduces Wigner 6j factors,
    - possible additional (-1)^{...} phases and numerical factors.
    """
    li = float(chan.l_i)
    lf = float(chan.l_f)
    Lf = float(L)

    pref_orb = np.sqrt((2.0 * lf + 1.0) * (2.0 * li + 1.0))

    # Wigner-3j(l_f L l_i; 0 0 0)
    three_j = _W3J(lf, Lf, li, 0.0, 0.0, 0.0)

    phase = (-1.0) ** (lf)

    F_L = phase * pref_orb * three_j * I_L
    return complex(F_L)


def _exchange_prefactor(L: int, I_L: float, chan: ChannelAngularInfo) -> complex:
    """
    Proto-exchange amplitude coefficient for multipole rank L.

    Physical idea:
    Exchange differs from direct by antisymmetrization of identical
    electrons and spin coupling. That usually gives different signs /
    6j recoupling and can insert (-1)^(l_i + l_f + L).

    Scaffold:
    G_L ~ (-1)^{l_i + l_f + L}
          * sqrt[(2 l_f + 1)(2 l_i + 1)]
          * ( l_f  L  l_i
              0    0   0 ) * I_L

    Again: final formula from the article can replace this in one shot.
    """
    li = float(chan.l_i)
    lf = float(chan.l_f)
    Lf = float(L)

    pref_orb = np.sqrt((2.0 * lf + 1.0) * (2.0 * li + 1.0))

    three_j = _W3J(lf, Lf, li, 0.0, 0.0, 0.0)

    phase_ex = (-1.0) ** (li + lf + Lf)

    G_L = phase_ex * pref_orb * three_j * I_L
    return complex(G_L)


# --- PUBLIC FUNCTION -------------------------------------------------------

def build_angular_coeffs_for_channel(
    I_L_dict: Dict[int, float],
    chan: ChannelAngularInfo
) -> DWBAAngularCoeffs:
    """
    Map radial DWBA integrals I_L -> angular coefficients F_L (direct),
    G_L (exchange), which define:
        f(θ) = Σ_L F_L P_L(cosθ)
        g(θ) = Σ_L G_L P_L(cosθ)

    This is THE bridge between radial physics (I_L) and
    observable cross section.

    What this does right now:
    - loops over all multipoles L available in I_L_dict,
    - computes a proto-direct coefficient F_L and proto-exchange G_L using
      _direct_prefactor and _exchange_prefactor (which already enforce the
      basic triangle/parity rules via the Wigner 3j symbol),
    - kills tiny numerical noise.

    What you will eventually do (once you paste the article's exact math):
    - modify _direct_prefactor and _exchange_prefactor to include the
      correct Wigner 6j factors, spin recoupling, (-1)^phase, etc.,
      as given in the DWBA formulas for f and g in the paper.

    After that, sigma_total.py will give you dσ/dΩ and σ_total
    with the physically correct weights.

    Parameters
    ----------
    I_L_dict : dict[int, float]
        Radial integrals I_L from dwba_matrix_elements.radial_ME_all_L.
    chan : ChannelAngularInfo
        Quantum numbers of the excitation channel.

    Returns
    -------
    DWBAAngularCoeffs
        {L -> F_L}, {L -> G_L}; ready for sigma_total.f_theta_from_coeffs().
    """
    F_L: Dict[int, complex] = {}
    G_L: Dict[int, complex] = {}

    for L, I_L_val in I_L_dict.items():
        # direct part (f)
        F = _direct_prefactor(L, I_L_val, chan)
        # exchange part (g)
        G = _exchange_prefactor(L, I_L_val, chan)

        # numerical cleanup of near-zero noise
        if abs(F.real) < 1e-18 and abs(F.imag) < 1e-18:
            F = 0.0 + 0.0j
        if abs(G.real) < 1e-18 and abs(G.imag) < 1e-18:
            G = 0.0 + 0.0j

        F_L[L] = complex(F)
        G_L[L] = complex(G)

    return DWBAAngularCoeffs(F_L=F_L, G_L=G_L)
