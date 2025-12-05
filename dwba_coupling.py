# dwba_coupling.py
#
# Angular + spin coupling layer for DWBA amplitudes.
# Implements Eq. (29) and (30) from the reference article:
#   Z.-J. Lai et al., J. At. Mol. Sci. 5 (2014) 311-323.
#
# Key components:
#   - Wigner 3j / 6j / Racah coefficients.
#   - Geometric factors connecting radial integrals I_L to 
#     scattering amplitudes f(theta) and g(theta).

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple
from dataclasses import dataclass

# --- WIGNER SYMBOL BACKEND SELECTION ---------------------------------------

def _build_wigner_backend() -> tuple[Callable, Callable]:
    """
    Return two callables: (w3j, w6j)
    Each callable must accept floats/ints and return a float.
    We attempt SciPy first; if unavailable, fall back to Sympy.
    """
    # Try SciPy 
    try:
        from scipy.special import wigner_3j as _scipy_w3j, wigner_6j as _scipy_w6j
        # Simple test to see if it works (some older scipy versions might miss it)
        _scipy_w3j(0,0,0,0,0,0)
        
        def _w3j(l1,l2,l3,m1,m2,m3):
            val = _scipy_w3j(l1,l2,l3,m1,m2,m3)
            # scipy might return array for scalar input
            return float(val) if np.isfinite(val) else 0.0
            
        def _w6j(j1,j2,j3,j4,j5,j6):
            val = _scipy_w6j(j1,j2,j3,j4,j5,j6)
            return float(val) if np.isfinite(val) else 0.0
            
        return _w3j, _w6j
    except Exception:
        pass

    # Fallback: Sympy
    try:
        from sympy.physics.wigner import wigner_3j as _sympy_w3j, wigner_6j as _sympy_w6j
        import sympy as sp

        def _w3j(l1,l2,l3,m1,m2,m3):
            val = _sympy_w3j(sp.Rational(l1), sp.Rational(l2), sp.Rational(l3),
                             sp.Rational(m1), sp.Rational(m2), sp.Rational(m3))
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

    def _nope_3j(*args, **kwargs):
        raise RuntimeError("No Wigner 3j implementation available. Install scipy or sympy.")
    def _nope_6j(*args, **kwargs):
        raise RuntimeError("No Wigner 6j implementation available. Install scipy or sympy.")
    return _nope_3j, _nope_6j


_W3J, _W6J = _build_wigner_backend()

# --- MATH HELPERS ----------------------------------------------------------

def tilde(l: float) -> float:
    """Returns sqrt(2l + 1). symbol: tilde{l}"""
    return np.sqrt(2.0 * l + 1.0)

def clebsch_gordan(j1, j2, j3, m1, m2, m3) -> float:
    """
    Computes Clebsch-Gordan coefficient <j1 m1 j2 m2 | j3 m3>.
    Relation to 3j symbol:
      <j1 m1 j2 m2 | j3 m3> = (-1)^(j1-j2+m3) * sqrt(2*j3+1) * (j1 j2 j3)
                                                               (m1 m2 -m3)
    """
    # selection rule for m:
    if abs(m1 + m2 - m3) > 1e-9:
        return 0.0
    
    pre = tilde(j3)
    phase = (-1.0)**(j1 - j2 + m3)
    w3 = _W3J(j1, j2, j3, m1, m2, -m3)
    return phase * pre * w3

def racah_w(l1, l2, l3, l4, l5, l6) -> float:
    """
    Computes Racah coefficient W(l1 l2 l3 l4; l5 l6).
    Relation to 6j symbol:
      W(a b c d; e f) = (-1)^(a+b+c+d) * {a b e}
                                         {d c f}
    """
    phase = (-1.0)**(l1 + l2 + l3 + l4)
    six_j = _W6J(l1, l2, l5, l4, l3, l6)
    return phase * six_j


# --- MAIN COUPLER ----------------------------------------------------------

def compute_direct_geometry(
    li: int, lf: int, Li: int, Lf: int, lT: int,
    mu_i: int, mu_f: int, Mi: int, Mf: int
) -> float:
    """
    Computes the geometric part of the Direct amplitude (Eq. 29).
    This factor multiplies:
       (1/(kf*ki)) * I_{lT} * Y_{lf -muf}(kf) * Y_{li mui}^*(ki) * (2/pi if needed by formula)
    
    Returns the scalar weight inside the sum over g.
    """
    # Cast to float for math
    fli, flf = float(li), float(lf)
    fLi, fLf = float(Li), float(Lf)
    flT = float(lT)
    fmui, fmuf = float(mu_i), float(mu_f)
    fMi, fMf = float(Mi), float(Mf)

    # Pre-checks for basic selection rules
    # 1. Triangle(lf, li, lT) for C(lf li lT; 000)
    if not (abs(fli - flf) <= flT <= fli + flf): return 0.0
    # 2. Triangle(lT, Li, Lf) for C(lT Li Lf; 000)
    if not (abs(flT - fLi) <= fLf <= flT + fLi): return 0.0
    # 3. Parity conservation: li+lf+lT must be even
    if (li + lf + lT) % 2 != 0: return 0.0
    
    pref = (tilde(flf) * tilde(fli) * tilde(fLi)) / tilde(flT)
    
    # Paper Eq 29 says: C(lf li lT; 0 0 0)
    c_orb = clebsch_gordan(flf, fli, flT, 0.0, 0.0, 0.0)
    c_target = clebsch_gordan(flT, fLi, fLf, 0.0, 0.0, 0.0) # C(lT Li Lf; 000)
    
    if abs(c_orb) < 1e-12 or abs(c_target) < 1e-12:
        return 0.0

    # Sum over g
    g_min = max(abs(fli - fLi), abs(flf - fLf))
    g_max = min(fli + fLi, flf + fLf)
    
    sum_g = 0.0
    start_g = int(np.ceil(g_min))
    end_g = int(np.floor(g_max))
    
    has_term = False
    for g_int in range(start_g, end_g + 1):
        fg = float(g_int)
        
        # C(li Li g; mui Mi mui+Mi)
        c1 = clebsch_gordan(fli, fLi, fg, fmui, fMi, fmui + fMi)
        if abs(c1) < 1e-12: continue

        # C(lf Lf g; muf -Mf muf-Mf)
        # Delta constraint: mui+Mi = Mf-muf (Eq 29)
        # This implies we only sum if Mf - muf == mui + Mi
        # If input args violate this, result should be 0, but usually caller handles loops.
        # We enforce it via Clebsch selection rule on projection M_g.
        
        M_g = fmui + fMi
        # Check consistency with second Clebsch
        # C(lf Lf g; muf, -Mf, muf-Mf)
        # The equation puts C(lf Lf g; muf, -Mf, muf-Mf).
        # Normal Clebsch requires sum of projections: muf - Mf = M_g.
        # So M_g must be muf-Mf.
        # Thus (mui+Mi) must equal (muf-Mf) -> mu_i + mu_f = M_f - M_i?
        # Eq 29 delta says: delta_{mu_i + M_i, M_f - mu_f}
        # Wait, if delta is { A, B } it means A=B.
        # So mui + Mi = Mf - muf.
        # => muf = Mf - Mi - mui.
        # We assume caller provides valid muf, or we check it.
        if abs((fmuf - fMf) - (fmui+fMi)) > 1e-9:
             # Wait, Eq 29 second Clebsch uses muf-Mf as third component.
             # Wait, standard Clebsch is <j1 m1 j2 m2 | J M>. So m1+m2=M.
             # C(lf Lf g; muf, -Mf, ?) -> ? = muf - Mf.
             # So the equation is consistent with Clebsch def.
             pass
             
        c2 = clebsch_gordan(flf, fLf, fg, fmuf, -fMf, M_g)
        if abs(c2) < 1e-12: continue

        w_val = racah_w(flf, fli, fLf, fLi, flT, fg)
        
        sum_g += c1 * c2 * w_val
        has_term = True

    if not has_term:
        return 0.0

    return pref * c_orb * c_target * sum_g


def compute_exchange_geometry(
    li: int, lf: int, Li: int, Lf: int, lT: int,
    mu_i: int, mu_f: int, Mi: int, Mf: int
) -> float:
    """
    Computes the geometric part of the Exchange amplitude (Eq. 30).
    """
    fli, flf = float(li), float(lf)
    fLi, fLf = float(Li), float(Lf)
    flT = float(lT)
    fmui, fmuf = float(mu_i), float(mu_f)
    fMi, fMf = float(Mi), float(Mf)

    if (Lf + li + lT) % 2 != 0: return 0.0
    if (lT + Li + lf) % 2 != 0: return 0.0
    
    pref = (tilde(fLf) * tilde(fli) * tilde(fLi)) / tilde(flT)
    phase_out = (-1.0)**(Lf + Mf)
    
    c_orb = clebsch_gordan(fLf, fli, flT, 0.0, 0.0, 0.0)
    c_target = clebsch_gordan(flT, fLi, flf, 0.0, 0.0, 0.0)
    
    if abs(c_orb) < 1e-12 or abs(c_target) < 1e-12:
        return 0.0

    g_min = max(abs(fli - fLi), abs(fLf - flf))
    g_max = min(fli + fLi, fLf + flf)
    
    sum_g = 0.0
    start_g = int(np.ceil(g_min))
    end_g = int(np.floor(g_max))
    
    has_term = False
    for g_int in range(start_g, end_g + 1):
        fg = float(g_int)
        
        # C(li Li g; mui Mi mui+Mi)
        M_g = fmui + fMi
        c1 = clebsch_gordan(fli, fLi, fg, fmui, fMi, M_g)
        if abs(c1) < 1e-12: continue

        # Eq 30 delta: delta_{mui+Mi, Mf+muf}
        # => mui+Mi = Mf+muf.
        # Clebsch: C(Lf lf g; -Mf -muf -Mf-muf)
        # Third comp is - (Mf+muf) = - (mui+Mi).
        # M_g in c1 was mui+Mi.
        # BUT wait. In sum over g, we are coupling angular momenta magnitude,
        # but the projections must match.
        # Eq 25 decomposion implies M of g is conserved.
        # But here C2 has projection -(Mf+muf) which is -M_g.
        # Does the sum require M_g or -M_g?
        # The expansion of coefficients usually matches M.
        # However, Eq 30 explicitly writes C(..., -Mf-muf).
        # We assume the formula is correct as written.
        
        c2 = clebsch_gordan(fLf, flf, fg, -fMf, -fmuf, -fMf - fmuf)
        if abs(c2) < 1e-12: continue
        
        w_val = racah_w(fLf, fli, flf, fLi, flT, fg)
        
        sum_g += c1 * c2 * w_val
        has_term = True
        
    if not has_term:
        return 0.0
        
    return phase_out * pref * c_orb * c_target * sum_g
