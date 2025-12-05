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
# Implementation uses built-in numeric Wigner 3j/6j symbols (Racah formula)
# to avoid external dependencies issues.
#

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

from sigma_total import DWBAAngularCoeffs

# ---- Wigner Symbols Implementation (Racah formulas) ----

def _log_factorial(n):
    """Logarithm of factorial n! for numerical stability."""
    if n < 0: return -1.0 
    if n <= 1: return 0.0
    return np.sum(np.log(np.arange(2, n + 1, dtype=float)))

def _delta_tri(a, b, c):
    """Triangle coefficient Delta(a,b,c) for Racah formula log-space."""
    return _log_factorial(a + b - c) + _log_factorial(a - b + c) + \
           _log_factorial(-a + b + c) - _log_factorial(a + b + c + 1)

def wigner_3j_num(j1, j2, j3, m1, m2, m3):
    """
    Numerical Wigner 3j symbol (j1 j2 j3 / m1 m2 m3).
    Using Racah formula implementation.
    """
    if abs(m1 + m2 + m3) > 1e-9:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    if (j1 + j2 + j3) % 1.0 != 0.0:
         return 0.0 
    
    t_min = max(0, int(j2 - m1 - j3), int(j1 + m2 - j3))
    t_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))
    
    if t_min > t_max:
        return 0.0
        
    log_pref = 0.5 * _delta_tri(j1, j2, j3)
    log_pref += 0.5 * (_log_factorial(j1+m1) + _log_factorial(j1-m1) +
                       _log_factorial(j2+m2) + _log_factorial(j2-m2) +
                       _log_factorial(j3+m3) + _log_factorial(j3-m3))
                       
    sum_val = 0.0
    for t in range(t_min, t_max + 1):
        denom = _log_factorial(t) + _log_factorial(t - int(j2 - m1 - j3)) + \
                _log_factorial(t - int(j1 + m2 - j3)) + \
                _log_factorial(int(j1 + j2 - j3) - t) + \
                _log_factorial(int(j1 - m1) - t) + \
                _log_factorial(int(j2 + m2) - t)
        
        term = (-1)**t * np.exp(log_pref - denom)
        sum_val += term
        
    sign = (-1.0)**int(j1 - j2 - m3)
    return sign * sum_val

def wigner_6j_num(j1, j2, j3, j4, j5, j6):
    """
    Numerical Wigner 6j symbol {j1 j2 j3 / j4 j5 j6}.
    """
    if not (abs(j1-j2)<=j3<=j1+j2 and abs(j1-j5)<=j6<=j1+j5 and
            abs(j4-j2)<=j6<=j4+j2 and abs(j4-j5)<=j3<=j4+j5):
        return 0.0
        
    t_min = max(int(j1+j2+j3), int(j1+j5+j6), int(j4+j2+j6), int(j4+j5+j3))
    t_max = min(int(j1+j2+j4+j5), int(j2+j3+j5+j6), int(j3+j1+j6+j4))
    
    if t_min > t_max:
        return 0.0
        
    log_tri = _delta_tri(j1, j2, j3) + _delta_tri(j1, j5, j6) + \
              _delta_tri(j4, j2, j6) + _delta_tri(j4, j5, j3)
    
    sum_val = 0.0
    for t in range(t_min, t_max + 1):
        denom = _log_factorial(t - int(j1+j2+j3)) + \
                _log_factorial(t - int(j1+j5+j6)) + \
                _log_factorial(t - int(j4+j2+j6)) + \
                _log_factorial(t - int(j4+j5+j3)) + \
                _log_factorial(int(j1+j2+j4+j5) - t) + \
                _log_factorial(int(j2+j3+j5+j6) - t) + \
                _log_factorial(int(j3+j1+j6+j4) - t)
        
        num = _log_factorial(t+1)
        term = (-1)**t * np.exp(num - denom) # Note: 0.5*log_tri outside sum? No, usually outside.
        # Check standard formula carefully: 
        # { ... } = Delta(...) * Delta(...) * Delta(...) * Delta(...) * Sum ...
        # My formula: exp(log_tri) outside?
        # In the loop I did exp(0.5*log_tri ...). No.
        # Wait, the log_tri calculation is sum of log_factorials.
        # Correct Racah formula:
        # {6j} = sqrt(Delta(123)Delta(156)...) * Sum (-1)^t (t+1)! / ((t-s1)!...)
        # sqrt(Delta) corresponds to 0.5 * log_tri in log space.
        
        sum_val += term
        
    return np.exp(0.5 * log_tri) * sum_val


# --- DATA CLASSES ----------------------------------------------------------

@dataclass(frozen=True)
class ChannelAngularInfo:
    """
    Quantum numbers describing the excitation channel.
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
    Calculate Direct Amplitude coefficient for multipole rank L.
    
    F_L  = (-1)^(l_f) * sqrt[(2 l_f + 1)(2 l_i + 1)]
           * ( l_f   L   l_i )
           * ( 0     0    0  ) * I_L
    """
    li = float(chan.l_i)
    lf = float(chan.l_f)
    Lf = float(L)

    # Selection rule
    if (li + lf + Lf) % 2 != 0:
        return 0.0j

    pref_orb = np.sqrt((2.0 * lf + 1.0) * (2.0 * li + 1.0))
    w3j = wigner_3j_num(lf, Lf, li, 0.0, 0.0, 0.0)
    phase = (-1.0) ** (lf)

    F_L = phase * pref_orb * w3j * I_L
    return complex(F_L)


def _exchange_prefactor(L: int, I_L: float, chan: ChannelAngularInfo) -> complex:
    """
    Calculate Exchange Amplitude coefficient.
    Assuming same angular structure as Direct (Multipole Expansion approximation).
    """
    li = float(chan.l_i)
    lf = float(chan.l_f)
    Lf = float(L)

    if (li + lf + Lf) % 2 != 0:
        return 0.0j

    pref_orb = np.sqrt((2.0 * lf + 1.0) * (2.0 * li + 1.0))
    w3j = wigner_3j_num(lf, Lf, li, 0.0, 0.0, 0.0)
    phase_ex = (-1.0) ** (li + lf + Lf)

    G_L = phase_ex * pref_orb * w3j * I_L
    return complex(G_L)


# --- PUBLIC FUNCTION -------------------------------------------------------

def build_angular_coeffs_for_channel(
    I_L_dict: Dict[int, float],
    chan: ChannelAngularInfo
) -> DWBAAngularCoeffs:
    """
    Map radial DWBA integrals I_L -> angular coefficients F_L, G_L.
    """
    F_L: Dict[int, complex] = {}
    G_L: Dict[int, complex] = {}

    for L, I_L_val in I_L_dict.items():
        F = _direct_prefactor(L, I_L_val, chan)
        G = _exchange_prefactor(L, I_L_val, chan)

        if abs(F.real) < 1e-18 and abs(F.imag) < 1e-18: F = 0j
        if abs(G.real) < 1e-18 and abs(G.imag) < 1e-18: G = 0j

        F_L[L] = complex(F)
        G_L[L] = complex(G)

    return DWBAAngularCoeffs(F_L=F_L, G_L=G_L)
