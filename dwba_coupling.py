# dwba_coupling.py
#
# Angular + spin coupling for DWBA amplitudes.
# Implements the full formulas (Eq. 412, 448) from the article.
#
# Optimization:
# Uses LRU caching for Wigner symbols and factorials to avoid recomputing
# expensive recursions/sums for the same sets of quantum numbers.
#

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import lru_cache

# We need Legendre polynomials / Spherical Harmonics
from scipy.special import sph_harm

@lru_cache(maxsize=2000)
def _log_factorial(n):
    if n < 0: return -1.0 
    if n <= 1: return 0.0
    return np.sum(np.log(np.arange(2, n + 1, dtype=float)))

def _delta_tri(a, b, c):
    # Helper for Wigner/Racah, no need to cache separately if main funcs are cached
    return _log_factorial(a + b - c) + _log_factorial(a - b + c) + \
           _log_factorial(-a + b + c) - _log_factorial(a + b + c + 1)

@lru_cache(maxsize=10000)
def wigner_3j_num(j1, j2, j3, m1, m2, m3):
    if abs(m1 + m2 + m3) > 1e-9: return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2): return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3: return 0.0
    if (j1 + j2 + j3) % 1.0 != 0.0: return 0.0 
    
    t_min = max(0, int(j2 - m1 - j3), int(j1 + m2 - j3))
    t_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))
    if t_min > t_max: return 0.0
        
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

@lru_cache(maxsize=10000)
def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    # CG = (-1)^(j1-j2+m3) * sqrt(2j3+1) * 3j(j1 j2 j3; m1 m2 -m3)
    w3j = wigner_3j_num(j1, j2, j3, m1, m2, -m3)
    factor = np.sqrt(2 * j3 + 1)
    phase = (-1.0)**int(j1 - j2 + m3)
    return phase * factor * w3j

@lru_cache(maxsize=10000)
def wigner_6j_num(j1, j2, j3, j4, j5, j6):
    # Simplified Racah formula
    if not (abs(j1-j2)<=j3<=j1+j2 and abs(j1-j5)<=j6<=j1+j5 and
            abs(j4-j2)<=j6<=j4+j2 and abs(j4-j5)<=j3<=j4+j5):
        return 0.0
    t_min = max(int(j1+j2+j3), int(j1+j5+j6), int(j4+j2+j6), int(j4+j5+j3))
    t_max = min(int(j1+j2+j4+j5), int(j2+j3+j5+j6), int(j3+j1+j6+j4))
    if t_min > t_max: return 0.0
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
        term = (-1)**t * np.exp(num - denom)
        sum_val += term
    return np.exp(0.5 * log_tri) * sum_val

@lru_cache(maxsize=10000)
def racah_W(l1, l2, l3, l4, l5, l6):
    # W(a b c d; e f) = (-1)^(a+b+c+d) * { a b e }
    #                                  { d c f }
    phase = (-1.0)**int(l1 + l2 + l3 + l4)
    return phase * wigner_6j_num(l1, l2, l5, l4, l3, l6)


@dataclass
class Amplitudes:
    f_theta: np.ndarray # Complex array (theta)
    g_theta: np.ndarray # Complex array (theta)

@dataclass(frozen=True)
class ChannelAngularInfo:
    """
    Quantum numbers describing the excitation channel.
    Kept for compatibility with ionization module.
    """
    l_i: int
    l_f: int
    S_i: float
    S_f: float
    L_i: float
    L_f: float
    J_i: float
    J_f: float
    k_i_au: float = 1.0
    k_f_au: float = 1.0

# Deprecated / Legacy Support for Ionization module
def build_angular_coeffs_for_channel(*args, **kwargs):
    raise NotImplementedError("This function is deprecated in v2 partial wave update. Ionization module needs update.")

def calculate_amplitude_contribution(
    theta_grid: np.ndarray,
    I_L_dir: Dict[int, float],
    I_L_exc: Dict[int, float],
    l_i: int,
    l_f: int,
    ki: float,
    kf: float,
    L_target_i: int, 
    L_target_f: int,
    M_target_i: int,
    M_target_f: int
) -> Amplitudes:
    """
    Computes the contribution to f_{M_f M_i}(theta) and g_{M_f M_i}(theta)
    from a single projectile partial wave channel (l_i -> l_f).
    
    Implements Eq. 412 and 448 from the article.
    
    Assumptions:
    - Quantization axis || k_i.
    - Thus mu_i = 0.
    - mu_f is fixed by M conservation: mu_f = M_i + mu_i - M_f = M_i - M_f.
    
    """
    mu_i = 0
    mu_f = M_target_i + mu_i - M_target_f
    
    # Check selection rules for projections
    if abs(mu_f) > l_f:
        return Amplitudes(np.zeros_like(theta_grid), np.zeros_like(theta_grid))
        
    # Precompute geometric factors
    # Eq 412: f = (2/pi) * i^(li+lf) * ...
    # Wait, Eq 412 has Y_{li, mui}^*(k_i). 
    # If k_i || z, Y_{li, 0}(0) = sqrt((2li+1)/4pi).
    
    Y_li_star = np.sqrt((2*l_i+1)/(4*np.pi)) # Real because m=0
    
    # Common factors
    pref_common = (2.0/np.pi) * (1.0/(ki*kf))
    
    # Evaluate Spherical Harmonics for scattered electron Y_{l_f, -mu_f}(k_f)
    # k_f direction is theta. phi can be 0 (azimuthal symmetry for TCS).
    # dsigma/dOmega depends only on theta if unpolarized.
    # Note: Eq 412 has Y_{lf, -mu_f}(k_f).
    # scipy sph_harm(m, l, phi, theta).
    # We set phi=0.
    
    # Note on Condon-Shortley phase: scipy includes it.
    # Article uses standard Ylm.
    
    phi_grid = np.zeros_like(theta_grid)
    Y_lf_val = sph_harm(-mu_f, l_f, phi_grid, theta_grid) 
    
    # DIRECT AMPLITUDE (Eq 412) --
    f_contrib = np.zeros_like(theta_grid, dtype=complex)
    
    # Sum over transfer L_T (Eq 412 says sum over l_T)
    # Sum over g (Eq 412 says sum over g)
    
    # I_L_dir contains values for available l_T (which we called L there).
    for l_T, R_dir in I_L_dir.items():
        if abs(R_dir) < 1e-20: continue
        
        # Factor e terms
        e_lf = np.sqrt(2*l_f+1)
        e_li = np.sqrt(2*l_i+1)
        e_Li = np.sqrt(2*L_target_i+1)
        e_lT = np.sqrt(2*l_T+1)
        
        pre_R = (e_lf * e_li * e_Li) / e_lT
        
        # CGs
        # C1: C(lf li lT; 0 0 0) - parity selection
        cg1 = clebsch_gordan(l_f, l_i, l_T, 0, 0, 0)
        if abs(cg1) < 1e-9: continue
        
        # C2: C(lT Li Lf; 0 0 0)
        cg2 = clebsch_gordan(l_T, L_target_i, L_target_f, 0, 0, 0)
        if abs(cg2) < 1e-9: continue
        
        # Sum over g
        # Triangle rules for g: 
        # From W(lf li Lf Li; lT g) -> g is coupling of (lf, Lf) and (li, Li)
        # Bounds for g
        g_min = max(abs(l_i - L_target_i), abs(l_f - L_target_f))
        g_max = min(l_i + L_target_i, l_f + L_target_f)
        
        sum_g = 0.0
        for g in range(g_min, g_max + 1):
            # W(lf li Lf Li; lT g)
            w_racah = racah_W(l_f, l_i, L_target_f, L_target_i, l_T, g)
            if abs(w_racah) < 1e-9: continue
            
            # C(li Li g; mui Mi mui+Mi) -> C(li Li g; 0 Mi Mi)
            cg_A = clebsch_gordan(l_i, L_target_i, g, 0, M_target_i, M_target_i)
            
            # C(lf Lf g; muf -Mf muf-Mf) 
            # Check Eq 412: C(lf Lf g; mu_f, -M_f, mu_f-M_f)
            # And delta_{mui+Mi, Mf-muf} => Mi = Mf - muf => muf-Mf = -Mi.
            # So 3rd comp is -Mi.
            cg_B = clebsch_gordan(l_f, L_target_f, g, mu_f, -M_target_f, -M_target_i)
            
            sum_g += w_racah * cg_A * cg_B
            
        term = pre_R * cg1 * cg2 * sum_g * R_dir
        f_contrib += term

    # Add global phases and Ylms
    # i^(li + lf)
    phase_i = 1j**(l_i + l_f)
    
    # f = (2/pi) * i^(...) * Y_lf_val * Y_li_star * sum(...)
    # Note: Eq 412 result is scalar (for fixed angles).
    f_total = pref_common * phase_i * Y_lf_val * Y_li_star * f_contrib
    


    # EXCHANGE AMPLITUDE (Eq 448) --
    g_contrib = np.zeros_like(theta_grid, dtype=complex)
    
    for l_T, R_exc in I_L_exc.items():
        if abs(R_exc) < 1e-20: continue
        
        e_Lf = np.sqrt(2*L_target_f+1)
        e_li = np.sqrt(2*l_i+1)
        e_Li = np.sqrt(2*L_target_i+1)
        e_lT = np.sqrt(2*l_T+1)
        
        pre_R = (e_Lf * e_li * e_Li) / e_lT
        
        # CG: C(Lf li lT; 0 0 0)  ! Note order in Eq 448
        cg1 = clebsch_gordan(L_target_f, l_i, l_T, 0, 0, 0)
        # CG: C(lT Li lf; 0 0 0)
        cg2 = clebsch_gordan(l_T, L_target_i, l_f, 0, 0, 0)
        
        if abs(cg1*cg2) < 1e-9: continue
        
        # Sum over g
        # W(Lf li lf Li; lT g)
        g_min = max(abs(l_i - L_target_i), abs(L_target_f - l_f))
        g_max = min(l_i + L_target_i, L_target_f + l_f)
        
        sum_g = 0.0
        for g in range(g_min, g_max + 1):
             w_racah = racah_W(L_target_f, l_i, l_f, L_target_i, l_T, g)
             
             # C(li Li g; mui Mi mui+Mi) -> C(li Li g; 0 Mi Mi)
             cg_A = clebsch_gordan(l_i, L_target_i, g, 0, M_target_i, M_target_i)
             
             # C(Lf lf g; -Mf -muf -Mf-muf)
             # Delta implies mui+Mi = Mf+muf => Mi = Mf+muf => -Mf-muf = -Mi.
             cg_B = clebsch_gordan(L_target_f, l_f, g, -M_target_f, -mu_f, -M_target_i)
             
             sum_g += w_racah * cg_A * cg_B
             
        term = pre_R * cg1 * cg2 * sum_g * R_exc
        g_contrib += term

    # Global phases for g
    # i^(li - lf) * (-1)^(Lf + Mf)
    phase_i_g = 1j**(l_i - l_f)
    phase_parity = (-1.0)**(L_target_f + M_target_f)
    
    # g = ... Y_lf^* ...
    # Y_lf_val computed above is Y(theta, 0). Conjugate is same (real).
    # But strictly Y_{l, m}^* = (-1)^m Y_{l, -m}.
    # Here we used sph_harm(-mu_f). Its conjugate is sph_harm(-mu_f)* ? 
    # Actually Y^* terms in eq 448: Y_{lf, muf}^*(k_f).
    # We Computed Y_{lf, -mu_f}. 
    # Relation: Y_{l, m}^* = (-1)^m Y_{l, -m}.
    # So Y_{lf, muf}^* = (-1)^muf Y_{lf, -muf}.
    # Since we computed Y_val = Y_{lf, -muf}, we just Mult by (-1)^muf.
    
    # Check consistency with Y_li^*. mu_i=0 -> Y_{li,0} is real.
    
    Y_lf_conj = ((-1.0)**mu_f) * Y_lf_val
    
    g_total = pref_common * phase_i_g * phase_parity * Y_lf_conj * Y_li_star * g_contrib

    return Amplitudes(f_total, g_total)
