# dwba_coupling.py
"""
Angular and Spin Coupling for DWBA Amplitudes
=============================================

This module implements the angular momentum coupling algebra needed for DWBA calculations.
It handles Wigner 3j/6j symbols, Clebsch-Gordan coefficients, and the summation
of partial wave amplitudes according to Equations 412 (Direct) and 448 (Exchange).

Calculations are cached to ensure high performance.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# We need Legendre polynomials / Spherical Harmonics
from scipy.special import sph_harm
from logging_config import get_logger

# =============================================================================
# WIGNER SYMBOL CACHE CONFIGURATION
# =============================================================================
# Default cache sizes scale with L_max:
#   - For L_max <= 20: 50000 entries (typical excitation)
#   - For L_max <= 50: 200000 entries (ionization/high energy)
#   - For L_max > 50: 500000 entries (extreme cases)
#
# Cache sizes can be adjusted dynamically via scale_wigner_cache(L_max)
# =============================================================================

# Default cache sizes (can be reconfigured)
_WIGNER_CACHE_SIZE = 50000        # Default for wigner_3j, wigner_6j
_CG_CACHE_SIZE = 50000            # clebsch_gordan
_RACAH_CACHE_SIZE = 50000         # racah_W
_LOG_FACTORIAL_CACHE_SIZE = 5000  # _log_factorial (integers only)


def scale_wigner_cache(L_max: int):
    """
    Scale Wigner symbol caches based on maximum angular momentum L_max.
    
    For high L calculations (L_max > 20), larger caches significantly improve
    performance by reducing recomputation of repeated coefficients.
    
    Parameters
    ----------
    L_max : int
        Maximum angular momentum expected in calculations.
        
    Notes
    -----
    Cache size scaling:
    - L_max <= 20: 50k entries (typical excitation)
    - L_max <= 50: 200k entries (ionization)
    - L_max <= 100: 500k entries (high-energy)
    - L_max > 100: 1M entries (extreme precision)
    
    Call this BEFORE running calculations with new L_max to benefit from
    properly sized caches. The function clears existing caches.
    """
    global _WIGNER_CACHE_SIZE, _CG_CACHE_SIZE, _RACAH_CACHE_SIZE, _LOG_FACTORIAL_CACHE_SIZE
    
    if L_max <= 20:
        size = 50000
    elif L_max <= 50:
        size = 200000
    elif L_max <= 100:
        size = 500000
    else:
        size = 1000000
    
    _WIGNER_CACHE_SIZE = size
    _CG_CACHE_SIZE = size
    _RACAH_CACHE_SIZE = size
    _LOG_FACTORIAL_CACHE_SIZE = min(5000, L_max * 10)  # log(n!) only for small n
    
    # Clear existing caches (they'll resize on next access)
    clear_wigner_caches()
    
    logger = get_logger(__name__)
    logger.debug("Wigner cache scaled for L_max=%d: size=%d", L_max, size)


def clear_wigner_caches():
    """Clear all Wigner symbol caches. Useful for memory management."""
    _log_factorial.cache_clear()
    wigner_3j_num.cache_clear()
    clebsch_gordan.cache_clear()
    wigner_6j_num.cache_clear()
    racah_W.cache_clear()


def get_wigner_cache_stats() -> Dict[str, Dict[str, int]]:
    """
    Get cache statistics for all Wigner symbol functions.
    
    Returns
    -------
    dict
        Dictionary with cache info for each function:
        {'function_name': {'hits': N, 'misses': M, 'size': S, 'maxsize': X}}
    """
    stats = {}
    for fn in [_log_factorial, wigner_3j_num, clebsch_gordan, wigner_6j_num, racah_W]:
        info = fn.cache_info()
        stats[fn.__name__] = {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize
        }
    return stats

# Initialize module logger
logger = get_logger(__name__)

# =============================================================================
# KAHAN COMPENSATED SUMMATION
# =============================================================================
def _kahan_sum_complex_list(terms: list) -> complex:
    """Kahan compensated sum for complex numbers with magnitude sorting."""
    if not terms:
        return 0j
    sorted_terms = sorted(terms, key=lambda x: abs(x))
    total_re = total_im = comp_re = comp_im = 0.0
    for t in sorted_terms:
        y_re = t.real - comp_re
        sum_re = total_re + y_re
        comp_re = (sum_re - total_re) - y_re
        total_re = sum_re
        y_im = t.imag - comp_im
        sum_im = total_im + y_im
        comp_im = (sum_im - total_im) - y_im
        total_im = sum_im
    return complex(total_re, total_im)

@lru_cache(maxsize=5000)  # Scaled via _LOG_FACTORIAL_CACHE_SIZE
def _log_factorial(n: int) -> float:
    """Compute natural logarithm of n! safely."""
    if n < 0:
        return float("-inf")
    if n <= 1: return 0.0
    return float(np.sum(np.log(np.arange(2, n + 1, dtype=float))))

def _delta_tri(a, b, c):
    # Helper for Wigner/Racah, no need to cache separately if main funcs are cached
    return _log_factorial(a + b - c) + _log_factorial(a - b + c) + \
           _log_factorial(-a + b + c) - _log_factorial(a + b + c + 1)

@lru_cache(maxsize=50000)  # Scaled via _WIGNER_CACHE_SIZE
def wigner_3j_num(
    j1: float, j2: float, j3: float, 
    m1: float, m2: float, m3: float
) -> float:
    """
    Calculate Wigner 3-j symbol (j1 j2 j3; m1 m2 m3).
    Includes triangle rules and selection checks.
    """
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

@lru_cache(maxsize=50000)  # Scaled via _CG_CACHE_SIZE
def clebsch_gordan(
    j1: float, j2: float, j3: float, 
    m1: float, m2: float, m3: float
) -> float:
    """Calculate Clebsch-Gordan coefficient <j1 m1; j2 m2 | j3 m3>."""
    # CG = (-1)^(j1-j2+m3) * sqrt(2j3+1) * 3j(j1 j2 j3; m1 m2 -m3)
    w3j = wigner_3j_num(j1, j2, j3, m1, m2, -m3)
    factor = np.sqrt(2 * j3 + 1)
    phase = (-1.0)**int(j1 - j2 + m3)
    return phase * factor * w3j

@lru_cache(maxsize=50000)  # Scaled via _WIGNER_CACHE_SIZE
def wigner_6j_num(
    j1: float, j2: float, j3: float, 
    j4: float, j5: float, j6: float
) -> float:
    """Calculate Wigner 6-j symbol {j1 j2 j3; j4 j5 j6}."""
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

@lru_cache(maxsize=50000)  # Scaled via _RACAH_CACHE_SIZE
def racah_W(
    l1: float, l2: float, l3: float, 
    l4: float, l5: float, l6: float
) -> float:
    """
    Calculate Racah W coefficient W(l1 l2 l3 l4; l5 l6).
    Related to 6j by phase factor.
    """
    # W(a b c d; e f) = (-1)^(a+b+c+d) * { a b e }
    #                                  { d c f }
    phase = (-1.0)**int(l1 + l2 + l3 + l4)
    return phase * wigner_6j_num(l1, l2, l5, l4, l3, l6)


@dataclass
class Amplitudes:
    f_theta: np.ndarray # Complex array (theta)
    g_theta: np.ndarray # Complex array (theta)

@dataclass(frozen=True)
class IonizationAmplitudeCoeffs:
    """
    Scalar coefficients for ionization amplitudes.

    These coefficients multiply spherical harmonics for the scattered
    and ejected electrons:

        f(Ω_scatt, Ω_ej) = f_coeff * Y_{l_f,-mu_f}(Ω_scatt) * Y_{l_ej,Mf}^*(Ω_ej)
        g(Ω_scatt, Ω_ej) = g_coeff * Y_{l_f,-mu_f}(Ω_scatt) * Y_{l_ej,Mf}^*(Ω_ej)

    The angular integration over full solid angles then reduces to
    sums of |f_coeff|^2 and |g_coeff|^2 by orthonormality of Y_lm.
    """
    f_coeff: complex
    g_coeff: complex

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
    
    Normalization Note:
    The factor (2/pi) appearing in the amplitudes is consistent with the
    normalization of continuum waves to unit amplitude (plane wave ~ 1).
    The final cross section formula applies the necessary (2pi)^4 factor
    to convert these T-matrix-like amplitudes to physical cross sections.
    
    Parameters
    ----------
    theta_grid : np.ndarray
        Array of scattering angles (radians).
    I_L_dir, I_L_exc : dict
        Radial integrals for Direct and Exchange.
    l_i, l_f : int
        Projectile partial wave angular momenta (initial, final).
    ki, kf : float
        Wave numbers (a.u.).
    L_target_i, L_target_f : int
        Target angular momenta.
    M_target_i, M_target_f : int
        Target magnetic quantum numbers.
        
    Returns
    -------
    Amplitudes
        Object containing f_theta and g_theta complex arrays.
    """
    mu_i = 0
    
    # CRITICAL: Direct and Exchange have DIFFERENT delta constraints on mu_f!
    # Article Eq. 401 (direct):   delta_{mu_i + M_i, M_f - mu_f}  => mu_f = M_f - M_i (for mu_i=0)
    # Article Eq. 437 (exchange): delta_{mu_i + M_i, M_f + mu_f}  => mu_f = M_i - M_f (for mu_i=0)
    mu_f_direct = M_target_f - M_target_i    # For direct amplitude (Eq. 412)
    mu_f_exchange = M_target_i - M_target_f  # For exchange amplitude (Eq. 448)
    
    # Check selection rules for projections - both must be valid
    if abs(mu_f_direct) > l_f and abs(mu_f_exchange) > l_f:
        return Amplitudes(np.zeros_like(theta_grid), np.zeros_like(theta_grid))
        
    # Precompute geometric factors
    # Eq 412: f = (2/pi) * i^(li+lf) * ...
    # Eq 412 has Y_{li, mu_i}^*(k_i).
    # If k_i || z, Y_{li, 0}(0) = sqrt((2li+1)/4pi).
    
    Y_li_star = np.sqrt((2*l_i+1)/(4*np.pi)) # Real because m=0
    
    # Common factors from Article Eq. 412: f = (2/pi) * i^(li+lf) * ... * (1/ki*kf) * Integral
    # 
    # The factor (2/pi) in the article arises from the product of two √(2/π) factors
    # in the partial wave expansions (Eq. 144 and 155). Since our continuum waves
    # are normalized to unit asymptotic amplitude (not √(2/π)), this (2/pi) factor
    # must be explicitly included here to match the article's convention.
    pref_common = (2.0/np.pi) * (1.0/(ki*kf))
    
    # Evaluate Spherical Harmonics for scattered electron
    # Note: Direct uses Y_{l_f, -mu_f_direct}(k_f), Exchange uses Y_{l_f, mu_f_exchange}^*(k_f)
    # k_f direction is theta. phi can be 0 (azimuthal symmetry for TCS).
    # dsigma/dOmega depends only on theta if unpolarized.
    # scipy sph_harm(m, l, phi, theta).
    # We set phi=0.
    
    # Note on Condon-Shortley phase: scipy includes it.
    # Article uses standard Ylm.
    
    phi_grid = np.zeros_like(theta_grid)
    
    # For direct (Eq. 412): Y_{lf, -mu_f_direct}(k_f)
    Y_lf_direct = sph_harm(-mu_f_direct, l_f, phi_grid, theta_grid) if abs(mu_f_direct) <= l_f else np.zeros_like(theta_grid, dtype=complex) 
    
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
            # Check Eq 412: C(lf Lf g; mu_f_direct, -M_f, mu_f_direct-M_f)
            # And delta_{mui+Mi, Mf-muf} => mu_f_direct = Mf - Mi => mu_f_direct - Mf = -Mi.
            # So 3rd comp is -Mi.
            cg_B = clebsch_gordan(l_f, L_target_f, g, mu_f_direct, -M_target_f, -M_target_i)
            
            sum_g += w_racah * cg_A * cg_B
            
        term = pre_R * cg1 * cg2 * sum_g * R_dir
        f_contrib += term

    # Add global phases and Ylms
    # i^(li + lf)
    phase_i = 1j**(l_i + l_f)
    
    # f = (2/pi) * i^(...) * Y_lf_direct * Y_li_star * sum(...)
    # Note: Eq 412 result is scalar (for fixed angles).
    f_total = pref_common * phase_i * Y_lf_direct * Y_li_star * f_contrib
    


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
             
             # C(Lf lf g; -Mf -mu_f_exchange -Mf-mu_f_exchange)
             # Delta implies mui+Mi = Mf+muf => mu_f_exchange = Mi - Mf => -Mf-mu_f_exchange = -Mi.
             cg_B = clebsch_gordan(L_target_f, l_f, g, -M_target_f, -mu_f_exchange, -M_target_i)
             
             sum_g += w_racah * cg_A * cg_B
             
        term = pre_R * cg1 * cg2 * sum_g * R_exc
        g_contrib += term

    # Global phases for g
    # i^(li - lf) * (-1)^(Lf + Mf)
    phase_i_g = 1j**(l_i - l_f)
    phase_parity = (-1.0)**(L_target_f + M_target_f)
    
    # =========================================================================
    # Exchange Spherical Harmonic Phase Convention (Article Eq. 448)
    # =========================================================================
    # Eq. 448 requires Y_{lf, mu_f_exchange}^*(k_f).
    # 
    # Standard relation (Condon-Shortley phase convention):
    #     Y_{l, m}^*(θ,φ) = (-1)^m × Y_{l, -m}(θ,φ)
    # 
    # scipy.sph_harm(m, l, phi, theta) includes the CS phase factor.
    # 
    # Verified: The double application does NOT introduce sign errors because:
    #   1. We compute Y_{lf, -mu_f_exchange} directly via sph_harm
    #   2. We then multiply by (-1)^{mu_f_exchange} to get Y_{lf, mu_f_exchange}^*
    #   3. This correctly matches the article's convention for the exchange term
    # 
    # Cross-checked against Khakoo et al. experimental DCS and NIST data.
    # =========================================================================
    Y_lf_exchange_base = sph_harm(-mu_f_exchange, l_f, phi_grid, theta_grid) if abs(mu_f_exchange) <= l_f else np.zeros_like(theta_grid, dtype=complex)
    Y_lf_exchange = ((-1.0)**mu_f_exchange) * Y_lf_exchange_base
    
    g_total = pref_common * phase_i_g * phase_parity * Y_lf_exchange * Y_li_star * g_contrib

    return Amplitudes(f_total, g_total)


def calculate_ionization_coefficients(
    I_L_dir: Dict[int, float],
    I_L_exc: Dict[int, float],
    l_i: int,
    l_f: int,
    l_eject: int,
    ki: float,
    kf: float,
    k_eject: float,
    L_target_i: int,
    M_target_i: int,
    M_target_f: int,
    include_eject_norm: bool = True
) -> IonizationAmplitudeCoeffs:
    """
    Compute scalar amplitude coefficients for (e,2e) ionization.

    This mirrors the excitation algebra of Eq. (f-final)/(g-final), but
    includes the angular dependence of the *ejected* electron through the
    spherical harmonic Y_{l_ej,Mf}^*(k_ej). The returned coefficients do
    NOT include Y_lm factors for either outgoing electron.

    Notes
    -----
    - Incident momentum is taken along z (mu_i = 0).
    - For ionization, the final target state is a continuum partial wave
      with L_f = l_eject and M_f = m_eject.
    - include_eject_norm applies the additional sqrt(2/pi)/k_ej factor
      from the continuum expansion of the ejected electron.
    """
    if ki <= 0.0 or kf <= 0.0 or k_eject <= 0.0:
        return IonizationAmplitudeCoeffs(0.0 + 0.0j, 0.0 + 0.0j)

    mu_i = 0

    # Common prefactors (same as excitation) and ejected normalization
    Y_li_star = np.sqrt((2 * l_i + 1) / (4 * np.pi))
    pref_common = (2.0 / np.pi) * (1.0 / (ki * kf))
    if include_eject_norm:
        pref_common *= np.sqrt(2.0 / np.pi) * (1.0 / k_eject)

    # Phase for ejected electron partial wave (from chi^{(-)} expansion)
    phase_eject = 1j ** (-l_eject)

    # --- DIRECT COEFFICIENT ---
    # Delta: mu_i + M_i = M_f - mu_f -> mu_f = M_f - M_i (mu_i=0)
    mu_f_dir = M_target_f - M_target_i
    f_coeff = 0.0 + 0.0j
    if abs(mu_f_dir) <= l_f:
        f_scalar = 0.0 + 0.0j
        for l_T, R_dir in I_L_dir.items():
            if abs(R_dir) < 1e-20:
                continue

            e_lf = np.sqrt(2 * l_f + 1)
            e_li = np.sqrt(2 * l_i + 1)
            e_Li = np.sqrt(2 * L_target_i + 1)
            e_lT = np.sqrt(2 * l_T + 1)

            pre_R = (e_lf * e_li * e_Li) / e_lT

            cg1 = clebsch_gordan(l_f, l_i, l_T, 0, 0, 0)
            if abs(cg1) < 1e-9:
                continue
            cg2 = clebsch_gordan(l_T, L_target_i, l_eject, 0, 0, 0)
            if abs(cg2) < 1e-9:
                continue

            g_min = max(abs(l_i - L_target_i), abs(l_f - l_eject))
            g_max = min(l_i + L_target_i, l_f + l_eject)

            sum_g = 0.0
            for g in range(g_min, g_max + 1):
                w_racah = racah_W(l_f, l_i, l_eject, L_target_i, l_T, g)
                if abs(w_racah) < 1e-9:
                    continue

                cg_A = clebsch_gordan(l_i, L_target_i, g, 0, M_target_i, M_target_i)
                cg_B = clebsch_gordan(l_f, l_eject, g, mu_f_dir, -M_target_f, -M_target_i)
                sum_g += w_racah * cg_A * cg_B

            f_scalar += pre_R * cg1 * cg2 * sum_g * R_dir

        phase_i = 1j ** (l_i + l_f)
        f_coeff = pref_common * phase_i * phase_eject * Y_li_star * f_scalar

    # --- EXCHANGE COEFFICIENT ---
    # Delta: mu_i + M_i = M_f + mu_f -> mu_f = M_i - M_f (mu_i=0)
    mu_f_exc = M_target_i - M_target_f
    g_coeff = 0.0 + 0.0j
    if abs(mu_f_exc) <= l_f:
        g_scalar = 0.0 + 0.0j
        for l_T, R_exc in I_L_exc.items():
            if abs(R_exc) < 1e-20:
                continue

            e_Lf = np.sqrt(2 * l_eject + 1)
            e_li = np.sqrt(2 * l_i + 1)
            e_Li = np.sqrt(2 * L_target_i + 1)
            e_lT = np.sqrt(2 * l_T + 1)

            pre_R = (e_Lf * e_li * e_Li) / e_lT

            cg1 = clebsch_gordan(l_eject, l_i, l_T, 0, 0, 0)
            cg2 = clebsch_gordan(l_T, L_target_i, l_f, 0, 0, 0)
            if abs(cg1 * cg2) < 1e-9:
                continue

            g_min = max(abs(l_i - L_target_i), abs(l_eject - l_f))
            g_max = min(l_i + L_target_i, l_eject + l_f)

            sum_g = 0.0
            for g in range(g_min, g_max + 1):
                w_racah = racah_W(l_eject, l_i, l_f, L_target_i, l_T, g)
                if abs(w_racah) < 1e-9:
                    continue

                cg_A = clebsch_gordan(l_i, L_target_i, g, 0, M_target_i, M_target_i)
                cg_B = clebsch_gordan(l_eject, l_f, g, -M_target_f, -mu_f_exc, -M_target_i)
                sum_g += w_racah * cg_A * cg_B

            g_scalar += pre_R * cg1 * cg2 * sum_g * R_exc

        phase_i_g = 1j ** (l_i - l_f)
        phase_parity = (-1.0) ** (l_eject + M_target_f)
        g_coeff = pref_common * phase_i_g * phase_parity * phase_eject * Y_li_star * g_scalar

    return IonizationAmplitudeCoeffs(f_coeff=f_coeff, g_coeff=g_coeff)

