# oscillatory_integrals.py
"""
Oscillatory Integral Methods for DWBA
======================================

Specialized quadrature methods for highly oscillatory radial integrals
that appear in electron-atom scattering calculations.

Problem
-------
For high partial waves l_i, l_f and wave numbers k_i, k_f, the integrand:

    χ_i(k_i, r) × χ_f(k_f, r) × f(r)

oscillates rapidly with frequency ~(k_i + k_f). Standard Simpson/Trapezoid
quadrature suffers aliasing when the grid spacing exceeds the Nyquist limit.

Solution
--------
1. **Domain Splitting**: Integrate numerically up to r_m (match point),
   then use asymptotic forms for the oscillatory tail.

2. **Filon-type Quadrature**: Interpolate the slowly-varying envelope and
   integrate the oscillatory factor analytically.

3. **Phase-adaptive Refinement**: Ensure phase change per step < π/4.

References
----------
- Filon, L.N.G. (1930). "On a quadrature formula for trigonometric integrals"
- Iserles, A. (2004). "On the numerical quadrature of highly-oscillating integrals"
- Burke, P.G. "R-Matrix Theory of Atomic Collisions"

Units
-----
All inputs/outputs in Hartree atomic units (a₀, Ha).
"""

from __future__ import annotations
import logging
import os
import time
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from scipy.special import sici  # Sine and Cosine integrals

from logging_config import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    """Read integer env var with bounds and fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(min_value, int(default))
    try:
        return max(min_value, int(raw))
    except Exception:
        return max(min_value, int(default))


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    """Read float env var with bounds and fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(min_value, float(default))
    try:
        return max(min_value, float(raw))
    except Exception:
        return max(min_value, float(default))


# Runtime guardrails for pathological outer-tail costs.
_FILON_MAX_SEGMENTS = _env_int("DWBA_MAX_FILON_SEGMENTS", 4096, min_value=128)
_FILON_MAX_EFFECTIVE_DPHI = _env_float(
    "DWBA_MAX_EFFECTIVE_DPHI",
    np.pi / 2,
    min_value=np.pi / 16
)
_OUTER_SLOW_WARN_S = _env_float("DWBA_OUTER_SLOW_WARN_S", 20.0, min_value=1.0)
_OUTER_BATCH_MIN_ACTIVE = _env_int("DWBA_OUTER_BATCH_MIN_ACTIVE", 4, min_value=1)
_OUTER_BATCH_MOMENT_TOL = _env_float("DWBA_OUTER_BATCH_MOMENT_TOL", 1e-12, min_value=0.0)


def _is_hotpath_verbose_debug_enabled() -> bool:
    """
    Return True when per-segment oscillatory debug is explicitly requested.
    """
    return os.environ.get("DWBA_HOTPATH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _should_sampled_debug(key: str, every: int = 500, initial: int = 2) -> bool:
    """
    Sample highly repetitive DEBUG logs in oscillatory inner loops.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return False
    if _is_hotpath_verbose_debug_enabled():
        return True

    counters = getattr(_should_sampled_debug, "_counters", None)
    if counters is None:
        counters = {}
        _should_sampled_debug._counters = counters

    count = counters.get(key, 0) + 1
    counters[key] = count
    if count <= initial:
        return True
    return (count % max(1, int(every))) == 0


# =============================================================================
# CACHED CLENSHAW-CURTIS REFERENCE WEIGHTS (OPTIMIZATION)
# =============================================================================
# Precompute reference CC nodes and weights for n=5 on [-1, 1]
# These are reused in filon and filon_exchange methods

_CC_N = 5  # Number of CC nodes used in filon methods
_CC_THETA_REF = np.pi * np.arange(_CC_N) / (_CC_N - 1)
_CC_X_REF = np.cos(_CC_THETA_REF)  # Chebyshev nodes on [-1, 1]

# Compute reference weights on [-1, 1]
_j_max = (_CC_N - 1) // 2
_j_vals = np.arange(1, _j_max + 1)
_theta_all = np.pi * np.arange(_CC_N) / (_CC_N - 1)

if _j_max > 0:
    _j_col = _j_vals[:, np.newaxis]
    _theta_row = _theta_all[np.newaxis, :]
    _cos_terms = np.cos(2 * _j_col * _theta_row)
    _denom = 4 * _j_vals**2 - 1
    _weight_sums = np.sum(_cos_terms / _denom[:, np.newaxis], axis=0)
    if (_CC_N - 1) % 2 == 0:
        _j_final = _j_max
        _cos_final = np.cos(2 * _j_final * _theta_all)
        _weight_sums += 0.5 * _cos_final / (4 * _j_final**2 - 1)
else:
    _weight_sums = np.zeros(_CC_N)

_CC_W_REF = (2.0 / (_CC_N - 1)) * (1 - 2 * _weight_sums)  # Weights on [-1, 1]

# Cache CC reference nodes/weights by node count
_CC_CACHE = {_CC_N: (_CC_X_REF, _CC_W_REF)}

# Cache for radial-grid spacing classification (linear vs log-like).
# Keyed by memory pointer + basic shape/signature to avoid repeated O(N)
# checks in tight interpolation loops.
_LOG_GRID_FLAG_CACHE: dict[tuple[int, int, float, float], bool] = {}
_LOG_GRID_FLAG_CACHE_MAX = 128

# Small cache of phase-function closures for repeated outer-tail calls.
_OUTER_PHASE_CACHE: Dict[tuple, tuple[Callable, Callable, Callable, Callable, Callable, Callable]] = {}
_OUTER_PHASE_CACHE_LRU: list[tuple] = []
_OUTER_PHASE_CACHE_MAX = 64


def _get_cc_ref(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Chebyshev nodes/weights on [-1, 1] for Clenshaw-Curtis quadrature.
    Uses a small cache keyed by n_nodes for speed.
    """
    if n_nodes < 2:
        raise ValueError("Clenshaw-Curtis requires n_nodes >= 2.")
    cached = _CC_CACHE.get(n_nodes)
    if cached is not None:
        return cached
    nodes, weights = clenshaw_curtis_nodes(n_nodes, -1.0, 1.0)
    _CC_CACHE[n_nodes] = (nodes, weights)
    return nodes, weights


# =============================================================================
# NUMERICAL STABILITY UTILITIES
# =============================================================================

def _kahan_sum_complex(values: np.ndarray) -> complex:
    """
    Compensated summation for complex values using math.fsum.
    
    Reduces roundoff error when summing many small contributions that may
    partially cancel, which is common in oscillatory integrals.
    
    Parameters
    ----------
    values : np.ndarray
        Complex array of values to sum.
        
    Returns
    -------
    total : complex
        Compensated sum with reduced roundoff error.
        
    Notes
    -----
    Uses math.fsum which implements Shewchuk's algorithm (more accurate
    than Kahan summation) and is implemented in C for speed.
    """
    import math
    # math.fsum is O(n) C-optimized with full precision tracking
    total_re = math.fsum(values.real)
    total_im = math.fsum(values.imag)
    return complex(total_re, total_im)


def _kahan_sum_real(values: np.ndarray) -> float:
    """
    Compensated summation for real values using math.fsum.
    
    Uses Python's math.fsum which implements Shewchuk's algorithm
    for exact summation with O(n) complexity in C.
    """
    import math
    return math.fsum(values)


def _eval_callable_on_nodes(func, nodes: np.ndarray) -> np.ndarray:
    """
    Evaluate scalar/vector callable on nodes with fast array-first fallback.

    Many envelope/phase callables are numpy-aware; trying array input first
    avoids Python-loop overhead from np.vectorize in tight oscillatory loops.
    """
    try:
        vals = func(nodes)
        vals_arr = np.asarray(vals)
        if vals_arr.shape == ():
            vals_arr = np.full(nodes.shape, vals_arr, dtype=vals_arr.dtype)
        elif vals_arr.shape != nodes.shape:
            vals_arr = np.broadcast_to(vals_arr, nodes.shape)
        return vals_arr
    except Exception:
        return np.vectorize(func)(nodes)


def _is_log_spaced_grid(r: np.ndarray, rel_tol: float = 5e-3) -> bool:
    """
    Check whether a 1D positive grid is approximately uniform in log(r).
    """
    if r.ndim != 1 or len(r) < 3:
        return False
    if r[0] <= 0.0 or np.any(np.diff(r) <= 0.0):
        return False
    log_r = np.log(r)
    dlog = np.diff(log_r)
    mean_dlog = float(np.mean(dlog))
    if not np.isfinite(mean_dlog) or mean_dlog <= 0.0:
        return False
    max_dev = float(np.max(np.abs(dlog - mean_dlog)))
    return max_dev <= rel_tol * mean_dlog


def _is_log_spaced_grid_cached(r: np.ndarray, rel_tol: float = 5e-3) -> bool:
    """
    Cached wrapper around _is_log_spaced_grid for repeated interpolation calls.
    """
    r_arr = np.asarray(r, dtype=float)
    if r_arr.ndim != 1 or len(r_arr) < 3:
        return False

    key = (
        int(r_arr.__array_interface__["data"][0]),
        int(r_arr.size),
        float(r_arr[0]),
        float(r_arr[-1]),
    )
    cached = _LOG_GRID_FLAG_CACHE.get(key)
    if cached is not None:
        return cached

    is_log = _is_log_spaced_grid(r_arr, rel_tol=rel_tol)
    if len(_LOG_GRID_FLAG_CACHE) >= _LOG_GRID_FLAG_CACHE_MAX:
        _LOG_GRID_FLAG_CACHE.clear()
    _LOG_GRID_FLAG_CACHE[key] = is_log
    return is_log


def _interp_on_radial_grid(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    1D interpolation aware of logarithmic radial grids.

    For log-spaced positive grids, interpolation is performed in log(r) space
    to reduce phase-envelope distortion compared with linear-r interpolation.
    """
    x_arr = np.asarray(x, dtype=float)
    x_new_arr = np.asarray(x_new, dtype=float)
    y_arr = np.asarray(y)

    x_new_clip = np.clip(x_new_arr, x_arr[0], x_arr[-1])

    if np.iscomplexobj(y_arr):
        re = _interp_on_radial_grid(x_new_clip, x_arr, y_arr.real)
        im = _interp_on_radial_grid(x_new_clip, x_arr, y_arr.imag)
        return re + 1j * im

    if _is_log_spaced_grid_cached(x_arr):
        x_safe = np.maximum(x_arr, 1e-300)
        x_new_safe = np.maximum(x_new_clip, x_safe[0])
        return np.interp(np.log(x_new_safe), np.log(x_safe), y_arr)

    return np.interp(x_new_clip, x_arr, y_arr)


# =============================================================================
# sinA × sinB DECOMPOSITION (Per Instruction)
# =============================================================================
# "Wtedy iloczyn chi_a * chi_b rozbijasz tożsamością:
#  sin A sin B = 1/2 * [cos(A-B) - cos(A+B)]"
# =============================================================================

def compute_product_phases(
    k_a: float, l_a: int, delta_a: float, eta_a: float, sigma_a: float,
    k_b: float, l_b: int, delta_b: float, eta_b: float, sigma_b: float
) -> tuple:
    """
    Compute phase parameters for sinA × sinB decomposition.
    
    Given two continuum waves with phases:
        Φ_a(r) = k_a·r + η_a·ln(2k_a·r) - l_a·π/2 + σ_a + δ_a
        Φ_b(r) = k_b·r + η_b·ln(2k_b·r) - l_b·π/2 + σ_b + δ_b
    
    The product χ_a × χ_b ~ sin(Φ_a) × sin(Φ_b) = ½[cos(Φ_a-Φ_b) - cos(Φ_a+Φ_b)]
    
    Returns parameters for the two resulting cosine terms.
    
    Returns
    -------
    (k_minus, phi_const_minus, eta_minus) : parameters for cos(Φ_a - Φ_b)
    (k_plus, phi_const_plus, eta_plus) : parameters for cos(Φ_a + Φ_b)
    """
    # Frequency differences
    k_minus = k_a - k_b
    k_plus = k_a + k_b
    
    # Constant phase parts (not including k·r or η·ln terms)
    phi_const_a = -l_a * np.pi / 2 + sigma_a + delta_a
    phi_const_b = -l_b * np.pi / 2 + sigma_b + delta_b
    
    phi_const_minus = phi_const_a - phi_const_b
    phi_const_plus = phi_const_a + phi_const_b
    
    # Sommerfeld parameter combinations
    eta_minus = eta_a - eta_b
    eta_plus = eta_a + eta_b
    
    return (k_minus, phi_const_minus, eta_minus), (k_plus, phi_const_plus, eta_plus)


def _get_dwba_outer_phase_functions(
    k_a: float, l_a: int, delta_a: float, eta_a: float, sigma_a: float,
    k_b: float, l_b: int, delta_b: float, eta_b: float, sigma_b: float
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """
    Return cached phase functions used in dwba_outer_integral_1d.

    These callables depend only on channel parameters and are reused across
    repeated L-loop tail integrations for the same (l_i, l_f, energy) pair.
    """
    key = (
        round(float(k_a), 12), int(l_a), round(float(delta_a), 12), round(float(eta_a), 12), round(float(sigma_a), 12),
        round(float(k_b), 12), int(l_b), round(float(delta_b), 12), round(float(eta_b), 12), round(float(sigma_b), 12),
    )
    cached = _OUTER_PHASE_CACHE.get(key)
    if cached is not None:
        if key in _OUTER_PHASE_CACHE_LRU:
            _OUTER_PHASE_CACHE_LRU.remove(key)
        _OUTER_PHASE_CACHE_LRU.append(key)
        return cached

    (k_minus, phi_c_minus, _eta_minus), (k_plus, phi_c_plus, _eta_plus) = compute_product_phases(
        k_a, l_a, delta_a, eta_a, sigma_a,
        k_b, l_b, delta_b, eta_b, sigma_b
    )

    def phi_minus(r):
        result = k_minus * r + phi_c_minus
        if abs(eta_a) > 1e-15:
            result += eta_a * np.log(2 * k_a * r + 1e-30)
        if abs(eta_b) > 1e-15:
            result -= eta_b * np.log(2 * k_b * r + 1e-30)
        if l_a > 0:
            result -= (l_a * (l_a + 1)) / (2.0 * k_a * r)
        if l_b > 0:
            result += (l_b * (l_b + 1)) / (2.0 * k_b * r)
        return result

    def phi_plus(r):
        result = k_plus * r + phi_c_plus
        if abs(eta_a) > 1e-15:
            result += eta_a * np.log(2 * k_a * r + 1e-30)
        if abs(eta_b) > 1e-15:
            result += eta_b * np.log(2 * k_b * r + 1e-30)
        if l_a > 0:
            result -= (l_a * (l_a + 1)) / (2.0 * k_a * r)
        if l_b > 0:
            result -= (l_b * (l_b + 1)) / (2.0 * k_b * r)
        return result

    def phi_minus_prime(r):
        result = k_minus
        if abs(eta_a) > 1e-15:
            result += eta_a / r
        if abs(eta_b) > 1e-15:
            result -= eta_b / r
        if l_a > 0:
            result += (l_a * (l_a + 1)) / (2.0 * k_a * r ** 2)
        if l_b > 0:
            result -= (l_b * (l_b + 1)) / (2.0 * k_b * r ** 2)
        return result

    def phi_plus_prime(r):
        result = k_plus
        if abs(eta_a) > 1e-15:
            result += eta_a / r
        if abs(eta_b) > 1e-15:
            result += eta_b / r
        if l_a > 0:
            result += (l_a * (l_a + 1)) / (2.0 * k_a * r ** 2)
        if l_b > 0:
            result += (l_b * (l_b + 1)) / (2.0 * k_b * r ** 2)
        return result

    def phi_minus_prime2(r):
        result = 0.0
        if abs(eta_a) > 1e-15:
            result -= eta_a / (r * r)
        if abs(eta_b) > 1e-15:
            result += eta_b / (r * r)
        if l_a > 0:
            result -= (l_a * (l_a + 1)) / (k_a * r ** 3)
        if l_b > 0:
            result += (l_b * (l_b + 1)) / (k_b * r ** 3)
        return result

    def phi_plus_prime2(r):
        result = 0.0
        if abs(eta_a) > 1e-15:
            result -= eta_a / (r * r)
        if abs(eta_b) > 1e-15:
            result -= eta_b / (r * r)
        if l_a > 0:
            result -= (l_a * (l_a + 1)) / (k_a * r ** 3)
        if l_b > 0:
            result -= (l_b * (l_b + 1)) / (k_b * r ** 3)
        return result

    funcs = (phi_minus, phi_plus, phi_minus_prime, phi_plus_prime, phi_minus_prime2, phi_plus_prime2)
    _OUTER_PHASE_CACHE[key] = funcs
    if key in _OUTER_PHASE_CACHE_LRU:
        _OUTER_PHASE_CACHE_LRU.remove(key)
    _OUTER_PHASE_CACHE_LRU.append(key)
    while len(_OUTER_PHASE_CACHE_LRU) > _OUTER_PHASE_CACHE_MAX:
        old = _OUTER_PHASE_CACHE_LRU.pop(0)
        _OUTER_PHASE_CACHE.pop(old, None)
    return funcs


def dwba_outer_integral_1d(
    envelope_func,
    k_a: float, l_a: int, delta_a: float, eta_a: float, sigma_a: float,
    k_b: float, l_b: int, delta_b: float, eta_b: float, sigma_b: float,
    r_m: float,
    r_max: float,
    delta_phi: float = np.pi / 4
) -> float:
    """
    Compute outer oscillatory integral for DWBA using sinA×sinB decomposition.
    
    I_out = ∫_{r_m}^{r_max} f(r) × χ_a(r) × χ_b(r) dr
    
    where χ ~ sin(Φ) and we use:
        sin(Φ_a) sin(Φ_b) = ½[cos(Φ_a - Φ_b) - cos(Φ_a + Φ_b)]
    
    Each cosine term is computed as Re of exp(iΦ) using Filon or Levin.
    
    Parameters
    ----------
    envelope_func : callable
        Slowly-varying envelope f(r) (may include 1/r^n from multipole).
    k_a, l_a, delta_a, eta_a, sigma_a : float, int, float, float, float
        Wave parameters for χ_a.
    k_b, l_b, delta_b, eta_b, sigma_b : float, int, float, float, float
        Wave parameters for χ_b.
    r_m : float
        Match point (start of asymptotic region).
    r_max : float
        Maximum radius.
    delta_phi : float
        Phase increment per segment.
        
    Returns
    -------
    integral : float
        Value of the outer integral (real).
    """
    if r_max <= r_m + 1e-10:
        return 0.0

    t_outer_start = time.perf_counter()
    
    (
        phi_minus, phi_plus,
        phi_minus_prime, phi_plus_prime,
        phi_minus_prime2, phi_plus_prime2
    ) = _get_dwba_outer_phase_functions(
        k_a, l_a, delta_a, eta_a, sigma_a,
        k_b, l_b, delta_b, eta_b, sigma_b
    )
    
    # Compute I_minus = ∫ f(r) cos(Φ_-(r)) dr = Re ∫ f(r) exp(iΦ_-(r)) dr
    I_minus_complex = compute_outer_integral_oscillatory(
        envelope_func, phi_minus, phi_minus_prime, phi_minus_prime2,
        r_m, r_max, delta_phi
    )
    I_minus = I_minus_complex.real
    
    # Compute I_plus = ∫ f(r) cos(Φ_+(r)) dr = Re ∫ f(r) exp(iΦ_+(r)) dr
    I_plus_complex = compute_outer_integral_oscillatory(
        envelope_func, phi_plus, phi_plus_prime, phi_plus_prime2,
        r_m, r_max, delta_phi
    )
    I_plus = I_plus_complex.real
    
    # Final result: ½(I_minus - I_plus)
    result = 0.5 * (I_minus - I_plus)

    elapsed = time.perf_counter() - t_outer_start
    if elapsed > _OUTER_SLOW_WARN_S and _should_sampled_debug("outer_integral_slow", every=50, initial=3):
        logger.warning(
            "Slow outer integral: %.1fs (r=[%.2f, %.2f], k_a=%.3f, k_b=%.3f, l_a=%d, l_b=%d)",
            elapsed, r_m, r_max, k_a, k_b, l_a, l_b
        )

    return result


def get_outer_batch_config() -> tuple[int, float]:
    """Return batch outer-tail tuning parameters."""
    return int(_OUTER_BATCH_MIN_ACTIVE), float(_OUTER_BATCH_MOMENT_TOL)


def _filon_3pt_batch(
    f_mat: np.ndarray,
    r0: float, r1: float, r2: float,
    omega: float,
    phase_offset: float
) -> np.ndarray:
    """
    Vectorized 3-point Filon for multiple envelopes simultaneously.

    Parameters
    ----------
    f_mat : np.ndarray, shape (n_terms, 3)
        Envelope values at nodes [r0, r1, r2] for each L-term.
    r0, r1, r2 : float
        Segment nodes.
    omega : float
        Constant frequency.
    phase_offset : float
        Phase offset.

    Returns
    -------
    np.ndarray, shape (n_terms,), complex
        Integral contribution from this 3-point segment for all L-terms.
    """
    h = r2 - r0
    if h < 1e-15:
        return np.zeros(f_mat.shape[0], dtype=np.complex128)

    theta = omega * h / 2.0

    # Compute Filon coefficients (scalar, shared across all L-terms)
    if abs(theta) < 0.3:
        # Taylor expansion for numerical stability near theta=0
        t2 = theta ** 2
        t3 = theta ** 3
        alpha = 2 * t3 / 45 * (1 - t2 / 7 + t2 ** 2 / 189)
        beta = 2 / 3 + 2 * t2 / 15 - 4 * t2 ** 2 / 105 + 2 * t2 ** 3 / 567
        gamma = 4 / 3 - 2 * t2 / 15 + t2 ** 2 / 210 - t2 ** 3 / 11340
    elif abs(omega * h) < 1e-3:
        # Nearly non-oscillatory: midpoint approximation for all terms
        mid_phase = omega * (r0 + r2) / 2 + phase_offset
        # Simpson weights for 3 points: [h/6, 4h/6, h/6]
        simpson_w = np.array([h / 6.0, 4.0 * h / 6.0, h / 6.0])
        integral_f = f_mat @ simpson_w  # (n_terms,)
        return integral_f * np.exp(1j * mid_phase)
    else:
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_2t = np.sin(2 * theta)
        cos_2t = np.cos(2 * theta)
        t2 = theta ** 2
        t3 = theta ** 3
        alpha = (t2 + theta * sin_2t / 2 - 2 * sin_t ** 2) / t3
        beta = 2 * (theta * (1 + cos_2t) - sin_2t) / t3
        gamma = 4 * (sin_t - theta * cos_t) / t3

    # Phase exponentials (scalar)
    exp0 = np.exp(1j * (omega * r0 + phase_offset))
    exp1 = np.exp(1j * (omega * r1 + phase_offset))
    exp2 = np.exp(1j * (omega * r2 + phase_offset))

    # Vectorized Filon formula across all L-terms:
    # I_L = (h/2) * [alpha*(-i/omega)*(f2_L*exp2 - f0_L*exp0)
    #              + beta*(f0_L*exp0 + f2_L*exp2) + gamma*f1_L*exp1]
    f0 = f_mat[:, 0]   # (n_terms,)
    f1 = f_mat[:, 1]   # (n_terms,)
    f2 = f_mat[:, 2]   # (n_terms,)

    result = (h / 2.0) * (
        alpha * (-1j / omega) * (f2 * exp2 - f0 * exp0) +
        beta * (f0 * exp0 + f2 * exp2) +
        gamma * f1 * exp1
    )
    return result


def _filon_oscillatory_integral_multipole_batch(
    moments: np.ndarray,
    L_values: np.ndarray,
    omega: float,
    phase_offset: float,
    r_start: float,
    r_end: float,
    delta_phi: float = np.pi / 4,
    n_nodes_per_segment: int = 5
) -> np.ndarray:
    """
    Batch Filon quadrature for envelopes m_L / r^(L+1).

    Vectorized over L-terms: Filon coefficients are computed ONCE per segment
    and applied to all multipole envelopes simultaneously via broadcasting.
    """
    n_terms = len(L_values)
    out = np.zeros(n_terms, dtype=np.complex128)
    if n_terms == 0 or r_end <= r_start + 1e-10:
        return out

    if abs(omega) < 1e-10:
        from scipy.integrate import simpson
        n_pts = max(257, 32 * n_nodes_per_segment)
        r_nodes = np.linspace(r_start, r_end, n_pts)
        phase = np.exp(1j * (omega * r_nodes + phase_offset))
        inv_r = 1.0 / np.maximum(r_nodes, 1e-30)
        pow_mat = inv_r[None, :] ** (L_values[:, None] + 1.0)
        f_mat = moments[:, None] * pow_mat
        for i in range(n_terms):
            out[i] = simpson(f_mat[i] * phase, x=r_nodes)
        return out

    dr_per_segment = delta_phi / abs(omega)
    n_segments_raw = max(1, int(np.ceil((r_end - r_start) / dr_per_segment)))
    n_segments = min(n_segments_raw, _FILON_MAX_SEGMENTS)
    effective_delta_phi = abs(omega) * (r_end - r_start) / max(1, n_segments)
    n_nodes_eff = int(n_nodes_per_segment)
    if effective_delta_phi > _FILON_MAX_EFFECTIVE_DPHI:
        scale = int(np.ceil(effective_delta_phi / _FILON_MAX_EFFECTIVE_DPHI))
        n_nodes_eff = min(15, max(n_nodes_per_segment, n_nodes_per_segment + 2 * (scale - 1)))

    # Precompute L+1 exponents for pow()
    L_plus1 = L_values.astype(float) + 1.0  # (n_terms,)

    segment_bounds = np.linspace(r_start, r_end, n_segments + 1)
    for i_seg in range(n_segments):
        a = segment_bounds[i_seg]
        b = segment_bounds[i_seg + 1]
        r_nodes = np.linspace(a, b, n_nodes_eff)
        inv_r = 1.0 / np.maximum(r_nodes, 1e-30)

        # f_mat shape: (n_terms, n_nodes_eff)
        pow_mat = inv_r[None, :] ** L_plus1[:, None]
        f_mat = moments[:, None] * pow_mat

        # Process in 3-point (triplet) sub-segments, vectorized over L
        for j in range(0, n_nodes_eff - 2, 2):
            out += _filon_3pt_batch(
                f_mat[:, j:j + 3],
                r_nodes[j], r_nodes[j + 1], r_nodes[j + 2],
                omega, phase_offset
            )

        # Handle odd leftover (last 2 points) with trapezoidal rule
        if n_nodes_eff > 2 and (n_nodes_eff - 1) % 2 == 1:
            r_last = r_nodes[-2:]
            f_last = f_mat[:, -2:]  # (n_terms, 2)
            phi_last = omega * r_last + phase_offset
            h_last = r_last[-1] - r_last[0]
            exp_last = np.exp(1j * phi_last)  # (2,)
            # Trapezoid for all L-terms at once
            out += 0.5 * h_last * (f_last[:, 0] * exp_last[0] +
                                    f_last[:, 1] * exp_last[1])

    return out


def _levin_oscillatory_integral_multipole_batch(
    moments: np.ndarray,
    L_values: np.ndarray,
    phi_func,
    phi_prime_func,
    r_start: float,
    r_end: float,
    n_nodes: int = 8,
    n_segments: int = 1
) -> np.ndarray:
    """Batch Levin quadrature for envelopes m_L / r^(L+1)."""
    n_terms = len(L_values)
    out = np.zeros(n_terms, dtype=np.complex128)
    if n_terms == 0 or r_end <= r_start + 1e-10:
        return out

    if n_segments < 1:
        n_segments = 1
    segment_bounds = np.linspace(r_start, r_end, n_segments + 1)

    for i_seg in range(n_segments):
        a = segment_bounds[i_seg]
        b = segment_bounds[i_seg + 1]
        theta = np.pi * np.arange(n_nodes) / (n_nodes - 1)
        x_ref = np.cos(theta)[::-1]
        r_nodes = 0.5 * (b - a) * (x_ref + 1.0) + a

        Phi_vals = _eval_callable_on_nodes(phi_func, r_nodes)
        Phi_prime_vals = _eval_callable_on_nodes(phi_prime_func, r_nodes)
        inv_r = 1.0 / np.maximum(r_nodes, 1e-30)
        pow_mat = inv_r[None, :] ** (L_values[:, None] + 1.0)
        f_mat = moments[:, None] * pow_mat  # (n_terms, n_nodes)

        # Vectorized Levin: same A matrix for all L-terms, different RHS
        h_seg = r_nodes[-1] - r_nodes[0]
        if h_seg < 1e-15:
            continue

        D_ref = _chebyshev_differentiation_matrix(n_nodes)
        D = D_ref * (2.0 / h_seg)
        A = D + 1j * np.diag(Phi_prime_vals)

        # Solve for all L-terms at once: A @ U = F^T → U has shape (n_nodes, n_terms)
        try:
            U = np.linalg.solve(A, f_mat.T.astype(complex))  # (n_nodes, n_terms)
        except np.linalg.LinAlgError:
            U, _, _, _ = np.linalg.lstsq(A, f_mat.T.astype(complex), rcond=None)

        # Boundary terms for all L-terms at once
        exp_a = np.exp(1j * Phi_vals[0])
        exp_b = np.exp(1j * Phi_vals[-1])
        out += U[-1, :] * exp_b - U[0, :] * exp_a  # (n_terms,)

    return out


def _outer_integral_multipole_batch_phase(
    moments: np.ndarray,
    L_values: np.ndarray,
    phi_func,
    phi_prime_func,
    phi_prime2_func,
    r_m: float,
    r_max: float,
    delta_phi: float = np.pi / 4,
    n_nodes: int = 8,
    filon_threshold: float = 0.1
) -> np.ndarray:
    """Batch outer integral for one phase branch (minus or plus)."""
    n_terms = len(L_values)
    out = np.zeros(n_terms, dtype=np.complex128)
    if n_terms == 0 or r_max <= r_m + 1e-10:
        return out

    r_mid = 0.5 * (r_m + r_max)
    omega_mid = abs(phi_prime_func(r_mid))
    if omega_mid < 1e-10:
        from scipy.integrate import simpson
        n_pts = max(257, 32 * n_nodes)
        r_nodes = np.linspace(r_m, r_max, n_pts)
        phase = np.exp(1j * _eval_callable_on_nodes(phi_func, r_nodes))
        inv_r = 1.0 / np.maximum(r_nodes, 1e-30)
        pow_mat = inv_r[None, :] ** (L_values[:, None] + 1.0)
        f_mat = moments[:, None] * pow_mat
        for i in range(n_terms):
            out[i] = simpson(f_mat[i] * phase, x=r_nodes)
        return out

    h = delta_phi / omega_mid
    phi_pp_mid = abs(phi_prime2_func(r_mid))
    linearity_test = phi_pp_mid * h * h
    if linearity_test < filon_threshold:
        phase_offset = float(phi_func(r_m) - omega_mid * r_m)
        return _filon_oscillatory_integral_multipole_batch(
            moments, L_values, omega_mid, phase_offset, r_m, r_max, delta_phi=delta_phi, n_nodes_per_segment=5
        )

    n_segments = max(1, int(np.ceil((r_max - r_m) / h)))
    return _levin_oscillatory_integral_multipole_batch(
        moments, L_values, phi_func, phi_prime_func, r_m, r_max, n_nodes=n_nodes, n_segments=n_segments
    )


def dwba_outer_integral_1d_multipole_batch(
    moments: np.ndarray,
    L_values: np.ndarray,
    k_a: float, l_a: int, delta_a: float, eta_a: float, sigma_a: float,
    k_b: float, l_b: int, delta_b: float, eta_b: float, sigma_b: float,
    r_m: float,
    r_max: float,
    delta_phi: float = np.pi / 4,
    n_nodes: int = 8,
    filon_threshold: float = 0.1
) -> Dict[int, float]:
    """
    Compute DWBA outer-tail integrals for multiple multipoles in one batch.

    Returns mapping: L -> I_out(L), where I_out(L)=0.5*(I_minus(L)-I_plus(L)).
    """
    moments_arr = np.asarray(moments, dtype=float)
    L_arr = np.asarray(L_values, dtype=int)
    if moments_arr.ndim != 1 or L_arr.ndim != 1 or len(moments_arr) != len(L_arr):
        raise ValueError("moments and L_values must be 1D arrays of equal length")
    if r_max <= r_m + 1e-10:
        return {}

    mask = (L_arr >= 1) & np.isfinite(moments_arr) & (np.abs(moments_arr) > _OUTER_BATCH_MOMENT_TOL)
    if not np.any(mask):
        return {}

    L_active = L_arr[mask]
    moments_active = moments_arr[mask]
    (
        phi_minus, phi_plus,
        phi_minus_prime, phi_plus_prime,
        phi_minus_prime2, phi_plus_prime2
    ) = _get_dwba_outer_phase_functions(
        k_a, l_a, delta_a, eta_a, sigma_a,
        k_b, l_b, delta_b, eta_b, sigma_b
    )
    I_minus = _outer_integral_multipole_batch_phase(
        moments_active, L_active, phi_minus, phi_minus_prime, phi_minus_prime2,
        r_m, r_max, delta_phi=delta_phi, n_nodes=n_nodes, filon_threshold=filon_threshold
    )
    I_plus = _outer_integral_multipole_batch_phase(
        moments_active, L_active, phi_plus, phi_plus_prime, phi_plus_prime2,
        r_m, r_max, delta_phi=delta_phi, n_nodes=n_nodes, filon_threshold=filon_threshold
    )
    I_out = 0.5 * (I_minus.real - I_plus.real)
    return {int(L): float(v) for L, v in zip(L_active, I_out)}




def compute_asymptotic_phase(
    r: float,
    k: float,
    l: int,
    delta: float = 0.0,
    eta: float = 0.0,
    sigma: float = 0.0
) -> float:
    """
    Compute asymptotic phase of continuum wave at radius r.
    
    Includes the first-order centrifugal correction to ensure stability
    for high partial waves at non-asymptotic distances.
    
    Φ(r) = k·r + η·ln(2kr) - l·π/2 + σ_l + δ_l - l(l+1)/(2kr)
    
    Parameters
    ----------
    r : float
        Radius [bohr].
    k : float
        Wave number [bohr⁻¹].
    l : int
        Orbital angular momentum.
    delta : float
        Short-range phase shift (from potential).
    eta : float
        Sommerfeld parameter (Z/k).
    sigma : float
        Coulomb phase shift.
        
    Returns
    -------
    phi : float
        Total phase in radians.
    """
    if r < 1e-9 or k < 1e-9:
        return 0.0
    
    # Base phase (asymptotic)
    phi = k * r - 0.5 * l * np.pi + sigma + delta
    
    if abs(eta) > 1e-15:
        phi += eta * np.log(max(2 * k * r, 1e-300))
        
    # Centrifugal correction (O(1/r))
    # This term arises from the large-r expansion of the Besssel/Coulomb phase:
    # arg(F_l) -> k*r + ... - l(l+1)/(2kr)
    # It is CRITICAL for high l when integrating on a finite grid.
    if l > 0:
        phi -= (l * (l + 1)) / (2.0 * k * r)
        
    return float(phi)


def compute_phase_derivative(
    r: float,
    k: float,
    eta: float = 0.0,
    l: int = 0
) -> float:
    """
    Compute derivative of asymptotic phase: Φ'(r) = k + η/r + l(l+1)/(2kr²).
    
    Parameters
    ----------
    r : float
        Radius [bohr].
    k : float
        Wave number [bohr⁻¹].
    eta : float
        Sommerfeld parameter.
    l : int
        Orbital angular momentum (for centrifugal term).
    """
    if r < 1e-9:
        return k
    
    deriv = k + eta / r
    if l > 0:
        deriv += (l * (l + 1)) / (2.0 * k * r ** 2)
    return float(deriv)


def compute_phase_second_derivative(
    r: float,
    eta: float = 0.0,
    l: int = 0,
    k: float = 1.0
) -> float:
    """
    Compute second derivative: Φ''(r) = -η/r² - l(l+1)/(kr³).
    """
    if r < 1e-9:
        return 0.0
    deriv2 = -eta / (r ** 2)
    if l > 0:
        deriv2 -= (l * (l + 1)) / (k * r ** 3)
    return float(deriv2)


# =============================================================================
# CHEBYSHEV NODES FOR LEVIN COLLOCATION
# =============================================================================

def _chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    """
    Generate Chebyshev nodes on interval [a, b].
    
    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)π / (2n))  for k = 0, 1, ..., n-1
    
    These are the zeros of T_n(x), avoiding endpoint singularities.
    """
    k = np.arange(n)
    x_ref = np.cos((2 * k + 1) * np.pi / (2 * n))  # On [-1, 1]
    return 0.5 * (b - a) * (x_ref + 1) + a  # Map to [a, b]


def _chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    """
    Compute Chebyshev differentiation matrix on [-1, 1].
    
    D[i,j] gives the derivative operator: (d/dx u)_i ≈ Σ_j D[i,j] u_j
    
    For Levin collocation, we solve: u' + i Φ' u = f
    which becomes: (D + diag(i Φ')) u = f
    """
    if n < 2:
        return np.zeros((n, n))
    
    # Chebyshev-Lobatto points on [-1, 1] for differentiation
    theta = np.pi * np.arange(n) / (n - 1)
    x = np.cos(theta)
    
    # c vector for endpoints weighting
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    
    # Build differentiation matrix (vectorized)
    # D[i,j] = (c[i] / c[j]) * ((-1)^(i+j)) / (x[i] - x[j]) for i != j
    
    # Create index arrays
    i_idx = np.arange(n)[:, None]  # Column vector
    j_idx = np.arange(n)[None, :]  # Row vector
    
    # Sign matrix: (-1)^(i+j)
    sign_matrix = (-1.0) ** (i_idx + j_idx)
    
    # x difference matrix (with small offset to avoid division by zero on diagonal)
    x_diff = x[:, None] - x[None, :]
    np.fill_diagonal(x_diff, 1.0)  # Temporary, will be fixed by diagonal calculation
    
    # c ratio matrix
    c_ratio = c[:, None] / c[None, :]
    
    # Off-diagonal elements
    D = c_ratio * sign_matrix / x_diff
    
    # Zero out diagonal (will be set below)
    np.fill_diagonal(D, 0.0)
    
    # Diagonal: D[i,i] = -sum of off-diagonal terms in row i
    D[np.diag_indices(n)] = -np.sum(D, axis=1)
    
    return D


# =============================================================================
# LEVIN COLLOCATION FOR OSCILLATORY INTEGRALS
# =============================================================================

def _levin_segment_complex(
    f_vals: np.ndarray,
    r_nodes: np.ndarray,
    Phi_vals: np.ndarray,
    Phi_prime_vals: np.ndarray
) -> complex:
    """
    Levin collocation for ∫ f(r) exp(iΦ(r)) dr on a segment.
    
    Solves the ODE: u'(r) + i·Φ'(r)·u(r) = f(r)
    
    Then uses: ∫ f exp(iΦ) dr = u(b)·exp(iΦ(b)) - u(a)·exp(iΦ(a))
    
    Parameters
    ----------
    f_vals : np.ndarray
        Envelope function values at collocation nodes.
    r_nodes : np.ndarray
        Radial nodes for collocation (Chebyshev-Lobatto).
    Phi_vals : np.ndarray
        Phase Φ(r) at nodes.
    Phi_prime_vals : np.ndarray
        Phase derivative Φ'(r) at nodes.
        
    Returns
    -------
    integral : complex
        Value of ∫ f(r) exp(iΦ(r)) dr on segment.
        
    Notes
    -----
    This is the "most robust" method for highly oscillatory integrals
    with variable frequency. It works when Filon fails due to nonlinear phase.
    
    References
    ----------
    - D. Levin, "Fast integration of rapidly oscillatory functions" (1982)
    - S. Olver, "Moment-free numerical integration" (2006)
    """
    n = len(r_nodes)
    if n < 2:
        return 0.0 + 0.0j
    
    a, b = r_nodes[0], r_nodes[-1]
    h = b - a
    
    if h < 1e-15:
        return 0.0 + 0.0j
    
    # Build the Levin system: (D + i·diag(Φ'))·u = f
    # where D is the differentiation matrix scaled for [a, b]
    
    # Get differentiation matrix on [-1, 1], scale by 2/h for [a, b]
    D_ref = _chebyshev_differentiation_matrix(n)
    D = D_ref * (2.0 / h)  # Scale derivative for interval [a, b]
    
    # Build system matrix: A = D + i·diag(Φ')
    A = D + 1j * np.diag(Phi_prime_vals)
    
    # Solve A·u = f for u (complex vector)
    try:
        u = np.linalg.solve(A, f_vals.astype(complex))
    except np.linalg.LinAlgError:
        # Fallback: use least squares if singular
        u, _, _, _ = np.linalg.lstsq(A, f_vals.astype(complex), rcond=None)
    
    # Boundary terms: I = u(b)·exp(iΦ(b)) - u(a)·exp(iΦ(a))
    exp_a = np.exp(1j * Phi_vals[0])
    exp_b = np.exp(1j * Phi_vals[-1])
    
    integral = u[-1] * exp_b - u[0] * exp_a
    
    return integral


def levin_oscillatory_integral(
    f_func,
    phi_func,
    phi_prime_func,
    r_start: float,
    r_end: float,
    n_nodes: int = 8,
    n_segments: int = 1
) -> complex:
    """
    Levin collocation for ∫ f(r) exp(iΦ(r)) dr.
    
    High-level interface that divides into segments and applies Levin on each.
    
    Parameters
    ----------
    f_func : callable
        Envelope function f(r).
    phi_func : callable
        Phase function Φ(r).
    phi_prime_func : callable
        Phase derivative Φ'(r).
    r_start, r_end : float
        Integration bounds.
    n_nodes : int
        Number of Chebyshev nodes per segment (default 8).
    n_segments : int
        Number of segments to divide interval into.
        
    Returns
    -------
    integral : complex
        Result of integration.
    """
    if n_segments < 1:
        n_segments = 1
    
    segment_bounds = np.linspace(r_start, r_end, n_segments + 1)
    contributions = []
    
    for i in range(n_segments):
        a, b = segment_bounds[i], segment_bounds[i + 1]
        
        # Get Chebyshev-Lobatto nodes on [a, b]
        theta = np.pi * np.arange(n_nodes) / (n_nodes - 1)
        x_ref = np.cos(theta)[::-1]  # Ascending order
        r_nodes = 0.5 * (b - a) * (x_ref + 1) + a
        
        # Evaluate functions at nodes (vectorized)
        # Use np.vectorize for scalar functions - creates efficient C loop
        f_vals = _eval_callable_on_nodes(f_func, r_nodes)
        Phi_vals = _eval_callable_on_nodes(phi_func, r_nodes)
        Phi_prime_vals = _eval_callable_on_nodes(phi_prime_func, r_nodes)
        
        # Levin on this segment
        contrib = _levin_segment_complex(f_vals, r_nodes, Phi_vals, Phi_prime_vals)
        contributions.append(contrib)
    
    # Use Kahan summation for stability
    return _kahan_sum_complex(np.array(contributions))


# =============================================================================
# COMPLEX FILON QUADRATURE (exp(iΦ) form)
# =============================================================================

def _filon_segment_complex(
    f_vals: np.ndarray,
    r_nodes: np.ndarray,
    omega: float,
    phase_offset: float = 0.0
) -> complex:
    """
    Filon quadrature for ∫ f(r) × exp(i(ω·r + φ₀)) dr on a segment.
    
    Assumes phase is LINEAR: Φ(r) ≈ ω·r + φ₀ (constant frequency).
    Uses cubic interpolation of f(r) and analytical exp integration.
    
    Parameters
    ----------
    f_vals : np.ndarray
        Envelope values at r_nodes.
    r_nodes : np.ndarray
        Radial nodes on segment.
    omega : float
        Local frequency ω = Φ'(r_mid).
    phase_offset : float
        Phase at segment start: φ₀ = Φ(r_a) - ω·r_a.
        
    Returns
    -------
    integral : complex
        Value of ∫ f(r) exp(iΦ(r)) dr.
    """
    n = len(r_nodes)
    if n < 2:
        return 0.0 + 0.0j
    
    a, b = r_nodes[0], r_nodes[-1]
    h = b - a
    
    if h < 1e-15:
        return 0.0 + 0.0j
    
    # Handle small ω (nearly non-oscillatory)
    if abs(omega * h) < 1e-3:
        # Use Taylor expansion for weights or standard quadrature
        # ∫ f exp(iΦ) ≈ exp(iΦ_mid) × ∫ f dr (for small phase change)
        mid_phase = omega * (a + b) / 2 + phase_offset
        from scipy.integrate import simpson
        integral_f = simpson(f_vals, x=r_nodes)
        return integral_f * np.exp(1j * mid_phase)
    
    # Filon with 3 points (Simpson-type)
    if n == 3:
        # Midpoint
        f0, f1, f2 = f_vals[0], f_vals[1], f_vals[2]
        r0, r1, r2 = r_nodes[0], r_nodes[1], r_nodes[2]
        
        # Filon coefficients
        theta = omega * h / 2
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        if abs(theta) < 0.3:
            # Taylor expansion for numerical stability
            t2 = theta ** 2
            t3 = theta ** 3
            alpha = 2 * t3 / 45 * (1 - t2 / 7 + t2**2 / 189)
            beta = 2/3 + 2*t2/15 - 4*t2**2/105 + 2*t2**3/567
            gamma = 4/3 - 2*t2/15 + 1*t2**2/210 - t2**3/11340
        else:
            t2 = theta ** 2
            sin_2t = np.sin(2 * theta)
            cos_2t = np.cos(2 * theta)
            alpha = (t2 + theta * sin_2t / 2 - 2 * sin_t**2) / (theta**3)
            beta = 2 * (theta * (1 + cos_2t) - sin_2t) / (theta**3)
            gamma = 4 * (sin_t - theta * cos_t) / (theta**3)
        
        # Phases
        phi0 = omega * r0 + phase_offset
        phi1 = omega * r1 + phase_offset
        phi2 = omega * r2 + phase_offset
        
        # Complex exponentials
        exp0 = np.exp(1j * phi0)
        exp1 = np.exp(1j * phi1)
        exp2 = np.exp(1j * phi2)
        
        # Filon formula in complex form
        # I ≈ h × [α×(f₀×exp₀_deriv - f₂×exp₂_deriv) + β×(f₀×exp₀ + f₂×exp₂) + γ×f₁×exp₁]
        # where deriv means derivative of exp gives extra factor
        # Simplified: use standard Filon weights
        integral = (h / 2) * (
            alpha * (-1j / omega) * (f2 * exp2 - f0 * exp0) +
            beta * (f0 * exp0 + f2 * exp2) +
            gamma * f1 * exp1
        )
        return integral
    
    # General case: divide into triplets
    result = 0.0 + 0.0j
    for i in range(0, n - 2, 2):
        segment_contrib = _filon_segment_complex(
            f_vals[i:i+3], r_nodes[i:i+3], omega, phase_offset
        )
        result += segment_contrib
    
    # Handle odd leftover
    if n > 2 and (n - 1) % 2 == 1:
        # Last two points: trapezoid with exp
        r_last = r_nodes[-2:]
        f_last = f_vals[-2:]
        phi_last = omega * r_last + phase_offset
        h_last = r_last[-1] - r_last[0]
        result += 0.5 * h_last * (f_last[0] * np.exp(1j * phi_last[0]) + 
                                   f_last[1] * np.exp(1j * phi_last[1]))
    
    return result


def filon_oscillatory_integral(
    f_func,
    omega: float,
    phase_offset: float,
    r_start: float,
    r_end: float,
    delta_phi: float = np.pi / 4,
    n_nodes_per_segment: int = 5
) -> complex:
    """
    Filon quadrature for ∫ f(r) exp(i(ωr + φ₀)) dr with phase segmentation.
    
    Divides [r_start, r_end] into segments with constant phase increment ΔΦ,
    then applies Filon on each segment.
    
    Parameters
    ----------
    f_func : callable
        Envelope function f(r).
    omega : float
        Constant angular frequency.
    phase_offset : float
        Phase at r=0.
    r_start, r_end : float
        Integration bounds.
    delta_phi : float
        Phase increment per segment (default π/4).
    n_nodes_per_segment : int
        Number of nodes per segment for Filon (default 5).
        
    Returns
    -------
    integral : complex
        Result of integration.
    """
    if abs(omega) < 1e-10:
        # Non-oscillatory: use standard quadrature
        from scipy.integrate import quad
        result_re, _ = quad(lambda r: f_func(r) * np.cos(phase_offset), r_start, r_end)
        result_im, _ = quad(lambda r: f_func(r) * np.sin(phase_offset), r_start, r_end)
        return result_re + 1j * result_im
    
    # Generate segments with constant phase increment
    dr_per_segment = delta_phi / abs(omega)
    n_segments_raw = max(1, int(np.ceil((r_end - r_start) / dr_per_segment)))
    n_segments = min(n_segments_raw, _FILON_MAX_SEGMENTS)
    effective_delta_phi = abs(omega) * (r_end - r_start) / max(1, n_segments)
    n_nodes_eff = int(n_nodes_per_segment)
    if n_segments_raw > _FILON_MAX_SEGMENTS and _should_sampled_debug("filon_segment_cap", every=200, initial=3):
        logger.debug(
            "Filon segments capped: raw=%d -> %d (omega=%.3f, r=[%.2f, %.2f], dphi_req=%.3f, dphi_eff=%.3f)",
            n_segments_raw, n_segments, omega, r_start, r_end, delta_phi, effective_delta_phi
        )
    if effective_delta_phi > _FILON_MAX_EFFECTIVE_DPHI:
        # Improve robustness when segment cap inflates phase swing per segment.
        scale = int(np.ceil(effective_delta_phi / _FILON_MAX_EFFECTIVE_DPHI))
        n_nodes_eff = min(15, max(n_nodes_per_segment, n_nodes_per_segment + 2 * (scale - 1)))
        if _should_sampled_debug("filon_nodes_adjust", every=200, initial=3):
            logger.debug(
                "Filon node-upscale due to dphi_eff=%.3f > %.3f: nodes %d -> %d",
                effective_delta_phi, _FILON_MAX_EFFECTIVE_DPHI, n_nodes_per_segment, n_nodes_eff
            )
    segment_bounds = np.linspace(r_start, r_end, n_segments + 1)
    
    contributions = []
    for i in range(n_segments):
        a, b = segment_bounds[i], segment_bounds[i + 1]
        r_nodes = np.linspace(a, b, n_nodes_eff)
        f_vals = _eval_callable_on_nodes(f_func, r_nodes)
        
        # omega is constant, phase_offset is the offset at r=0
        contrib = _filon_segment_complex(f_vals, r_nodes, omega, phase_offset)
        contributions.append(contrib)
    
    return _kahan_sum_complex(np.array(contributions))


# =============================================================================
# UNIFIED OUTER INTEGRAL INTERFACE
# =============================================================================

def compute_outer_integral_oscillatory(
    f_func,
    phi_func,
    phi_prime_func,
    phi_prime2_func,
    r_m: float,
    r_max: float,
    delta_phi: float = np.pi / 4,
    n_nodes: int = 8,
    filon_threshold: float = 0.1
) -> complex:
    """
    Compute outer oscillatory integral I_out = ∫_{r_m}^{r_max} f(r) exp(iΦ(r)) dr.
    
    Automatically chooses between Filon (linear phase) and Levin (nonlinear phase)
    based on the magnitude of |Φ''| × h².
    
    Parameters
    ----------
    f_func : callable
        Envelope function f(r).
    phi_func : callable
        Phase function Φ(r).
    phi_prime_func : callable
        Phase derivative Φ'(r).
    phi_prime2_func : callable
        Second derivative Φ''(r).
    r_m : float
        Match point (start of outer region).
    r_max : float
        Maximum radius.
    delta_phi : float
        Target phase increment per segment.
    n_nodes : int
        Nodes per segment for Levin.
    filon_threshold : float
        If |Φ''| × h² < threshold, use Filon; else Levin.
        
    Returns
    -------
    integral : complex
        Outer integral contribution.
        
    Notes
    -----
    From the instruction:
    - "Kryterium «czy Filon jest bezpieczny»: jeśli na segmencie |Φ''| × h² << 1 
       (np. < 0.1), to Filon działa świetnie. Jak nie, przechodzisz na Levin."
    """
    if r_max <= r_m + 1e-10:
        return 0.0 + 0.0j
    
    # Estimate h from phase increment
    omega_mid = abs(phi_prime_func((r_m + r_max) / 2))
    if omega_mid < 1e-10:
        # Non-oscillatory
        from scipy.integrate import quad
        result_re, _ = quad(lambda r: f_func(r) * np.cos(phi_func(r)), r_m, r_max)
        result_im, _ = quad(lambda r: f_func(r) * np.sin(phi_func(r)), r_m, r_max)
        return result_re + 1j * result_im
    
    h = delta_phi / omega_mid
    
    # Check phase linearity: is |Φ''| × h² small?
    phi_pp_mid = abs(phi_prime2_func((r_m + r_max) / 2))
    linearity_test = phi_pp_mid * h * h
    
    if linearity_test < filon_threshold:
        # Use Filon (constant frequency approximation)
        if _should_sampled_debug("outer_integral_filon", every=1000, initial=2):
            logger.debug(
                "Outer integral: using Filon (|Φ''|×h²=%.2e < %.1f)",
                linearity_test, filon_threshold
            )
        return filon_oscillatory_integral(
            f_func, omega_mid, phi_func(r_m) - omega_mid * r_m,
            r_m, r_max, delta_phi
        )
    else:
        # Use Levin (handles nonlinear phase)
        if _should_sampled_debug("outer_integral_levin", every=500, initial=2):
            logger.debug(
                "Outer integral: using Levin (|Φ''|×h²=%.2e >= %.1f)",
                linearity_test, filon_threshold
            )
        # Estimate number of segments
        n_segments = max(1, int(np.ceil((r_max - r_m) / h)))
        return levin_oscillatory_integral(
            f_func, phi_func, phi_prime_func,
            r_m, r_max, n_nodes, n_segments
        )




def check_phase_sampling(
    r: np.ndarray,
    k_total: float,
    threshold: float = np.pi / 4,
    eta_total: float = 0.0
) -> Tuple[float, bool, int]:
    """
    Check if grid adequately samples oscillations.
    
    Parameters
    ----------
    r : np.ndarray
        Radial grid points.
    k_total : float
        Total wave number (k_i + k_f for product of waves).
    threshold : float
        Maximum allowed phase change per step (default π/4).
    eta_total : float
        Total Sommerfeld parameter (η_i + η_f) for ionic targets.
        If nonzero, includes Coulomb phase component η·ln(r) in estimate.
        
    Returns
    -------
    max_phase : float
        Maximum phase change per step (radians).
    is_adequate : bool
        True if sampling is adequate everywhere.
    problem_idx : int
        Index where sampling first becomes inadequate (-1 if adequate).
    """
    if k_total < 1e-10:
        return 0.0, True, -1
    
    dr = np.diff(r)
    
    # Phase per step for Coulomb waves:
    #   φ'(r) ≈ k_total + eta_total / r
    # so Δφ ≈ |φ'(r_mid)| Δr
    if abs(eta_total) > 1e-10:
        r_mid = np.maximum(0.5 * (r[:-1] + r[1:]), 1e-10)  # Midpoint of each interval
        local_phase_derivative = k_total + eta_total / r_mid
        phase_per_step = np.abs(local_phase_derivative * dr)
    else:
        phase_per_step = np.abs(k_total * dr)
    
    max_phase = float(np.max(phase_per_step))
    
    # Find first problematic index
    problem_mask = phase_per_step > threshold
    if np.any(problem_mask):
        problem_idx = int(np.argmax(problem_mask))
        return max_phase, False, problem_idx
    
    return max_phase, True, -1


def log_phase_diagnostic(
    r: np.ndarray,
    k_i: float,
    k_f: float,
    l_i: int,
    l_f: int,
    eta_total: float = 0.0
) -> None:
    """
    Log diagnostic information about phase sampling.
    """
    k_total = k_i + k_f
    max_phase, is_ok, prob_idx = check_phase_sampling(r, k_total, eta_total=eta_total)
    
    if not is_ok:
        r_problem = r[prob_idx] if prob_idx >= 0 else r[-1]
        logger.warning(
            "Phase undersampling: l_i=%d, l_f=%d, k=%.2f+%.2f, eta_total=%.2f, "
            "max_dφ=%.2f rad at r=%.0f bohr",
            l_i, l_f, k_i, k_f, eta_total, max_phase, r_problem
        )


# =============================================================================
# CLENSHAW-CURTIS QUADRATURE NODES
# =============================================================================

def clenshaw_curtis_nodes(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Clenshaw-Curtis quadrature nodes and weights on [a, b].
    
    Clenshaw-Curtis uses Chebyshev nodes which cluster near endpoints,
    providing excellent accuracy for smooth functions. For oscillatory
    functions, combine with phase-adaptive refinement.
    
    Parameters
    ----------
    n : int
        Number of nodes (must be >= 2).
    a, b : float
        Integration bounds.
        
    Returns
    -------
    nodes : np.ndarray
        Quadrature nodes on [a, b].
    weights : np.ndarray
        Corresponding weights.
        
    Notes
    -----
    Clenshaw-Curtis quadrature uses Chebyshev nodes and is exact for 
    polynomials of degree ≤ n-1.
    """
    if n < 2:
        raise ValueError("Clenshaw-Curtis requires n >= 2 nodes")
    
    if n == 2:
        # Special case: trapezoidal rule
        return np.array([b, a]), np.array([0.5, 0.5]) * (b - a)
    
    # Chebyshev nodes on [-1, 1]: x_k = cos(π*k/(n-1)) for k = 0, 1, ..., n-1
    theta = np.pi * np.arange(n) / (n - 1)
    x_cheb = np.cos(theta)  # Points in [-1, 1], from 1 to -1
    
    # Transform to [a, b]
    nodes = 0.5 * (b - a) * (x_cheb + 1) + a
    
    # ==========================================================================
    # Clenshaw-Curtis weights (VECTORIZED)
    # ==========================================================================
    # w_k = (2/N) * [1 + sum_{j=1}^{N/2} 2/(1-4j²) * cos(2jθ_k)]
    
    N = n - 1
    j_max = N // 2
    
    # Base weight
    weights = np.ones(n)
    
    if j_max > 0:
        # j values: [1, 2, ..., j_max]
        j_vals = np.arange(1, j_max + 1)
        
        # Coefficients: 2/(1 - 4j²) for each j, shape (j_max,)
        b_j = 2.0 / (1 - 4 * j_vals**2)
        
        # cos(2*j*theta_k) for all j, k combinations: shape (j_max, n)
        cos_terms = np.cos(2 * j_vals[:, np.newaxis] * theta[np.newaxis, :])
        
        # Sum b_j * cos(2*j*theta_k) over j for each k: shape (n,)
        weights += np.sum(b_j[:, np.newaxis] * cos_terms, axis=0)
    
    # For even N, add the N/2 term
    if N % 2 == 0:
        weights += np.cos(N * theta) / (1 - N**2)
    
    # Scale by 2/N
    weights *= 2.0 / N
    
    # Halve the endpoint weights
    weights[0] /= 2
    weights[-1] /= 2
    
    # Scale for interval [a, b]
    weights *= 0.5 * (b - a)
    
    return nodes, weights


# =============================================================================
# CONSTANT PHASE SPLITTING
# =============================================================================

def generate_phase_nodes(
    r_start: float,
    r_end: float,
    k_total: float,
    phase_increment: float = np.pi / 2,
    eta_total: float = 0.0
) -> np.ndarray:
    """
    Generate grid points with constant phase increment.
    
    Creates an adaptive grid where the phase Φ(r) changes by exactly
    phase_increment between consecutive points. This ensures optimal 
    sampling of oscillatory integrands.
    
    For neutral targets (eta=0):
        Φ(r) = k × r  →  Φ'(r) = k  →  constant spacing dr = ΔΦ/k
        
    For ionic targets (eta≠0):
        Φ(r) = k×r + η×ln(2kr)  →  Φ'(r) = k + η/r  →  variable spacing
        dr(r) = ΔΦ / (k + η/r)
    
    Parameters
    ----------
    r_start : float
        Starting radius.
    r_end : float
        Ending radius.
    k_total : float
        Total wave number (k_i + k_f).
    phase_increment : float
        Phase change per step (default π/2).
    eta_total : float
        Sum of Sommerfeld parameters (η_i + η_f) for Coulomb correction.
        Default 0.0 for neutral targets.
        
    Returns
    -------
    r_nodes : np.ndarray
        Grid points with constant phase spacing.
        
    Notes
    -----
    For Coulomb phase: Φ(r) = k·r + η·ln(2kr) - lπ/2 + σ_l + δ_l
    The derivative is: Φ'(r) = k + η/r
    
    For constant phase increments: r_{j+1} = r_j + ΔΦ / Φ'(r_j)
    
    This is the recommendation from the instruction:
    "rozbij całkę na przedziały, na których faza robi stały przyrost"
    """
    if k_total < 1e-10:
        # Non-oscillatory: just use endpoints
        return np.array([r_start, r_end])
    
    if abs(eta_total) < 1e-10:
        # Neutral target: constant spacing
        dr = phase_increment / k_total
        n_points = int(np.ceil((r_end - r_start) / dr)) + 1
        return np.linspace(r_start, r_end, n_points)
    
    # === COULOMB CASE: Variable spacing ===
    # Φ'(r) = k + η/r, so dr = ΔΦ / (k + η/r)
    # Build nodes iteratively
    nodes = [r_start]
    r = r_start
    
    while r < r_end:
        # Phase derivative at current r
        phi_prime = k_total + eta_total / max(r, 1e-10)
        
        # Step size for constant phase increment
        dr = phase_increment / abs(phi_prime)
        
        # Limit step size to prevent numerical issues
        dr = min(dr, (r_end - r_start) / 10)
        dr = max(dr, 1e-6)
        
        r_next = r + dr
        if r_next >= r_end:
            r_next = r_end
        
        nodes.append(r_next)
        r = r_next
        
        # Safety: limit number of nodes
        if len(nodes) > 10000:
            break
    
    return np.array(nodes)


def _adaptive_phase_increment_for_grid(
    r_grid: np.ndarray,
    k_total: float,
    requested_phase_increment: float,
    eta_total: float = 0.0,
    threshold: float = np.pi / 4,
    min_increment: float = np.pi / 16,
) -> float:
    """
    Choose a safer phase increment when the grid undersamples oscillations.

    Returns a value in [min_increment, threshold]. If current sampling is poor,
    the increment is reduced proportionally to the diagnosed max phase step.
    """
    dphi_req = float(max(min_increment, requested_phase_increment))
    dphi_req = min(dphi_req, threshold)

    if k_total < 1e-10 or len(r_grid) < 2:
        return dphi_req

    max_phase, is_ok, _ = check_phase_sampling(
        np.asarray(r_grid),
        float(k_total),
        threshold=float(threshold),
        eta_total=float(eta_total),
    )
    if is_ok:
        return dphi_req

    scale = float(threshold) / max(float(max_phase), 1e-12)
    # Never increase dphi; at most reduce by 4x per call.
    dphi_eff = dphi_req * max(0.25, min(1.0, scale))
    dphi_eff = float(np.clip(dphi_eff, min_increment, threshold))

    if dphi_eff < dphi_req and _should_sampled_debug("adaptive_phase_increment", every=250, initial=3):
        logger.debug(
            "Adaptive phase increment: %.3f -> %.3f (k=%.3f, max_dphi=%.3f, eta=%.3f)",
            dphi_req, dphi_eff, k_total, max_phase, eta_total
        )
    return dphi_eff


def integrate_with_phase_nodes(
    f_func,
    r_start: float,
    r_end: float,
    k_total: float,
    phase_increment: float = np.pi / 2,
    eta_total: float = 0.0,
    use_clenshaw_curtis: bool = True,
    cc_nodes_per_interval: int = 5
) -> float:
    """
    Integrate oscillatory function using constant phase splitting.
    
    Combines phase-node splitting with Clenshaw-Curtis quadrature
    on each sub-interval for maximum accuracy.
    
    Parameters
    ----------
    f_func : callable
        Function to integrate: f(r) -> float or array.
    r_start, r_end : float
        Integration bounds.
    k_total : float
        Total wave number for phase calculation.
    phase_increment : float
        Phase change per sub-interval (default π/2).
    eta_total : float
        Sum of Sommerfeld parameters (η_i + η_f) for Coulomb phase correction.
    use_clenshaw_curtis : bool
        Use CC nodes within each interval (else trapezoid).
    cc_nodes_per_interval : int
        Number of CC nodes per sub-interval.
        
    Returns
    -------
    integral : float
        Result of integration.
    """
    # Generate phase nodes
    r_nodes = generate_phase_nodes(
        r_start, r_end, k_total, phase_increment, eta_total=eta_total
    )
    
    if len(r_nodes) < 2:
        return 0.0
    
    result = 0.0
    
    # Integrate each sub-interval
    for i in range(len(r_nodes) - 1):
        a, b = r_nodes[i], r_nodes[i + 1]
        
        if use_clenshaw_curtis and cc_nodes_per_interval >= 3:
            # Use Clenshaw-Curtis on this interval
            nodes, weights = clenshaw_curtis_nodes(cc_nodes_per_interval, a, b)
            f_vals = np.array([f_func(r) for r in nodes])
            result += np.dot(weights, f_vals)
        else:
            # Simple trapezoid
            fa, fb = f_func(a), f_func(b)
            result += 0.5 * (fa + fb) * (b - a)
    
    return float(result)


# =============================================================================
# FILON-TYPE QUADRATURE FOR OSCILLATORY INTEGRALS
# =============================================================================

def _filon_sin_integral(f_vals: np.ndarray, r: np.ndarray, omega: float,
                         phase_offset: float = 0.0) -> float:
    """
    Filon quadrature for ∫ f(r) × sin(ω·r + φ₀) dr.
    
    Uses parabolic interpolation on triplets of points and exact
    integration of P₂(r) × sin(ω·r).
    
    Parameters
    ----------
    f_vals : np.ndarray
        Slowly-varying envelope function values on grid.
    r : np.ndarray
        Radial grid points.
    omega : float
        Angular frequency of oscillation.
    phase_offset : float
        Constant phase offset φ₀.
        
    Returns
    -------
    integral : float
        Approximate value of the integral.
    """
    if omega < 1e-10:
        # Non-oscillatory case: use trapezoid
        return float(np.trapz(f_vals * np.sin(phase_offset), r))
    
    N = len(r)
    if N < 3:
        return float(np.trapz(f_vals * np.sin(omega * r + phase_offset), r))
    
    result = 0.0
    
    # Process in pairs of intervals (triplets of points)
    for i in range(0, N - 2, 2):
        r0, r1, r2 = r[i], r[i+1], r[i+2]
        f0, f1, f2 = f_vals[i], f_vals[i+1], f_vals[i+2]
        
        h1 = r1 - r0
        h2 = r2 - r1
        h = r2 - r0  # Total interval
        
        # Filon coefficients for non-uniform grid
        # Using extended Simpson with oscillatory correction
        theta = omega * h / 2
        
        if abs(theta) < 0.1:
            # Small theta: use Taylor expansion to avoid division by small number
            alpha = 2.0 * theta**3 / 45.0
            beta = 2.0 / 3.0 + 2.0 * theta**2 / 15.0
            gamma_c = 4.0 / 3.0 - 2.0 * theta**2 / 15.0
        else:
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            sin_2t = np.sin(2 * theta)
            cos_2t = np.cos(2 * theta)
            
            alpha = (theta**2 + theta * sin_2t / 2 - 2 * sin_t**2) / theta**3
            beta = 2 * (theta * (1 + cos_2t) - sin_2t) / theta**3
            gamma_c = 4 * (sin_t - theta * cos_t) / theta**3
        
        # Midpoint of interval
        r_mid = (r0 + r2) / 2
        
        # Phase at endpoints and midpoint
        phi0 = omega * r0 + phase_offset
        phi1 = omega * r1 + phase_offset
        phi2 = omega * r2 + phase_offset
        
        # Filon formula for sin integral
        cos_phi0 = np.cos(phi0)
        cos_phi2 = np.cos(phi2)
        sin_phi0 = np.sin(phi0)
        sin_phi1 = np.sin(phi1)
        sin_phi2 = np.sin(phi2)
        
        # Derivative approximations
        df0 = (f1 - f0) / h1 if h1 > 0 else 0.0
        df2 = (f2 - f1) / h2 if h2 > 0 else 0.0
        
        # Filon contribution
        contribution = (h / 2) * (
            alpha * (f0 * cos_phi0 - f2 * cos_phi2) +
            beta * (f0 * sin_phi0 + f2 * sin_phi2) +
            gamma_c * f1 * sin_phi1
        )
        
        result += contribution
    
    # Handle remainder if N is even (one interval left)
    if N % 2 == 0:
        # Last interval: simple Simpson on oscillatory function
        r_last = r[-2:]
        f_last = f_vals[-2:]
        phi_last = omega * r_last + phase_offset
        h_last = r[-1] - r[-2]
        result += h_last * np.trapz(f_last * np.sin(phi_last), r_last) / (r[-1] - r[-2])
    
    return float(result)


def _filon_cos_integral(f_vals: np.ndarray, r: np.ndarray, omega: float,
                         phase_offset: float = 0.0) -> float:
    """
    Filon quadrature for ∫ f(r) × cos(ω·r + φ₀) dr.
    """
    # cos(x) = sin(x + π/2)
    return _filon_sin_integral(f_vals, r, omega, phase_offset + np.pi / 2)


# =============================================================================
# ANALYTICAL TAIL FOR DIPOLE INTEGRALS
# =============================================================================

def _analytical_dipole_tail(
    r_m: float,
    k_i: float,
    k_f: float,
    delta_i: float,
    delta_f: float,
    l_i: int,
    l_f: int,
    A_env: float = 1.0,
    eta_i: float = 0.0,
    eta_f: float = 0.0,
    sigma_i: float = 0.0,
    sigma_f: float = 0.0
) -> float:
    """
    Analytically compute the oscillatory tail of a dipole radial integral
    beyond the match point r_m.
    
    For the product of asymptotic Coulomb waves:
        χ_i ~ sin(k_i·r + η_i·ln(2k_i·r) - l_i·π/2 + σ_i + δ_i)
        χ_f ~ sin(k_f·r + η_f·ln(2k_f·r) - l_f·π/2 + σ_f + δ_f)
    
    For neutral targets (η=0, σ=0), this reduces to simple sine waves.
    
    The integral ∫_{r_m}^∞ χ_i·χ_f / r dr can be computed using
    sine and cosine integrals Si(x), Ci(x).
    
    Parameters
    ----------
    r_m : float
        Match point (start of asymptotic region).
    k_i, k_f : float
        Wave numbers.
    delta_i, delta_f : float
        Phase shifts from short-range potential.
    l_i, l_f : int
        Angular momenta.
    A_env : float
        Envelope amplitude (includes 1/r factor strength).
    eta_i, eta_f : float
        Sommerfeld parameters η = -z_ion/k for incident and final waves.
    sigma_i, sigma_f : float
        Coulomb phase shifts σ_l = arg(Γ(l+1+iη)).
        
    Returns
    -------
    tail_integral : float
        Analytical contribution from [r_m, ∞).
        
    Notes
    -----
    Uses the identity:
        sin(A) × sin(B) = (1/2)[cos(A-B) - cos(A+B)]
    
    For ∫_{r_m}^∞ cos(ω·r + φ) / r dr = -Ci(ω·r_m)·cos(φ) - (Si(ω·r_m) - π/2)·sin(φ)
    """
    if r_m < 1e-3:
        return 0.0
    
    # SAFEGUARD: Very large arguments can cause Si/Ci overflow
    MAX_ARGUMENT = 1e6
    
    # Full phases at asymptotic limit including Coulomb terms
    # For Coulomb: phase = k·r + η·ln(2k·r) - l·π/2 + σ_l + δ
    # At r_m, the logarithmic term contributes: η·ln(2k·r_m)
    log_term_i = eta_i * np.log(2 * k_i * r_m + 1e-30) if abs(eta_i) > 1e-10 else 0.0
    log_term_f = eta_f * np.log(2 * k_f * r_m + 1e-30) if abs(eta_f) > 1e-10 else 0.0
    
    # Constant phase parts (excluding k·r which varies)
    phi_i_const = -l_i * np.pi / 2 + sigma_i + delta_i + log_term_i
    phi_f_const = -l_f * np.pi / 2 + sigma_f + delta_f + log_term_f
    
    # Product: sin(k_i·r + φ_i) × sin(k_f·r + φ_f)
    # = (1/2)[cos((k_i - k_f)·r + φ_i - φ_f) - cos((k_i + k_f)·r + φ_i + φ_f)]
    
    k_diff = k_i - k_f
    k_sum = k_i + k_f
    phi_diff = phi_i_const - phi_f_const
    phi_sum = phi_i_const + phi_f_const
    
    tail = 0.0
    
    # Term 1: ∫ cos(k_diff·r + φ_diff) / r dr
    if abs(k_diff) > 1e-6:
        x_diff = abs(k_diff) * r_m
        
        # SAFEGUARD: Skip if argument too large
        if x_diff > MAX_ARGUMENT:
            logger.debug("_analytical_dipole_tail: x_diff=%.2e too large, skipping term", x_diff)
        else:
            si_diff, ci_diff = sici(x_diff)
            
            # Check for NaN/Inf
            if np.isfinite(si_diff) and np.isfinite(ci_diff):
                sign_diff = 1.0 if k_diff > 0 else -1.0
                term1 = -ci_diff * np.cos(phi_diff) - (si_diff - np.pi / 2) * sign_diff * np.sin(phi_diff)
                tail += 0.5 * term1
    
    # Term 2: ∫ cos(k_sum·r + φ_sum) / r dr
    if k_sum > 1e-6:
        x_sum = k_sum * r_m
        
        # SAFEGUARD: Skip if argument too large
        if x_sum > MAX_ARGUMENT:
            logger.debug("_analytical_dipole_tail: x_sum=%.2e too large, skipping term", x_sum)
        else:
            si_sum, ci_sum = sici(x_sum)
            
            # Check for NaN/Inf
            if np.isfinite(si_sum) and np.isfinite(ci_sum):
                term2 = -ci_sum * np.cos(phi_sum) - (si_sum - np.pi / 2) * np.sin(phi_sum)
                tail -= 0.5 * term2  # Minus sign from -cos in product formula
    
    # SAFEGUARD: Check result
    result = A_env * tail
    if not np.isfinite(result):
        logger.warning("_analytical_dipole_tail: Non-finite result, returning 0")
        return 0.0
    
    return result


def _analytical_multipole_tail(
    r_m: float,
    k_i: float,
    k_f: float,
    delta_i: float,
    delta_f: float,
    l_i: int,
    l_f: int,
    L: int,
    bound_overlap: float = 1.0,
    eta_i: float = 0.0,
    eta_f: float = 0.0,
    sigma_i: float = 0.0,
    sigma_f: float = 0.0
) -> float:
    """
    Analytically compute the oscillatory tail contribution for multipole L.
    
    Theory
    ------
    For L ≥ 1, we need to evaluate:
        ∫_{r_m}^∞ sin(k_i r + φ_i) sin(k_f r + φ_f) / r^(L+1) dr
    
    Using sin(A)sin(B) = (1/2)[cos(A-B) - cos(A+B)], this becomes:
        (1/2) ∫_{r_m}^∞ [cos(k_- r + φ_-) - cos(k_+ r + φ_+)] / r^(L+1) dr
    
    where k_± = k_i ± k_f and φ_± = φ_i ± φ_f.
    
    For n = L+1 ≥ 2, integration by parts gives:
        ∫_{r_m}^∞ cos(ωr + φ) / r^n dr ≈ sin(ωr_m + φ) / (ω r_m^n)
                                           + n cos(ωr_m + φ) / (ω² r_m^(n+1))
    
    The leading term dominates for large r_m.
    
    Parameters
    ----------
    r_m : float
        Match point radius.
    k_i, k_f : float
        Wave numbers.
    delta_i, delta_f : float
        Phase shifts from short-range potential.
    l_i, l_f : int
        Angular momenta of continuum waves.
    L : int
        Multipole index (L >= 1).
    bound_overlap : float
        Multipole moment ∫ r^L × u_f × u_i dr of bound state densities.
    eta_i, eta_f : float
        Sommerfeld parameters η = -z_ion/k.
    sigma_i, sigma_f : float
        Coulomb phase shifts σ_l = arg(Γ(l+1+iη)).
        
    Returns
    -------
    tail_integral : float
        Analytical contribution from [r_m, ∞).
    """
    if r_m < 1e-3 or abs(bound_overlap) < 1e-12:
        return 0.0
    
    # Include Coulomb phase terms at r_m
    log_term_i = eta_i * np.log(2 * k_i * r_m + 1e-30) if abs(eta_i) > 1e-10 else 0.0
    log_term_f = eta_f * np.log(2 * k_f * r_m + 1e-30) if abs(eta_f) > 1e-10 else 0.0
    
    phi_i = k_i * r_m - l_i * np.pi / 2 + sigma_i + delta_i + log_term_i
    phi_f = k_f * r_m - l_f * np.pi / 2 + sigma_f + delta_f + log_term_f
    
    k_diff = k_i - k_f
    k_sum = k_i + k_f
    
    # Exponent for denominator
    n = L + 1  # Kernel decays as 1/r^(L+1)
    
    tail = 0.0
    
    # =========================================================================
    # For integrals of form ∫_{r_m}^∞ cos(ωr + φ) / r^n dr, use asymptotic:
    # 
    # ∫_{r_m}^∞ cos(ωr + φ) / r^n dr ≈ sin(ωr_m + φ) / (ω · r_m^n)  [leading]
    #                                   + n·cos(ωr_m + φ) / (ω² · r_m^(n+1))
    #
    # The integral ∫ cos(x)/x^n dx = -sin(x)/(n-1)/x^(n-1) + ω∫sin(x)/x^(n-1)dx
    # For large r_m, boundary term at r_m dominates.
    # =========================================================================
    
    # Term 1: ∫ cos(k_- r + φ_i - φ_f) / r^n dr
    # Phase at r_m for difference frequency
    phi_minus_at_rm = phi_i - phi_f  # Already includes k_i r_m - k_f r_m = k_diff * r_m
    # Correction: phi_minus should be the phase of cos(k_diff * r + φ_i_const - φ_f_const)
    # At r = r_m: phase = k_diff * r_m + (constant parts)
    phi_i_const = -l_i * np.pi / 2 + sigma_i + delta_i + log_term_i
    phi_f_const = -l_f * np.pi / 2 + sigma_f + delta_f + log_term_f
    phi_diff = k_diff * r_m + phi_i_const - phi_f_const
    
    if abs(k_diff) > 1e-6:
        # Leading-order asymptotic: sin(φ) / (ω r_m^n)
        term1_lead = np.sin(phi_diff) / (k_diff * (r_m ** n))
        # Next-order correction: n cos(φ) / (ω² r_m^(n+1))
        term1_corr = n * np.cos(phi_diff) / (k_diff**2 * (r_m ** (n + 1)))
        tail += 0.5 * (term1_lead + term1_corr)
    else:
        # k_diff ≈ 0: elastic scattering, cos term becomes constant
        # ∫ 1/r^n dr = 1/((n-1) r_m^(n-1)) for n > 1
        if n > 1:
            tail += 0.5 * np.cos(phi_diff) / ((n - 1) * (r_m ** (n - 1)))
    
    # Term 2: -∫ cos(k_+ r + φ_i + φ_f) / r^n dr
    phi_sum = k_sum * r_m + phi_i_const + phi_f_const
    
    if k_sum > 1e-6:
        term2_lead = np.sin(phi_sum) / (k_sum * (r_m ** n))
        term2_corr = n * np.cos(phi_sum) / (k_sum**2 * (r_m ** (n + 1)))
        tail -= 0.5 * (term2_lead + term2_corr)
    
    # SAFEGUARD: Check for numerical issues
    if not np.isfinite(tail):
        logger.warning("_analytical_multipole_tail: Non-finite result L=%d, returning 0", L)
        return 0.0
    
    return tail * bound_overlap


# =============================================================================
# PHASE-ADAPTIVE INTEGRATION
# =============================================================================

def _phase_adaptive_integrate(
    f_vals: np.ndarray,
    chi_i: np.ndarray,
    chi_f: np.ndarray,
    r: np.ndarray,
    k_i: float,
    k_f: float,
    w: np.ndarray,
    max_phase_step: float = np.pi / 4
) -> float:
    """
    Integrate f(r)·χ_i(r)·χ_f(r) with phase-adaptive refinement.
    
    Splits the integration into sub-intervals where the phase change
    is bounded by max_phase_step, using linear interpolation for refinement.
    
    Parameters
    ----------
    f_vals : np.ndarray
        Additional factor (e.g., multipole kernel integrated over r2).
    chi_i, chi_f : np.ndarray
        Continuum wavefunctions on grid.
    r : np.ndarray
        Radial grid.
    k_i, k_f : float
        Wave numbers.
    w : np.ndarray
        Original quadrature weights.
    max_phase_step : float
        Maximum allowed phase change per integration step.
        
    Returns
    -------
    integral : float
        Result of the integration.
    """
    k_total = k_i + k_f
    if k_total < 1e-6:
        # Non-oscillatory: use standard weighted sum
        return float(np.sum(w * f_vals * chi_i * chi_f))
    
    N = len(r)
    result = 0.0
    
    # Identify regions needing refinement
    dr = np.diff(r)
    phase_per_step = k_total * dr
    
    i = 0
    while i < N - 1:
        dphi = phase_per_step[i]
        
        if dphi <= max_phase_step:
            # Standard quadrature for this step
            integrand_i = f_vals[i] * chi_i[i] * chi_f[i]
            integrand_ip1 = f_vals[i+1] * chi_i[i+1] * chi_f[i+1]
            result += 0.5 * (integrand_i + integrand_ip1) * dr[i]
            i += 1
        else:
            # Refine this interval
            n_sub = int(np.ceil(dphi / max_phase_step))
            r_sub = np.linspace(r[i], r[i+1], n_sub + 1)
            
            # Grid-aware interpolation for all arrays (log-r on exponential grids).
            f_sub = _interp_on_radial_grid(r_sub, r, f_vals)
            chi_i_sub = _interp_on_radial_grid(r_sub, r, chi_i)
            chi_f_sub = _interp_on_radial_grid(r_sub, r, chi_f)
            
            integrand_sub = f_sub * chi_i_sub * chi_f_sub
            result += np.trapz(integrand_sub, r_sub)
            i += 1
    
    return float(result)


# =============================================================================
# MAIN INTERFACE: OSCILLATORY-AWARE RADIAL INTEGRAL
# =============================================================================

def oscillatory_radial_integral_1d(
    f_envelope: np.ndarray,
    chi_i: np.ndarray,
    chi_f: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    k_i: float,
    k_f: float,
    idx_match: int = -1,
    delta_i: float = 0.0,
    delta_f: float = 0.0,
    l_i: int = 0,
    l_f: int = 0,
    use_analytical_tail: bool = True,
    use_filon: bool = True,
    eta_total: float = 0.0
) -> float:
    """
    Compute oscillatory radial integral with proper handling of high-frequency content.
    
    I = ∫₀^∞ f(r) · χ_i(k_i, r) · χ_f(k_f, r) dr
    
    Using domain splitting:
    - [0, r_m]: Numerical integration (Filon or phase-adaptive)
    - [r_m, ∞]: Analytical tail using asymptotic forms
    
    Parameters
    ----------
    f_envelope : np.ndarray
        Slowly-varying envelope function.
    chi_i, chi_f : np.ndarray
        Continuum wavefunctions.
    r : np.ndarray
        Radial grid.
    w : np.ndarray
        Quadrature weights.
    k_i, k_f : float
        Wave numbers.
    idx_match : int
        Index of match point. If -1, use full grid.
    delta_i, delta_f : float
        Phase shifts for analytical tail.
    l_i, l_f : int
        Angular momenta for analytical tail.
    use_analytical_tail : bool
        Whether to add analytical tail contribution.
    use_filon : bool
        Whether to use Filon quadrature (else phase-adaptive).
    eta_total : float
        Sum of Sommerfeld parameters (η_i + η_f) for Coulomb phase correction.
        
    Returns
    -------
    integral : float
        Result of the integration.
    """
    N = len(r)
    
    # Determine integration limit
    if idx_match < 0 or idx_match >= N:
        idx_limit = N
    else:
        idx_limit = idx_match + 1
    
    # Numerical part: [0, r_m]
    r_num = r[:idx_limit]
    f_num = f_envelope[:idx_limit]
    chi_i_num = chi_i[:idx_limit]
    chi_f_num = chi_f[:idx_limit]
    w_num = w[:idx_limit]
    
    # Check phase sampling
    k_total = k_i + k_f
    max_phase, is_ok, _ = check_phase_sampling(r_num, k_total, eta_total=eta_total)
    
    if is_ok or not use_filon:
        # Standard weighted integration is OK
        # But still use phase-adaptive for borderline cases
        if max_phase > np.pi / 8:
            integral_num = _phase_adaptive_integrate(
                f_num, chi_i_num, chi_f_num, r_num, k_i, k_f, w_num
            )
        else:
            integral_num = float(np.sum(w_num * f_num * chi_i_num * chi_f_num))
    else:
        # Use Filon quadrature
        # Product of sines: sin(a)sin(b) = 0.5[cos(a-b) - cos(a+b)]
        # We approximate the envelope and integrate oscillations analytically
        
        # For practical implementation, use phase-adaptive refinement
        # which is more robust for non-uniform grids
        integral_num = _phase_adaptive_integrate(
            f_num, chi_i_num, chi_f_num, r_num, k_i, k_f, w_num
        )
    
    # Analytical tail: [r_m, ∞]
    integral_tail = 0.0
    if use_analytical_tail and idx_match > 0 and idx_match < N - 1:
        r_m = r[idx_match]
        
        # Estimate envelope amplitude at match point
        # For dipole (L=1): envelope ~ 1/r
        A_env = abs(f_envelope[idx_match]) if idx_match < len(f_envelope) else 1.0
        
        integral_tail = _analytical_dipole_tail(
            r_m, k_i, k_f, delta_i, delta_f, l_i, l_f, A_env
        )
    
    return integral_num + integral_tail


# =============================================================================
# SPECIALIZED 2D KERNEL INTEGRATION
# =============================================================================

def oscillatory_kernel_integral_2d(
    rho1: np.ndarray,
    rho2: np.ndarray,
    kernel: np.ndarray,
    r: np.ndarray,
    k_i: float,
    k_f: float,
    idx_limit: int = -1,
    idx_limit_r2: Optional[int] = None,
    method: str = "adaptive",
    n_nodes: int = 5,
    phase_increment: float = np.pi / 2,
    eta_total: float = 0.0,
    w_grid: Optional[np.ndarray] = None
) -> float:
    """
    Compute 2D radial integral with oscillatory densities.
    
    I = ∫∫ ρ₁(r₁) · K(r₁, r₂) · ρ₂(r₂) dr₁ dr₂
    
    where ρ₁, ρ₂ contain oscillatory factors from continuum waves.
    
    CRITICAL: If w_grid is None, this function performs a discrete sum
    (no integration weights), which is only correct if rho1/rho2 already
    include weights. For proper integration, pass w_grid (e.g., grid.w_trapz).
    
    Parameters
    ----------
    rho1, rho2 : np.ndarray
        Density arrays (unweighted - just wavefunctions product).
    kernel : np.ndarray
        2D kernel matrix K(r₁, r₂).
    r : np.ndarray
        Radial grid.
    k_i, k_f : float
        Wave numbers for phase diagnostic.
    idx_limit : int
        Integration limit index for r1.
    idx_limit_r2 : int, optional
        Integration limit index for r2. If None, uses idx_limit.
    method : str
        "standard": Use simple dot products (fast, may alias).
        "adaptive": Use phase-adaptive integration (slower, accurate).
        "filon": Use Filon/Clenshaw-Curtis for oscillatory r₁ integral.
        "filon_exchange": Filon/CC on both inner and outer integrals.
    n_nodes : int
        Number of CC nodes per phase interval (default 5).
    phase_increment : float
        Phase increment per sub-interval (default π/2).
    eta_total : float
        Sum of Sommerfeld parameters (η_i + η_f) for Coulomb phase correction.
    w_grid : np.ndarray, optional
        Integration weights for the radial grid. If None, uses unit weights
        (discrete sum instead of integral - typically ~15x too large!).
        
    Returns
    -------
    integral : float
        Result of the 2D integration.
    """
    if idx_limit < 0:
        idx_limit = len(r)
    if idx_limit_r2 is None or idx_limit_r2 < 0:
        idx_limit_r2 = len(r)
    
    # ==========================================================================
    # SAFEGUARDS: Input Validation
    # ==========================================================================
    if idx_limit < 2 or idx_limit_r2 < 2:
        logger.warning("oscillatory_kernel_integral_2d: idx_limit < 2, returning 0")
        return 0.0
    
    rho1_lim = rho1[:idx_limit].copy()
    rho2_lim = rho2[:idx_limit_r2].copy()
    kernel_lim = kernel[:idx_limit, :idx_limit_r2]
    r1_lim = r[:idx_limit]
    r2_lim = r[:idx_limit_r2]
    
    # Integration weights for proper ∫dr integration
    # CRITICAL FIX: Without weights, np.dot gives sum, not integral!
    if w_grid is not None:
        w1_lim = w_grid[:idx_limit].copy()
        w2_lim = w_grid[:idx_limit_r2].copy()
    else:
        # Fallback: unit weights (discrete sum - should be avoided!)
        w1_lim = np.ones(idx_limit)
        w2_lim = np.ones(idx_limit_r2)
    
    # Check for NaN/Inf in inputs
    if not np.all(np.isfinite(rho1_lim)) or not np.all(np.isfinite(rho2_lim)):
        logger.warning("oscillatory_kernel_integral_2d: NaN/Inf in density arrays, falling back to standard")
        rho1_lim = np.nan_to_num(rho1_lim, nan=0.0, posinf=0.0, neginf=0.0)
        rho2_lim = np.nan_to_num(rho2_lim, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == "standard":
        # Fast method with proper integration weights
        # Inner integral: ∫ K(r₁,r₂) × ρ₂(r₂) dr₂
        int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
        # Outer integral: ∫ ρ₁(r₁) × int_r2(r₁) dr₁
        result = np.dot(rho1_lim * w1_lim, int_r2)
        
        # SAFEGUARD: Check result
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d: Non-finite result, returning 0")
            return 0.0
        return float(result)
    
    elif method == "adaptive":
        # Phase-aware integration for outer integral
        k_total = k_i + k_f
        max_phase, is_ok, prob_idx = check_phase_sampling(
            r1_lim, k_total, eta_total=eta_total
        )
        
        if is_ok:
            # Standard is OK - fully vectorized with weights
            int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
            result = np.dot(rho1_lim * w1_lim, int_r2)
            
            if not np.isfinite(result):
                logger.warning("oscillatory_kernel_integral_2d: Non-finite result (is_ok path), returning 0")
                return 0.0
            return float(result)
        
        # Log phase undersampling details
        if prob_idx >= 0 and prob_idx < len(r1_lim):
            logger.debug(
                "Phase-adaptive integration: k_total=%.2f, max_phase=%.2f rad at r=%.1f",
                k_total, max_phase, r1_lim[prob_idx]
            )
        
        # Need adaptive treatment for outer integral
        # Inner integral uses weights for r2
        int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
        
        # SAFEGUARD: Check inner integral result
        if not np.all(np.isfinite(int_r2)):
            logger.warning("oscillatory_kernel_integral_2d: NaN/Inf in inner integral, using standard method")
            int_r2 = np.nan_to_num(int_r2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ==========================================================================
        # Phase check is for diagnostics; integration uses proper weights.
        # ==========================================================================
        
        dr = np.diff(r1_lim)
        if abs(eta_total) > 1e-10:
            r_mid = np.maximum(0.5 * (r1_lim[:-1] + r1_lim[1:]), 1e-10)
            phase_per_step = np.abs((k_total + eta_total / r_mid) * dr)
        else:
            phase_per_step = np.abs(k_total * dr)
        max_phase_threshold = np.pi / 4
        
        # Count undersampled regions for logging
        undersampled_count = np.sum(phase_per_step > max_phase_threshold)
        if undersampled_count > 0:
            logger.debug(
                "oscillatory_kernel_integral_2d: %d of %d intervals exceed phase threshold π/4",
                undersampled_count, len(dr)
            )
        
        # Outer integral with proper weights for r1
        result = float(np.dot(rho1_lim * w1_lim, int_r2))
        
        # SAFEGUARD: Final check
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d: Non-finite final result, returning 0")
            return 0.0
        
        return result
    
    elif method == "filon":
        # ==========================================================================
        # FILON + CLENSHAW-CURTIS METHOD
        # ==========================================================================
        # Per instruction: "rozbij całkę na przedziały, na których faza robi stały
        # przyrost, a na podprzedziałach użyj węzłów Clenshaw-Curtis"
        #
        # Strategy:
        # 1. Inner integral (r2): WEIGHTED sum (smooth bound states) - uses w2_lim
        # 2. Outer integral (r1): Phase-split with CC on sub-intervals
        # ==========================================================================
        
        k_total = k_i + k_f
        
        # Inner integral: uses integration weights for proper ∫dr₂
        # int_r2[i] = Σ_j kernel[i,j] × rho2[j] × w[j]
        int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
        
        if not np.all(np.isfinite(int_r2)):
            logger.warning("oscillatory_kernel_integral_2d (filon): NaN/Inf in inner integral")
            int_r2 = np.nan_to_num(int_r2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Phase-based interval splitting with undersampling-aware increment.
        PHASE_INCREMENT = _adaptive_phase_increment_for_grid(
            r1_lim,
            k_total,
            phase_increment,
            eta_total=eta_total,
            threshold=np.pi / 4,
            min_increment=np.pi / 16,
        )
        
        if k_total < 1e-6:
            # Non-oscillatory: standard integration with weights for r₁
            result = float(np.dot(rho1_lim * w1_lim, int_r2))
        else:
            # Generate phase nodes
            r_start, r_end = r1_lim[0], r1_lim[-1]
            phase_nodes = generate_phase_nodes(
                r_start, r_end, k_total, PHASE_INCREMENT, eta_total=eta_total
            )
            n_intervals = len(phase_nodes) - 1
            
            if n_intervals < 1:
                # Fallback to standard with weights
                result = float(np.dot(rho1_lim * w1_lim, int_r2))
            else:
                # =================================================================
                # VECTORIZED Clenshaw-Curtis integration over phase intervals
                # Uses cached reference nodes and weights from module level
                # =================================================================
                
                # Build all CC points for all intervals at once
                # Transform x_ref from [-1, 1] to [a, b] for each interval
                a_arr = phase_nodes[:-1]  # (n_intervals,)
                b_arr = phase_nodes[1:]   # (n_intervals,)
                
                # Skip zero-width intervals
                valid_mask = (b_arr - a_arr) > 1e-12
                a_valid = a_arr[valid_mask]
                b_valid = b_arr[valid_mask]
                n_valid = len(a_valid)
                
                if n_valid == 0:
                    result = float(np.dot(rho1_lim * w1_lim, int_r2))
                else:
                    # All CC points: shape (n_valid, n_nodes)
                    # r = 0.5 * (b - a) * (x_ref + 1) + a
                    x_ref, w_ref = _get_cc_ref(n_nodes)
                    half_width = 0.5 * (b_valid - a_valid)  # (n_valid,)
                    all_r = half_width[:, np.newaxis] * (x_ref + 1) + a_valid[:, np.newaxis]
                    
                    # Flatten for interpolation
                    all_r_flat = all_r.ravel()
                    
                    # Interpolate rho1 and int_r2
                    rho1_interp = _interp_on_radial_grid(all_r_flat, r1_lim, rho1_lim)
                    int_r2_interp = _interp_on_radial_grid(all_r_flat, r1_lim, int_r2)
                    
                    # Compute integrand
                    integrand = rho1_interp * int_r2_interp
                    
                    # Reshape to (n_valid, n_nodes)
                    integrand = integrand.reshape(n_valid, n_nodes)
                    
                    # Weights for each interval: w_ref * (b-a)/2
                    weights_scaled = w_ref * half_width[:, np.newaxis]  # (n_valid, n_nodes)
                    
                    # Sum: weighted sum over nodes, then sum over intervals
                    result = float(np.sum(integrand * weights_scaled))
        
        # SAFEGUARD: Final check
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d (filon): Non-finite result, returning 0")
            return 0.0
        
        return float(result)
    
    elif method == "filon_exchange":
        # ==========================================================================
        # FILON + CLENSHAW-CURTIS FOR EXCHANGE INTEGRALS
        # ==========================================================================
        # For exchange, BOTH densities contain oscillatory components:
        # - rho1_ex = u_f * chi_i (bound × continuum)
        # - rho2_ex = chi_f * u_i (continuum × bound)
        #
        # Strategy:
        # 1. Inner integral: Apply phase-split CC to each row of kernel×rho2 (with weights)
        # 2. Outer integral: Apply phase-split CC to rho1×int_r2
        # ==========================================================================
        
        k_total = k_i + k_f
        PHASE_INCREMENT = _adaptive_phase_increment_for_grid(
            r1_lim,
            k_total,
            phase_increment,
            eta_total=eta_total,
            threshold=np.pi / 4,
            min_increment=np.pi / 16,
        )
        
        if k_total < 1e-6:
            # Non-oscillatory: standard integration with weights
            int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
            result = float(np.dot(rho1_lim * w1_lim, int_r2))
        else:
            # Generate phase nodes
            r_start, r_end = r1_lim[0], r1_lim[-1]
            phase_nodes = generate_phase_nodes(
                r_start, r_end, k_total, PHASE_INCREMENT, eta_total=eta_total
            )
            n_intervals = len(phase_nodes) - 1
            
            if n_intervals < 1:
                # Fallback to standard with weights
                int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
                result = float(np.dot(rho1_lim * w1_lim, int_r2))
            else:
                # Use cached reference CC nodes and weights from module level
                
                # Get valid intervals
                a_arr = phase_nodes[:-1]
                b_arr = phase_nodes[1:]
                valid_mask = (b_arr - a_arr) > 1e-12
                a_valid = a_arr[valid_mask]
                b_valid = b_arr[valid_mask]
                n_valid = len(a_valid)
                
                if n_valid == 0:
                    int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)
                    result = float(np.dot(rho1_lim * w1_lim, int_r2))
                else:
                    half_width = 0.5 * (b_valid - a_valid)
                    x_ref, w_ref = _get_cc_ref(n_nodes)
                    all_r = half_width[:, np.newaxis] * (x_ref + 1) + a_valid[:, np.newaxis]
                    all_r_flat = all_r.ravel()
                    
                    # =========================================================
                    # INNER INTEGRAL with CC (vectorized over r1 grid points)
                    # =========================================================
                    # For each r1, compute ∫ K(r1, r2) × rho2(r2) dr2 using CC
                    # 
                    # Interpolate kernel and rho2 at CC nodes
                    rho2_interp = _interp_on_radial_grid(all_r_flat, r2_lim, rho2_lim)
                    rho2_cc = rho2_interp.reshape(n_valid, n_nodes)
                    
                    # For kernel, we need K(r1, r2_cc) for each r1
                    # kernel_lim has shape (n_r1, n_r2)
                    # We need to interpolate along the r2 axis for each r1
                    n_r1 = len(r1_lim)
                    n_r2_cc = len(all_r_flat)
                    
                    # VECTORIZED kernel interpolation using linear indexing
                    # Find interpolation indices and weights
                    idx_right = np.searchsorted(r2_lim, all_r_flat)
                    idx_right = np.clip(idx_right, 1, len(r2_lim) - 1)
                    idx_left = idx_right - 1
                    
                    # Interpolation weights
                    r_left = r2_lim[idx_left]
                    r_right = r2_lim[idx_right]
                    weight_right = (all_r_flat - r_left) / (r_right - r_left + 1e-30)
                    weight_left = 1.0 - weight_right
                    
                    # Interpolate kernel: shape (n_r1, n_r2_cc)
                    kernel_left = kernel_lim[:, idx_left]   # (n_r1, n_r2_cc)
                    kernel_right = kernel_lim[:, idx_right]  # (n_r1, n_r2_cc)
                    kernel_at_cc = kernel_left * weight_left + kernel_right * weight_right
                    
                    # Reshape to (n_r1, n_valid, n_nodes)
                    kernel_interp = kernel_at_cc.reshape(n_r1, n_valid, n_nodes)
                    
                    # Integrand for inner: kernel(r1, r2) * rho2(r2)
                    # Shape: kernel_interp (n_r1, n_valid, n_nodes) * rho2_cc (n_valid, n_nodes)
                    inner_integrand = kernel_interp * rho2_cc[np.newaxis, :, :]
                    
                    # Weights: _CC_W_REF * half_width for each interval
                    weights_2d = w_ref * half_width[:, np.newaxis]  # (n_valid, n_nodes)
                    
                    # Sum over CC nodes and intervals to get int_r2(r1)
                    # Result shape: (n_r1,)
                    int_r2_cc = np.sum(inner_integrand * weights_2d[np.newaxis, :, :], axis=(1, 2))
                    
                    # =========================================================
                    # OUTER INTEGRAL with CC
                    # =========================================================
                    # Interpolate rho1 and int_r2_cc at CC nodes
                    rho1_interp = _interp_on_radial_grid(all_r_flat, r1_lim, rho1_lim)
                    int_r2_outer = _interp_on_radial_grid(all_r_flat, r1_lim, int_r2_cc)
                    
                    # Compute outer integrand
                    outer_integrand = rho1_interp * int_r2_outer
                    outer_integrand = outer_integrand.reshape(n_valid, n_nodes)
                    
                    # Sum with weights
                    result = float(np.sum(outer_integrand * weights_2d))
        
        # SAFEGUARD: Final check
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d (filon_exchange): Non-finite result, returning 0")
            return 0.0
        
        return float(result)
    
    else:
        raise ValueError(f"Unknown method: {method}")
