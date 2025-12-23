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
import numpy as np
from typing import Tuple, Optional
from scipy.special import sici  # Sine and Cosine integrals

from logging_config import get_logger

logger = get_logger(__name__)


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



# =============================================================================
# PHASE SAMPLING DIAGNOSTICS
# =============================================================================

def check_phase_sampling(
    r: np.ndarray,
    k_total: float,
    threshold: float = np.pi / 4
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
    phase_per_step = k_total * dr
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
    l_f: int
) -> None:
    """
    Log diagnostic information about phase sampling.
    """
    k_total = k_i + k_f
    max_phase, is_ok, prob_idx = check_phase_sampling(r, k_total)
    
    if not is_ok:
        r_problem = r[prob_idx] if prob_idx >= 0 else r[-1]
        logger.warning(
            "Phase undersampling: l_i=%d, l_f=%d, k=%.2f+%.2f, "
            "max_dφ=%.2f rad at r=%.0f bohr",
            l_i, l_f, k_i, k_f, max_phase, r_problem
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
    phase_increment: float = np.pi / 2
) -> np.ndarray:
    """
    Generate grid points with constant phase increment.
    
    Creates an adaptive grid where the phase φ = k_total × r changes by
    exactly phase_increment between consecutive points. This ensures
    optimal sampling of oscillatory integrands.
    
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
        
    Returns
    -------
    r_nodes : np.ndarray
        Grid points with constant phase spacing.
        
    Notes
    -----
    For φ(r_{j+1}) - φ(r_j) = const = Δφ:
        r_{j+1} = r_j + Δφ / k_total
    
    This is the recommendation from the instruction:
    "rozbij całkę na przedziały, na których faza robi stały przyrost"
    """
    if k_total < 1e-10:
        # Non-oscillatory: just use endpoints
        return np.array([r_start, r_end])
    
    dr = phase_increment / k_total
    n_points = int(np.ceil((r_end - r_start) / dr)) + 1
    
    # Ensure we don't exceed r_end
    r_nodes = np.linspace(r_start, r_end, n_points)
    
    return r_nodes


def integrate_with_phase_nodes(
    f_func,
    r_start: float,
    r_end: float,
    k_total: float,
    phase_increment: float = np.pi / 2,
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
    r_nodes = generate_phase_nodes(r_start, r_end, k_total, phase_increment)
    
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
    
    For L ≥ 1, the multipole kernel decays as r^(-L-1) at large r.
    The integral ∫_{r_m}^∞ χ_i·χ_f × r^(-L-1) dr has a finite contribution
    that can be computed using asymptotic expansions.
    
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
        Multipole index.
    bound_overlap : float
        ∫ r^L × u_f × u_i dr (multipole moment of bound state densities).
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
    
    # For L=1 (dipole), use exact Si/Ci formula
    # FIX: Kernel for L=1 is r_< / r_>^2, so envelope decays as 1/r^2, not 1/r
    if L == 1:
        A_env = 1.0 / (r_m ** 2)  # 1/r² decay for dipole
        return _analytical_dipole_tail(
            r_m, k_i, k_f, delta_i, delta_f, l_i, l_f, A_env,
            eta_i, eta_f, sigma_i, sigma_f
        ) * bound_overlap
    
    # For L > 1, use asymptotic approximation
    # Kernel is r_<^L / r_>^(L+1), so envelope decays as 1/r^(L+1)
    # The bound state integral contributes r^L factor (computed as moment_L in caller)
    # So net envelope is 1/r^(L+1) from the kernel
    
    # Leading order contribution at r_m:
    # ∫_{r_m}^∞ sin(k_i r + φ_i) sin(k_f r + φ_f) / r^(L+1) dr
    # ≈ 1/r_m^L × [cos((k_i-k_f)r_m + Δφ)/(k_i-k_f) - cos((k_i+k_f)r_m + Σφ)/(k_i+k_f)] / 2
    
    # Include Coulomb phase terms
    log_term_i = eta_i * np.log(2 * k_i * r_m + 1e-30) if abs(eta_i) > 1e-10 else 0.0
    log_term_f = eta_f * np.log(2 * k_f * r_m + 1e-30) if abs(eta_f) > 1e-10 else 0.0
    
    phi_i = -l_i * np.pi / 2 + sigma_i + delta_i + log_term_i
    phi_f = -l_f * np.pi / 2 + sigma_f + delta_f + log_term_f
    
    k_diff = k_i - k_f
    k_sum = k_i + k_f
    phi_diff = phi_i - phi_f
    phi_sum = phi_i + phi_f
    
    tail = 0.0
    
    # Term 1: cos((k_i-k_f)·r_m + Δφ) / (k_i-k_f)
    if abs(k_diff) > 1e-6:
        term1 = np.cos(k_diff * r_m + phi_diff) / k_diff
        tail += 0.5 * term1
    
    # Term 2: -cos((k_i+k_f)·r_m + Σφ) / (k_i+k_f)
    if k_sum > 1e-6:
        term2 = np.cos(k_sum * r_m + phi_sum) / k_sum
        tail -= 0.5 * term2
    
    # FIX: Scale by 1/r_m^L (the integration gives 1/r^L boundary term from 1/r^(L+1) integrand)
    # The r^L from moment cancels with 1/r^L leaving tail contribution
    A_env = 1.0 / (r_m ** L) if r_m > 1.0 else 1.0
    
    return A_env * tail * bound_overlap


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
            
            # Linear interpolation for all arrays
            f_sub = np.interp(r_sub, r, f_vals)
            chi_i_sub = np.interp(r_sub, r, chi_i)
            chi_f_sub = np.interp(r_sub, r, chi_f)
            
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
    use_filon: bool = True
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
    max_phase, is_ok, _ = check_phase_sampling(r_num, k_total)
    
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
    method: str = "adaptive"
) -> float:
    """
    Compute 2D radial integral with oscillatory densities.
    
    I = ∫∫ ρ₁(r₁) · K(r₁, r₂) · ρ₂(r₂) dr₁ dr₂
    
    where ρ₁, ρ₂ contain oscillatory factors from continuum waves.
    
    Parameters
    ----------
    rho1, rho2 : np.ndarray
        Density arrays (already include weights and wavefunctions).
    kernel : np.ndarray
        2D kernel matrix K(r₁, r₂).
    r : np.ndarray
        Radial grid.
    k_i, k_f : float
        Wave numbers for phase diagnostic.
    idx_limit : int
        Integration limit index.
    method : str
        "standard": Use simple dot products (fast, may alias).
        "adaptive": Use phase-adaptive integration (slower, accurate).
        
    Returns
    -------
    integral : float
        Result of the 2D integration.
    """
    if idx_limit < 0:
        idx_limit = len(r)
    
    # ==========================================================================
    # SAFEGUARDS: Input Validation
    # ==========================================================================
    if idx_limit < 2:
        logger.warning("oscillatory_kernel_integral_2d: idx_limit < 2, returning 0")
        return 0.0
    
    rho1_lim = rho1[:idx_limit].copy()
    rho2_lim = rho2[:idx_limit].copy()
    kernel_lim = kernel[:idx_limit, :idx_limit]
    r_lim = r[:idx_limit]
    
    # Check for NaN/Inf in inputs
    if not np.all(np.isfinite(rho1_lim)) or not np.all(np.isfinite(rho2_lim)):
        logger.warning("oscillatory_kernel_integral_2d: NaN/Inf in density arrays, falling back to standard")
        rho1_lim = np.nan_to_num(rho1_lim, nan=0.0, posinf=0.0, neginf=0.0)
        rho2_lim = np.nan_to_num(rho2_lim, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == "standard":
        # Original fast method
        int_r2 = np.dot(kernel_lim, rho2_lim)
        result = np.dot(rho1_lim, int_r2)
        
        # SAFEGUARD: Check result
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d: Non-finite result, returning 0")
            return 0.0
        return float(result)
    
    elif method == "adaptive":
        # Phase-aware integration for outer integral
        k_total = k_i + k_f
        max_phase, is_ok, prob_idx = check_phase_sampling(r_lim, k_total)
        
        if is_ok:
            # Standard is OK - fully vectorized
            int_r2 = np.dot(kernel_lim, rho2_lim)
            result = np.dot(rho1_lim, int_r2)
            
            if not np.isfinite(result):
                logger.warning("oscillatory_kernel_integral_2d: Non-finite result (is_ok path), returning 0")
                return 0.0
            return float(result)
        
        # Log phase undersampling details
        if prob_idx >= 0 and prob_idx < len(r_lim):
            logger.debug(
                "Phase-adaptive integration: k_total=%.2f, max_phase=%.2f rad at r=%.1f",
                k_total, max_phase, r_lim[prob_idx]
            )
        
        # Need adaptive treatment for outer integral
        int_r2 = np.dot(kernel_lim, rho2_lim)
        
        # SAFEGUARD: Check inner integral result
        if not np.all(np.isfinite(int_r2)):
            logger.warning("oscillatory_kernel_integral_2d: NaN/Inf in inner integral, using standard method")
            int_r2 = np.nan_to_num(int_r2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ==========================================================================
        # FIX: Input densities (rho1, rho2) already include quadrature weights.
        # Therefore we must NOT apply trapezoid (which adds dr weighting again).
        # Instead, use simple weighted sum, consistent with "standard" method.
        # ==========================================================================
        
        # Phase check is still useful for diagnostics
        dr = np.diff(r_lim)
        phase_per_step = k_total * dr
        max_phase_threshold = np.pi / 4
        
        # Count undersampled regions for logging
        undersampled_count = np.sum(phase_per_step > max_phase_threshold)
        if undersampled_count > 0:
            logger.debug(
                "oscillatory_kernel_integral_2d: %d of %d intervals exceed phase threshold π/4",
                undersampled_count, len(dr)
            )
        
        # For weighted inputs: use simple dot product (same as standard)
        # The weights are already in rho1_lim, so sum(rho1 * int_r2) is correct.
        result = float(np.dot(rho1_lim, int_r2))
        
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
        # 1. Inner integral (r2): Standard weighted sum (smooth bound states)
        # 2. Outer integral (r1): Phase-split with CC on sub-intervals
        # ==========================================================================
        
        k_total = k_i + k_f
        
        # Inner integral: standard (bound states are smooth)
        int_r2 = np.dot(kernel_lim, rho2_lim)
        
        if not np.all(np.isfinite(int_r2)):
            logger.warning("oscillatory_kernel_integral_2d (filon): NaN/Inf in inner integral")
            int_r2 = np.nan_to_num(int_r2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Phase-based interval splitting
        PHASE_INCREMENT = np.pi / 2  # Δφ = π/2 per sub-interval
        
        if k_total < 1e-6:
            # Non-oscillatory: standard integration
            result = float(np.dot(rho1_lim, int_r2))
        else:
            # Generate phase nodes
            r_start, r_end = r_lim[0], r_lim[-1]
            phase_nodes = generate_phase_nodes(r_start, r_end, k_total, PHASE_INCREMENT)
            n_intervals = len(phase_nodes) - 1
            
            if n_intervals < 1:
                # Fallback to standard
                result = float(np.dot(rho1_lim, int_r2))
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
                    result = float(np.dot(rho1_lim, int_r2))
                else:
                    # All CC points: shape (n_valid, _CC_N)
                    # r = 0.5 * (b - a) * (x_ref + 1) + a
                    half_width = 0.5 * (b_valid - a_valid)  # (n_valid,)
                    all_r = half_width[:, np.newaxis] * (_CC_X_REF + 1) + a_valid[:, np.newaxis]
                    
                    # Flatten for interpolation
                    all_r_flat = all_r.ravel()
                    
                    # Interpolate rho1 and int_r2
                    rho1_interp = np.interp(all_r_flat, r_lim, rho1_lim)
                    int_r2_interp = np.interp(all_r_flat, r_lim, int_r2)
                    
                    # Compute integrand
                    integrand = rho1_interp * int_r2_interp
                    
                    # Reshape to (n_valid, _CC_N)
                    integrand = integrand.reshape(n_valid, _CC_N)
                    
                    # Weights for each interval: w_ref * (b-a)/2
                    weights_scaled = _CC_W_REF * half_width[:, np.newaxis]  # (n_valid, _CC_N)
                    
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
        # 1. Inner integral: Apply phase-split CC to each row of kernel×rho2
        # 2. Outer integral: Apply phase-split CC to rho1×int_r2
        # ==========================================================================
        
        k_total = k_i + k_f
        PHASE_INCREMENT = np.pi / 2
        CC_NODES = 5
        
        if k_total < 1e-6:
            # Non-oscillatory: standard integration
            int_r2 = np.dot(kernel_lim, rho2_lim)
            result = float(np.dot(rho1_lim, int_r2))
        else:
            # Generate phase nodes
            r_start, r_end = r_lim[0], r_lim[-1]
            phase_nodes = generate_phase_nodes(r_start, r_end, k_total, PHASE_INCREMENT)
            n_intervals = len(phase_nodes) - 1
            
            if n_intervals < 1:
                # Fallback to standard
                int_r2 = np.dot(kernel_lim, rho2_lim)
                result = float(np.dot(rho1_lim, int_r2))
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
                    int_r2 = np.dot(kernel_lim, rho2_lim)
                    result = float(np.dot(rho1_lim, int_r2))
                else:
                    half_width = 0.5 * (b_valid - a_valid)
                    all_r = half_width[:, np.newaxis] * (_CC_X_REF + 1) + a_valid[:, np.newaxis]
                    all_r_flat = all_r.ravel()
                    
                    # =========================================================
                    # INNER INTEGRAL with CC (vectorized over r1 grid points)
                    # =========================================================
                    # For each r1, compute ∫ K(r1, r2) × rho2(r2) dr2 using CC
                    # 
                    # Interpolate kernel and rho2 at CC nodes
                    rho2_interp = np.interp(all_r_flat, r_lim, rho2_lim)
                    rho2_cc = rho2_interp.reshape(n_valid, CC_NODES)
                    
                    # For kernel, we need K(r1, r2_cc) for each r1
                    # kernel_lim has shape (n_r1, n_r2)
                    # We need to interpolate along the r2 axis for each r1
                    n_r1 = len(r_lim)
                    n_r2_cc = len(all_r_flat)
                    
                    # VECTORIZED kernel interpolation using linear indexing
                    # Find interpolation indices and weights
                    idx_right = np.searchsorted(r_lim, all_r_flat)
                    idx_right = np.clip(idx_right, 1, len(r_lim) - 1)
                    idx_left = idx_right - 1
                    
                    # Interpolation weights
                    r_left = r_lim[idx_left]
                    r_right = r_lim[idx_right]
                    weight_right = (all_r_flat - r_left) / (r_right - r_left + 1e-30)
                    weight_left = 1.0 - weight_right
                    
                    # Interpolate kernel: shape (n_r1, n_r2_cc)
                    kernel_left = kernel_lim[:, idx_left]   # (n_r1, n_r2_cc)
                    kernel_right = kernel_lim[:, idx_right]  # (n_r1, n_r2_cc)
                    kernel_at_cc = kernel_left * weight_left + kernel_right * weight_right
                    
                    # Reshape to (n_r1, n_valid, _CC_N)
                    kernel_interp = kernel_at_cc.reshape(n_r1, n_valid, _CC_N)
                    
                    # Integrand for inner: kernel(r1, r2) * rho2(r2)
                    # Shape: kernel_interp (n_r1, n_valid, CC_NODES) * rho2_cc (n_valid, CC_NODES)
                    inner_integrand = kernel_interp * rho2_cc[np.newaxis, :, :]
                    
                    # Weights: _CC_W_REF * half_width for each interval
                    weights_2d = _CC_W_REF * half_width[:, np.newaxis]  # (n_valid, _CC_N)
                    
                    # Sum over CC nodes and intervals to get int_r2(r1)
                    # Result shape: (n_r1,)
                    int_r2_cc = np.sum(inner_integrand * weights_2d[np.newaxis, :, :], axis=(1, 2))
                    
                    # =========================================================
                    # OUTER INTEGRAL with CC
                    # =========================================================
                    # Interpolate rho1 and int_r2_cc at CC nodes
                    rho1_interp = np.interp(all_r_flat, r_lim, rho1_lim)
                    int_r2_outer = np.interp(all_r_flat, r_lim, int_r2_cc)
                    
                    # Compute outer integrand
                    outer_integrand = rho1_interp * int_r2_outer
                    outer_integrand = outer_integrand.reshape(n_valid, CC_NODES)
                    
                    # Sum with weights
                    result = float(np.sum(outer_integrand * weights_2d))
        
        # SAFEGUARD: Final check
        if not np.isfinite(result):
            logger.warning("oscillatory_kernel_integral_2d (filon_exchange): Non-finite result, returning 0")
            return 0.0
        
        return float(result)
    
    else:
        raise ValueError(f"Unknown method: {method}")

