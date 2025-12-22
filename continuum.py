# continuum.py
"""
Distorted-Wave Continuum Solver
================================

Solves the radial Schrödinger equation for scattering electron partial waves
χ_l(k,r) in the DWBA formulation.

Equation (atomic units)
-----------------------
    [-1/2 d²/dr² + l(l+1)/(2r²) + U_j(r)] χ_l(k,r) = (k²/2) χ_l(k,r)

where:
- l = partial-wave angular momentum
- U_j(r) = distorting potential (vanishes at large r)
- k = asymptotic wave number (E = k²/2)

Boundary Conditions
-------------------
- Origin: χ(r) ~ r^(l+1) (regular solution)
- Asymptotic: χ_l ~ sin(kr - lπ/2 + δ_l) with unit amplitude

Output
------
ContinuumWave objects containing:
- l, k_au (angular momentum, wave number)
- χ_l(r) normalized to unit asymptotic amplitude
- δ_l (phase shift in radians)

Implementation
--------------
- Uses scipy.integrate.solve_ivp for ODE integration
- Cubic spline interpolation for smooth potential
- Coulomb matching for charged targets

Logging
-------
Uses logging_config. Set DWBA_LOG_LEVEL=DEBUG for verbose output.
"""



from __future__ import annotations
import numpy as np
# mpmath removed for performance optimization (replaced with WKB approx)
from dataclasses import dataclass
from typing import Tuple

from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn, loggamma


from grid import RadialGrid, k_from_E_eV, ev_to_au
from distorting_potential import DistortingPotential
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)


@dataclass(frozen=True)
class ContinuumWave:
    """
    Distorted-wave continuum partial wave χ_l(k,r).

    Attributes
    ----------
    l : int
        Orbital angular momentum ℓ of this partial wave.
    k_au : float
        Wave number k in atomic units (1/bohr), related to kinetic
        energy by E = k^2 / 2 (Hartree).
    chi_of_r : np.ndarray
        The radial function χ_l(k,r) on grid.r, AFTER normalization so that
        asymptotically
            χ_l(k,r) ~ sin(k r - l π/2 + δ_l)
        (i.e. unit amplitude).
        Shape (N,), same as the global radial grid.
    phase_shift : float
        δ_l in radians, extracted from the log-derivative matching.
        Physically this encodes the short-range distortion due to U_j(r).
    r_match : float
        Matching radius where log-derivative was matched. Beyond this point,
        the wavefunction is essentially free (Bessel/Coulomb).
        Used for splitting radial integrals into numerical + analytical parts.
    idx_match : int
        Grid index corresponding to r_match. For integrals, use [0:idx_match+1]
        for reliable numerical part.
    """
    l: int
    k_au: float
    chi_of_r: np.ndarray
    phase_shift: float
    r_match: float = 0.0      # Match point radius
    idx_match: int = -1       # Match point grid index (-1 = not set, use full grid)


    @property
    def u_of_r(self) -> np.ndarray:
        """Alias for chi_of_r to ensure compatibility with BoundOrbital interface."""
        return self.chi_of_r


# =============================================================================
# RICCATI-BESSEL AND COULOMB FUNCTIONS FOR PHASE EXTRACTION
# =============================================================================

def _riccati_bessel_jn(l: int, rho: float) -> Tuple[float, float]:
    """
    Riccati-Bessel function ĵ_l(ρ) = ρ·j_l(ρ) and its derivative.
    
    Returns (ĵ_l, ĵ_l') where derivative is with respect to ρ.
    Uses recurrence: ĵ_l'(ρ) = ĵ_{l-1}(ρ) - (l/ρ)·ĵ_l(ρ) for l>0
    """
    if rho < 1e-10:
        # Small argument limit: j_l(ρ) ~ ρ^l / (2l+1)!!
        return 0.0, 0.0
    
    jl = float(spherical_jn(l, rho))
    j_hat = rho * jl  # ĵ_l = ρ·j_l
    
    if l == 0:
        # ĵ_0 = sin(ρ), ĵ_0' = cos(ρ)
        j_hat_prime = np.cos(rho)
    else:
        # Use j_{l-1} for derivative
        jl_minus1 = float(spherical_jn(l - 1, rho))
        j_hat_prime = rho * jl_minus1 - l * jl
    
    return j_hat, j_hat_prime


def _riccati_bessel_yn(l: int, rho: float) -> Tuple[float, float]:
    """
    Riccati-Neumann function n̂_l(ρ) = ρ·y_l(ρ) and its derivative.
    
    Returns (n̂_l, n̂_l') where derivative is with respect to ρ.
    Note: y_l is also called n_l (Neumann function) in some texts.
    """
    if rho < 1e-10:
        # y_l diverges at origin
        return -1e30, -1e30
    
    yl = float(spherical_yn(l, rho))
    n_hat = rho * yl  # n̂_l = ρ·y_l
    
    if l == 0:
        # n̂_0 = -cos(ρ), n̂_0' = sin(ρ)
        n_hat_prime = np.sin(rho)
    else:
        # Use y_{l-1} for derivative
        yl_minus1 = float(spherical_yn(l - 1, rho))
        n_hat_prime = rho * yl_minus1 - l * yl
    
    return n_hat, n_hat_prime


def _coulomb_FG_asymptotic(l: int, eta: float, rho: float) -> Tuple[float, float, float, float]:
    """
    Asymptotic Coulomb functions F_l, G_l and their derivatives.
    
    For ρ >> l, η:
        F_l(η,ρ) ≈ sin(θ)
        G_l(η,ρ) ≈ cos(θ)
    where θ = ρ + η·ln(2ρ) - lπ/2 + σ_l
    
    Returns (F, F', G, G')
    """
    if rho < 1e-10:
        return 0.0, 0.0, 1e30, 1e30
    
    # Coulomb phase shift σ_l = arg Γ(l+1+iη)
    sigma_l = np.imag(loggamma(l + 1 + 1j * eta))
    
    # Asymptotic argument
    theta = rho + eta * np.log(2.0 * rho) - (l * np.pi / 2.0) + sigma_l
    
    # Derivative of theta: dθ/dρ = 1 + η/ρ
    theta_prime = 1.0 + eta / rho
    
    F = np.sin(theta)
    G = np.cos(theta)
    F_prime = theta_prime * np.cos(theta)
    G_prime = -theta_prime * np.sin(theta)
    
    return F, F_prime, G, G_prime


def _find_match_point(r_grid: np.ndarray, U_arr: np.ndarray, k_au: float, l: int,
                       threshold: float = 1e-4, idx_start: int = 0) -> Tuple[int, float]:
    """
    Find optimal matching point r_m where potential is negligible.
    
    Criteria:
        |2U(r_m)| < threshold × k²
        |2U(r_m)| < threshold × l(l+1)/r_m²  (for l > 0)
    
    Parameters
    ----------
    idx_start : int
        Minimum index to consider (where propagation actually starts).
        Match point must be >= idx_start to have valid wavefunction.
    
    Returns the index and r value of the match point.
    Searches from large r inward to find the outermost valid point.
    """
    k2 = k_au ** 2
    N = len(r_grid)
    
    # Start from ~80% of grid to avoid edge effects
    search_start = int(0.8 * N)
    
    # Minimum search index: must be beyond idx_start (where χ has amplitude)
    # Add buffer to ensure we're well into oscillatory region
    min_search_idx = max(idx_start + 10, N // 4)
    
    for idx in range(search_start, min_search_idx, -1):
        r = r_grid[idx]
        U = U_arr[idx]
        
        # Criterion 1: potential small vs kinetic energy
        if abs(2.0 * U) > threshold * k2:
            continue
        
        # Criterion 2: potential small vs centrifugal (for l > 0)
        if l > 0:
            centrifugal = l * (l + 1) / (r * r)
            if abs(2.0 * U) > threshold * centrifugal:
                continue
        
        # Found a good point
        return idx, r
    
    # Fallback: use point slightly after idx_start (ensure valid wavefunction)
    fallback_idx = min(search_start, max(idx_start + 20, int(0.5 * N)))
    logger.debug(f"No ideal match point found for L={l}, k={k_au:.3f}; using idx={fallback_idx}")
    return fallback_idx, r_grid[fallback_idx]


def _extract_phase_logderiv_neutral(Y_m: float, k_au: float, r_m: float, l: int) -> float:
    """
    Extract phase shift using log-derivative matching for neutral target.
    
    Formula:
        tan(δ_l) = [Y_m · ĵ_l(kr_m) - ĵ_l'(kr_m)] / [n̂_l'(kr_m) - Y_m · n̂_l(kr_m)]
    
    Parameters
    ----------
    Y_m : float
        Log-derivative χ'/χ at match point.
    k_au : float
        Wave number in atomic units.
    r_m : float
        Match point radius.
    l : int
        Angular momentum.
        
    Returns
    -------
    delta_l : float
        Phase shift in radians.
    """
    rho_m = k_au * r_m
    
    # Get Riccati-Bessel functions and derivatives at match point
    j_hat, j_hat_prime = _riccati_bessel_jn(l, rho_m)
    n_hat, n_hat_prime = _riccati_bessel_yn(l, rho_m)
    
    # Convert Y_m (derivative w.r.t. r) to Y_rho (derivative w.r.t. ρ)
    # χ' = dχ/dr = k · dχ/dρ, so Y_m = k · Y_rho
    # But Riccati functions use ρ, so we need Y in ρ-space
    Y_rho = Y_m / k_au
    
    # tan(δ) formula per instruction:
    # tan(δ) = [Y_m · u1 - u1'] / [u2' - Y_m · u2]
    # where u1 = ĵ, u2 = n̂
    numerator = Y_rho * j_hat - j_hat_prime
    denominator = n_hat_prime - Y_rho * n_hat  # CORRECT per instruction: u2' - Y·u2
    
    if abs(denominator) < 1e-15:
        # Near resonance - δ ≈ ±π/2
        return np.pi / 2.0 if numerator > 0 else -np.pi / 2.0
    
    tan_delta = numerator / denominator
    delta_l = np.arctan(tan_delta)
    
    return delta_l


def _extract_phase_logderiv_coulomb(Y_m: float, k_au: float, r_m: float, l: int, 
                                     z_ion: float) -> float:
    """
    Extract phase shift using log-derivative matching for ionic target.
    
    Formula:
        tan(δ_l) = [Y_m · F_l - F_l'] / [G_l' - Y_m · G_l]
    
    where F_l, G_l are Coulomb functions at (η, kr_m).
    """
    eta = -z_ion / k_au
    rho_m = k_au * r_m
    
    # Get Coulomb functions at match point
    F, F_prime, G, G_prime = _coulomb_FG_asymptotic(l, eta, rho_m)
    
    # Convert Y_m to ρ-space
    Y_rho = Y_m / k_au
    
    # tan(δ) = [Y_ρ·F - F'] / [G' - Y_ρ·G]  (per instruction)
    numerator = Y_rho * F - F_prime
    denominator = G_prime - Y_rho * G  # CORRECT per instruction: u2' - Y·u2
    
    if abs(denominator) < 1e-15:
        return np.pi / 2.0 if numerator > 0 else -np.pi / 2.0
    
    tan_delta = numerator / denominator
    delta_l = np.arctan(tan_delta)
    
    return delta_l


def _schrodinger_rhs_factory(l: int, U_spline: CubicSpline, k_au: float):
    """
    Build RHS function for solve_ivp representing the system:
        y0 = χ(r)
        y1 = χ'(r)
        y0' = y1
        y1' = S(r) * y0
    where
        S(r) = l(l+1)/r^2 + 2 U(r) - k^2.

    Parameters
    ----------
    l : int
        Partial-wave angular momentum.
    U_spline : CubicSpline
        Interpolant U(r) in Hartree.
    k_au : float
        Wave number k in a.u.

    Returns
    -------
    rhs : callable
        rhs(r, y) suitable for solve_ivp.
    """
    ell = float(l)
    k2 = k_au ** 2

    def rhs(r, y):
        # y = [χ, χ']
        chi = y[0]
        dchi = y[1]

        # Evaluate potential at this r
        U_val = float(U_spline(r))

        # Build S(r)
        # Guard against r=0: we never integrate exactly from r=0,
        # start from r_min>0, so division is fine.
        S = ell * (ell + 1.0) / (r * r) + 2.0 * U_val - k2

        # d/dr [χ]   = χ'
        # d/dr [χ']  = S(r)*χ
        return np.array([dchi, S * chi], dtype=float)

    return rhs


def _initial_conditions_regular(r0: float, l: int) -> np.ndarray:
    """
    Set initial conditions at small r (r0 ~ r_min) for a regular solution.

    Near r -> 0, the regular radial solution behaves as:
        χ(r) ~ r^{l+1}
        χ'(r) ~ (l+1) r^{l}

    This is standard for partial waves in a central potential and matches
    the physical incoming channel requirement: no singularity at the origin.

    Parameters
    ----------
    r0 : float
        Smallest radius where we start integration (bohr). Must be >0.
    l : int
        Orbital angular momentum l.

    Returns
    -------
    y0 : np.ndarray, shape (2,)
        Initial state vector [χ(r0), χ'(r0)].
    """
    if r0 <= 0.0:
        raise ValueError("r0 must be > 0 for regular boundary condition.")
    ell = float(l)
    
    # Calculate physical value
    # Beware of underflow for large l and small r0 (e.g. 0.05^100)
    # chi_phys ~ r0^(l+1)
    # We only care about the shape (log-derivative), as the amplitude is normalized later.
    # So we can scale this to be within a safe numerical range (e.g. 1e-20).
    
    try:
        # Check magnitude in log space to avoid immediate underflow
        log_chi = (ell + 1.0) * np.log(r0)
        
        target_log = -20.0 # ~1e-9 .. 1e-10 range is safe for standard tolerances
        
        if log_chi < target_log:
            # We are in underflow/tiny territory.
            # We set chi0 = exp(target_log)
            # And dchi0 must preserve ratio: dchi/chi = (l+1)/r0
            chi0 = np.exp(target_log)
            dchi0 = chi0 * (ell + 1.0) / r0
        else:
            # Safe to compute directly
            chi0 = r0 ** (ell + 1.0)
            dchi0 = (ell + 1.0) * (r0 ** ell)

        # Scaling: For high L, r0^(L+1) without (2L+1)!! factor implies
        # the solution at large r will be roughly (2L+1)!! times larger (huge).
        # We rescale the initial condition to a safe small magnitude (e.g. 1e-20)
        # to prevent overflow of the unnormalized solution at r_max.
        current_mag = np.hypot(chi0, dchi0)
        if current_mag > 1e-100:  # Avoid division by zero
            scale_factor = 1e-20 / current_mag
            chi0 *= scale_factor
            dchi0 *= scale_factor
            
    except:
        # Fallback if logs fail (shouldn't happen for r0>0)
        chi0 = 1e-20
        dchi0 = chi0 * (ell + 1.0) / r0

    return np.array([chi0, dchi0], dtype=float)


def _initial_conditions_high_L(r0: float, l: int, k_au: float, z_ion: float) -> np.ndarray:
    """
    Compute precise initial conditions at r0 (inside/near barrier).
    
    OPTIMIZATION NOTE: 
    Previously used mpmath.coulombf which is extremely slow (arbitrary precision).
    For High L, we are typically deep inside the centrifugal barrier where the
    wavefunction is exponentially growing (tunneling).
    
    We use the WKB approximation for the logarithmic derivative:
        χ'/χ ≈ κ(r) = sqrt( V_eff(r) - E )
                 = sqrt( l(l+1)/r^2 + 2*U(r) - k^2 )
    
    This is accurate enough for initialization of the ODE solver, which then
    corrects the shape as it integrates outward.
    """
    rho = k_au * r0
    
    if abs(z_ion) < 1e-3:
        # Neutral target -> Free particle solution (Bessel)
        # scipy.special.spherical_jn(n, z)
        # j_l
        jl = spherical_jn(l, rho)
        # j_{l-1}
        # handle l=0 case safely (though high L implies l>0)
        if l > 0:
            jlm1 = spherical_jn(l-1, rho)
        else:
            jlm1 = np.cos(rho)/rho if rho>0 else 0.0 # limit? linear for small rho
            
        u_val = r0 * jl
        
        # u' = kr j_{l-1} - l j_l
        u_der = rho * jlm1 - l * jl
        
        # Scaling to avoid underflow
        # Scale to 1e-3 to be well above solver atol (1e-8)
        norm_val = np.hypot(u_val, u_der)
        if norm_val < 1e-3 and norm_val > 0.0:
            scale = 1e-3 / norm_val
            u_val *= scale
            u_der *= scale
            
        return np.array([u_val, u_der], dtype=float)
        
    else:
        # Ionic target -> Coulomb-like region
        # Use WKB approximation instead of slow Coulomb functions
        
        ell = float(l)
        # Effective potential terms in the radial equation:
        # chi'' = [ l(l+1)/r^2 + 2*V(r) - k^2 ] chi
        # inside barrier, RHS > 0.
        
        # Estimate V(r) ~ -z_ion/r (Coulomb)
        # (ignoring short range deviations of core potential as L term dominates)
        
        kappa_sq = ell*(ell+1.0)/(r0*r0) - 2.0*z_ion/r0 - k_au*k_au
        
        if kappa_sq > 0:
            # Inside barrier
            kappa = np.sqrt(kappa_sq)
            
            # WKB solution grows as ~ exp(integral kappa dr)
            # Log derivative is +kappa (integrating outward)
            
            val = 1.0e-20 # Arbitrary small value
            der = val * kappa
            
            return np.array([val, der], dtype=float)
        else:
            # We are outside the barrier or logic failed?
            # Theoretically shouldn't happen if r0 is set to 0.9 * screening_radius
            # Fallback to neutral bessel or regular condition
             return _initial_conditions_regular(r0, l)



def _fit_asymptotic_phase_neutral(
    r_tail: np.ndarray,
    chi_tail: np.ndarray,
    l: int,
    k_au: float
) -> Tuple[float, float]:
    """
    Fit χ(r) in the asymptotic region to:
        χ(r) ≈ A sin(k r - l π/2 + δ_l)

    We do this by linearizing:
        sin(k r - l π/2 + δ) =
           sin(k r - l π/2) cos δ + cos(k r - l π/2) sin δ

    So we fit χ(r) ≈ A_s * sin_part + A_c * cos_part
    with least squares, where:
        sin_part = sin(k r - l π/2)
        cos_part = cos(k r - l π/2)

    Then:
        A   = sqrt(A_s^2 + A_c^2)
        δ_l = atan2(A_c, A_s)
    because
        A_s = A cos δ_l
        A_c = A sin δ_l

    Parameters
    ----------
    r_tail : np.ndarray
        Radii in the "tail" region (large r where U_j ~ 0).
    chi_tail : np.ndarray
        Computed χ(r_tail).
    l : int
        Partial wave index.
    k_au : float
        Wave number k (a.u.).

    Returns
    -------
    A : float
        Amplitude of the asymptotic sinusoid.
    delta_l : float
        Phase shift δ_l in radians.
    """
    ell = float(l)
    phase_free = k_au * r_tail - ell * np.pi / 2.0

    sin_part = np.sin(phase_free)
    cos_part = np.cos(phase_free)

    # Linear least squares:
    # chi_tail ≈ A_s * sin_part + A_c * cos_part
    # Solve M [A_s, A_c]^T ≈ chi_tail
    M = np.vstack([sin_part, cos_part]).T  # shape (n_tail, 2)

    # Least squares solve
    coeffs, *_ = np.linalg.lstsq(M, chi_tail, rcond=None)
    A_s, A_c = coeffs

    # Use hypot for overflow safety
    A = np.hypot(A_s, A_c)
    
    # Only check for numerical explosion (not small amplitude - that's physical)
    if not np.isfinite(A) or A > 1e100:
        return 0.0, 0.0

    delta_l = np.arctan2(A_c, A_s)

    return float(A), float(delta_l)


def _fit_asymptotic_phase_coulomb(r_tail: np.ndarray, chi_tail: np.ndarray, l: int, k_au: float, z_ion: float) -> Tuple[float, float]:
    """
    Fit A and delta to the tail using asymptotic Coulomb functions.
    Replaces mpmath with fast asymptotic formulas valid at large r.
    
    F_l ~ sin(theta)
    G_l ~ cos(theta)
    theta = kr - l*pi/2 + eta*ln(2kr) + sigma_l
    """
    eta = -z_ion / k_au
    
    # Coulomb phase shift sigma_l = arg(Gamma(l + 1 + i*eta))
    coulomb_phase = np.imag(loggamma(l + 1 + 1j * eta))
    
    rho = k_au * r_tail
    
    # Asymptotic argument theta
    # Standard Coulomb phase: theta = rho + eta ln(2 rho) - l*pi/2 + sigma_l
    # (eta = -Z/k), so the sign here matters for ionic targets.
    theta = rho + eta * np.log(2.0 * rho) - (l * np.pi / 2.0) + coulomb_phase
    
    F_arr = np.sin(theta)
    G_arr = np.cos(theta)
    
    # Fit chi ~ C1 * F + C2 * G
    # C1 = A cos(delta)
    # C2 = A sin(delta)
    
    M = np.vstack([F_arr, G_arr]).T
    coeffs, *_ = np.linalg.lstsq(M, chi_tail, rcond=None)
    c1, c2 = coeffs
    
    A = np.hypot(c1, c2)
    
    # Only check for numerical explosion
    if not np.isfinite(A) or A > 1e100:
        return 0.0, 0.0
    
    delta_l = np.arctan2(c2, c1)
    
    return float(A), float(delta_l)


# =============================================================================
# NUMEROV PROPAGATOR WITH RENORMALIZATION
# =============================================================================
#
# The Numerov method solves χ''(r) = Q(r)·χ(r) with O(h⁶) accuracy.
# It is more stable than RK45 for oscillatory solutions and avoids the
# chi reconstruction issues of the log-derivative method.
#
# Key formula (constant step h):
#     a[i] = 1 - (h²/12)·Q[i]
#     b[i] = 2 + (5h²/6)·Q[i]
#     χ[i+1] = (b[i]·χ[i] - a[i-1]·χ[i-1]) / a[i+1]
#
# Periodic renormalization prevents under/overflow.
# =============================================================================

def _derivative_5point(chi: np.ndarray, r_grid: np.ndarray, idx: int) -> float:
    """
    Derivative using central difference for NON-UNIFORM grid.
    
    For non-uniform grids, the 5-point formula requires weighted coefficients.
    We use a simpler 3-point central difference with local step sizes, 
    which is still O(h²) accurate and works for exponential grids.
    
    Formula: χ'(r_i) ≈ (χ[i+1] - χ[i-1]) / (r[i+1] - r[i-1])
    
    Parameters
    ----------
    chi : np.ndarray
        Wavefunction array.
    r_grid : np.ndarray
        Radial grid array.
    idx : int
        Index at which to compute derivative.
        
    Returns
    -------
    dchi : float
        Derivative χ'(r_idx).
    """
    N = len(chi)
    
    if idx <= 0:
        # Forward difference at left boundary
        h_local = r_grid[1] - r_grid[0]
        return (chi[1] - chi[0]) / h_local
    elif idx >= N - 1:
        # Backward difference at right boundary
        h_local = r_grid[-1] - r_grid[-2]
        return (chi[-1] - chi[-2]) / h_local
    else:
        # Central difference with local step
        return (chi[idx + 1] - chi[idx - 1]) / (r_grid[idx + 1] - r_grid[idx - 1])



def _numerov_propagate(
    r_grid: np.ndarray,
    Q_arr: np.ndarray,
    chi0: float,
    chi1: float,
    renorm_interval: int = 100,
    renorm_scale: float = 1e50
) -> Tuple[np.ndarray, float]:
    """
    Propagate χ using Numerov method: χ'' = Q·χ
    
    This version handles NON-UNIFORM grids by using local step sizes.
    For exponential grids, h varies with position.
    
    Uses periodic renormalization to prevent over/underflow.
    
    Parameters
    ----------
    r_grid : np.ndarray
        Radial grid (can be non-uniform / exponential).
    Q_arr : np.ndarray
        Effective potential Q(r) = l(l+1)/r² + 2U(r) - k² on grid.
    chi0, chi1 : float
        Initial values χ(r₀), χ(r₁).
    renorm_interval : int
        Renormalize every N steps.
    renorm_scale : float
        Scale factor for renormalization threshold.
        
    Returns
    -------
    chi : np.ndarray
        Wavefunction on grid (before final normalization).
    log_scale : float
        Cumulative log of renormalization factors: χ_true = χ * exp(log_scale).
    """
    N = len(r_grid)
    chi = np.zeros(N, dtype=float)
    chi[0] = chi0
    chi[1] = chi1
    
    # Precompute local step sizes (non-uniform!)
    h_arr = np.diff(r_grid)  # h[i] = r[i+1] - r[i], length N-1
    
    log_scale = 0.0
    
    # Propagate step by step with local Numerov formula
    # For non-uniform grid, use local h at each step
    for i in range(1, N - 1):
        h_prev = h_arr[i - 1]  # r[i] - r[i-1]
        h_next = h_arr[i]      # r[i+1] - r[i]
        
        # For slightly varying h, use average for Numerov coefficients
        h_avg = 0.5 * (h_prev + h_next)
        h2 = h_avg * h_avg
        
        # Numerov coefficients with local h
        a_prev = 1.0 - (h2 / 12.0) * Q_arr[i - 1]
        b_curr = 2.0 + (5.0 * h2 / 6.0) * Q_arr[i]
        a_next = 1.0 - (h2 / 12.0) * Q_arr[i + 1]
        
        # Guard against division by zero
        if abs(a_next) < 1e-15:
            a_next = 1e-15 if a_next >= 0 else -1e-15
            
        chi[i + 1] = (b_curr * chi[i] - a_prev * chi[i - 1]) / a_next
        
        # Periodic renormalization
        if (i + 1) % renorm_interval == 0:
            max_val = max(abs(chi[i]), abs(chi[i + 1]))
            if max_val > renorm_scale:
                # Scale down
                scale = renorm_scale / max_val
                chi[:i + 2] *= scale
                log_scale -= np.log(scale)
            elif max_val < 1.0 / renorm_scale and max_val > 0:
                # Scale up
                scale = 1.0 / (max_val * 1e10)
                chi[:i + 2] *= scale
                log_scale -= np.log(scale)
    
    return chi, log_scale


def _build_asymptotic_wave(
    r_grid: np.ndarray,
    k_au: float,
    l: int,
    delta_l: float,
    z_ion: float,
    idx_start: int = 0
) -> np.ndarray:
    """
    Build the asymptotic wavefunction with proper normalization.
    
    For normalization to δ(k-k'), the asymptotic amplitude should be sqrt(2/π).
    
    Neutral: χ_as(r) = A·[ĵ_l(kr)·cos(δ) - n̂_l(kr)·sin(δ)]
    Ionic:   χ_as(r) = A·[F_l(η,kr)·cos(δ) - G_l(η,kr)·sin(δ)]
    
    where A = sqrt(2/π) for δ(k-k') normalization.
    
    Parameters
    ----------
    r_grid : np.ndarray
        Radial grid.
    k_au : float
        Wave number in atomic units.
    l : int
        Angular momentum.
    delta_l : float
        Phase shift in radians.
    z_ion : float
        Ionic charge (0 for neutral).
    idx_start : int
        Start index (set chi=0 before this for turning point).
        
    Returns
    -------
    chi_as : np.ndarray
        Asymptotic wavefunction with proper normalization.
    """
    N = len(r_grid)
    chi_as = np.zeros(N, dtype=float)
    
    # Normalization factor for δ(k-k')
    A = np.sqrt(2.0 / np.pi)
    
    cos_d = np.cos(delta_l)
    sin_d = np.sin(delta_l)
    
    if abs(z_ion) < 1e-3:
        # Neutral: use Riccati-Bessel functions
        for i in range(idx_start, N):
            r = r_grid[i]
            rho = k_au * r
            if rho < 1e-10:
                continue
            j_hat, _ = _riccati_bessel_jn(l, rho)
            n_hat, _ = _riccati_bessel_yn(l, rho)
            chi_as[i] = A * (j_hat * cos_d - n_hat * sin_d)
    else:
        # Ionic: use Coulomb functions
        eta = -z_ion / k_au
        for i in range(idx_start, N):
            r = r_grid[i]
            rho = k_au * r
            if rho < 1e-10:
                continue
            F, _, G, _ = _coulomb_FG_asymptotic(l, eta, rho)
            chi_as[i] = A * (F * cos_d - G * sin_d)
    
    return chi_as


# =============================================================================
# Johnson Log-Derivative Propagator (Fallback)
# =============================================================================
#
# The Johnson method propagates the log-derivative Y(r) = χ'(r)/χ(r) instead
# of χ directly. This is numerically stable even in classically forbidden 
# regions (tunneling) where χ itself would under/overflow.
#
# Reference: B.R. Johnson, J. Comp. Phys. 13, 445 (1973)
# 
# The Riccati equation for Y is:
#     dY/dr = -Y² - S(r)
# where S(r) = l(l+1)/r² + 2U(r) - k²
#
# =============================================================================

def _johnson_log_derivative_solve(
    r_grid: np.ndarray,
    U_arr: np.ndarray,
    l: int,
    k_au: float,
    renorm_interval: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve radial Schrödinger using Johnson's log-derivative propagator.
    
    This method is much more stable than direct ODE integration for high L
    because it propagates Y = χ'/χ which stays O(1) even when χ is tiny.
    
    Parameters
    ----------
    r_grid : np.ndarray
        Radial grid points (must be monotonically increasing).
    U_arr : np.ndarray  
        Potential values U(r) on grid.
    l : int
        Angular momentum quantum number.
    k_au : float
        Wave number in atomic units.
    renorm_interval : int
        Renormalize wavefunction every N steps to prevent overflow.
        
    Returns
    -------
    chi : np.ndarray
        Wavefunction χ(r) on grid.
    dchi : np.ndarray
        Derivative χ'(r) on grid.
    """
    N = len(r_grid)
    k2 = k_au ** 2
    ell = float(l)
    
    # Initialize arrays
    chi = np.zeros(N, dtype=float)
    dchi = np.zeros(N, dtype=float)
    
    # --- Initial Conditions ---
    # For high L in tunneling region, use WKB approximation
    # Y_WKB = sqrt(S(r)) where S > 0 (inside barrier)
    r0 = r_grid[0]
    S0 = ell * (ell + 1.0) / (r0 * r0) + 2.0 * U_arr[0] - k2
    
    if S0 > 0:
        # Inside barrier: WKB gives Y ≈ +sqrt(S)
        Y_init = np.sqrt(S0)
    else:
        # Outside barrier: oscillatory, Y ≈ (l+1)/r as small r
        Y_init = (ell + 1.0) / r0
    
    # Start with unit amplitude - will be renormalized anyway
    chi[0] = 1.0
    dchi[0] = Y_init * chi[0]
    
    Y_current = Y_init
    
    # --- RK4 Propagation for Y ---
    # dY/dr = -Y² - S(r)
    # More stable than simple Euler
    
    for i in range(N - 1):
        r = r_grid[i]
        r_next = r_grid[i + 1]
        h = r_next - r
        
        # S(r) function
        def S_func(rr, idx_hint):
            # Interpolate U if needed
            if idx_hint < N - 1:
                t = (rr - r_grid[idx_hint]) / (r_grid[idx_hint + 1] - r_grid[idx_hint])
                U_interp = U_arr[idx_hint] * (1 - t) + U_arr[idx_hint + 1] * t
            else:
                U_interp = U_arr[-1]
            return ell * (ell + 1.0) / (rr * rr) + 2.0 * U_interp - k2
        
        # RK4 for Y
        def dY(rr, Y_val, idx):
            return -Y_val**2 - S_func(rr, idx)
        
        k1 = dY(r, Y_current, i)
        k2_rk = dY(r + 0.5*h, Y_current + 0.5*h*k1, i)
        k3 = dY(r + 0.5*h, Y_current + 0.5*h*k2_rk, i)
        k4 = dY(r_next, Y_current + h*k3, i)
        
        Y_next = Y_current + (h/6.0) * (k1 + 2*k2_rk + 2*k3 + k4)
        
        # Limit Y to prevent numerical explosion
        Y_next = np.clip(Y_next, -100.0, 100.0)
        
        # Reconstruct chi from Y using logarithmic integration
        Y_avg = 0.5 * (Y_current + Y_next)
        exp_arg = h * Y_avg
        exp_arg = np.clip(exp_arg, -30, 30)  # Stricter limits
        
        chi[i + 1] = chi[i] * np.exp(exp_arg)
        dchi[i + 1] = Y_next * chi[i + 1]
        
        Y_current = Y_next
        
        # Renormalize more frequently to prevent issues
        if (i + 1) % renorm_interval == 0:
            max_val = np.max(np.abs(chi[:i + 2]))
            if max_val > 1e5 or (max_val < 1e-100 and max_val > 0):
                if max_val > 0:
                    scale = 1.0 / max_val
                    chi[:i + 2] *= scale
                    dchi[:i + 2] *= scale
    
    # Final normalization to reasonable scale
    max_val = np.max(np.abs(chi))
    if max_val > 0 and max_val != 1.0:
        chi /= max_val
        dchi /= max_val
    
    return chi, dchi



def _verify_wronskian(
    chi: np.ndarray,
    dchi: np.ndarray,
    r_match: float,
    r_grid: np.ndarray,
    l: int,
    k_au: float,
    z_ion: float
) -> float:
    """
    Verify solution quality at match point by checking log-derivative smoothness.
    
    Instead of comparing with asymptotic formula (which has singularities),
    we check that the log-derivative Y = χ'/χ is smooth by comparing
    values at two nearby points.
    
    Returns
    -------
    smoothness_error : float
        Relative variation in Y between two nearby points. <0.1 is good.
    """
    # Find match index
    idx = np.searchsorted(r_grid, r_match)
    idx = min(max(idx, 2), len(r_grid) - 3)
    
    # Compare Y at two nearby points
    chi_1 = chi[idx]
    dchi_1 = dchi[idx]
    chi_2 = chi[idx - 2]
    dchi_2 = dchi[idx - 2]
    
    if abs(chi_1) < 1e-100 or abs(chi_2) < 1e-100:
        return 0.0  # Can't check if amplitude is tiny
    
    Y_1 = dchi_1 / chi_1
    Y_2 = dchi_2 / chi_2
    
    # The log-derivative should be smooth (slowly varying)
    # Compute relative change normalized by k (natural scale)
    delta_r = abs(r_grid[idx] - r_grid[idx - 2])
    dY_dr = (Y_1 - Y_2) / (delta_r + 1e-10)
    
    # Expected rate of change is O(1/r) or O(k), so normalize by k
    smoothness_error = abs(dY_dr) / (k_au + 1.0)
    
    # Also check if Y is within reasonable bounds
    # For oscillatory solution, |Y| should be O(k)
    if abs(Y_1) > 10.0 * k_au and k_au > 0.1:
        smoothness_error = max(smoothness_error, 1.0)  # Flag as bad
    
    return min(smoothness_error, 10.0)  # Cap at 10 for readability


def _log_convergence_diagnostics(
    l: int,
    k_au: float,
    A_amp: float,
    delta_l: float,
    phase_stable: bool,
    wronskian_error: float,
    method_used: str
):
    """
    Log convergence diagnostics for a partial wave.
    
    Parameters
    ----------
    l : int
        Angular momentum.
    k_au : float
        Wave number.
    A_amp : float
        Fitted amplitude.
    delta_l : float
        Phase shift.
    phase_stable : bool
        Whether phase is stable across tail regions.
    wronskian_error : float
        Wronskian deviation from expected.
    method_used : str
        "RK45" or "Johnson".
    """
    status = "OK"
    issues = []
    
    if not phase_stable:
        issues.append("phase_unstable")
        status = "WARN"
    
    if wronskian_error > 0.1:
        issues.append(f"wronskian_err={wronskian_error:.2f}")
        status = "WARN"
    
    if A_amp < 1e-10:
        issues.append("tiny_amplitude")
        status = "WARN"
    
    if status == "WARN":
        logger.warning(f"L={l:3d} [{method_used}] A={A_amp:.2e} δ={delta_l:+.4f} "
                       f"issues: {', '.join(issues)}")
    else:
        logger.debug(f"L={l:3d} [{method_used}] A={A_amp:.2e} δ={delta_l:+.4f} OK")


def solve_continuum_wave(

    grid: RadialGrid,
    U_channel: DistortingPotential,
    l: int,
    E_eV: float,
    z_ion: float = 0.0,
    tail_fraction: float = 0.1,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> ContinuumWave:
    """
    Solve for the distorted-wave scattering solution χ_l(k,r) in channel j.

    This corresponds to χ_l^{(i)} for the entrance channel (with U_i),
    or χ_l^{(f)} for the exit channel (with U_f), as constructed in the
    article's DWBA formalism.

    Physics:
    --------
    We solve:
        χ''(r) = [ l(l+1)/r^2 + 2 U_j(r) - k^2 ] χ(r)
    for a FIXED kinetic energy E, where
        E_au = E_eV / 27.211...
        k = sqrt(2 E_au).

    Boundary conditions:
    - Regular at small r:
        χ(r_min)   = r_min^{l+1}
        χ'(r_min)  = (l+1) r_min^{l}
      This enforces physical regularity (no singularity at r=0),
      matching the physical incoming partial wave.

    - Asymptotically, U_j(r) ~ 0, so:
        χ_l(k,r) ~ A sin(k r - l π/2 + δ_l)
      We fit A and δ_l from the numerical tail, then renormalize χ
      so that A = 1. Thus χ_l has the standard phase normalization
      used in DWBA.

    Numerics:
    ---------
    1. We spline-interpolate U_j(r) so it's smooth for the ODE solver.
    2. We integrate outward with solve_ivp from r_min to r_max on a dense
       set of internal points.
    3. We sample χ(r) back on grid.r (to stay consistent everywhere).
    4. We fit the tail (last tail_fraction of the grid) to a sine/cosine.
    5. We divide χ by the fitted amplitude A, so the asymptotic
       amplitude = 1, and store δ_l.

    Parameters
    ----------
    grid : RadialGrid
        The global radial grid (bohr).
    U_channel : DistortingPotential
        Distorted potential U_j(r) for this channel (U_i or U_f).
        We'll use U_channel.U_of_r as U_j(r).
    l : int
        Partial-wave angular momentum ℓ of the scattering electron.
    E_eV : float
        Kinetic energy (in eV) of the scattering electron in this channel.
        Must satisfy E_eV > 0 (open channel). If <= 0, the resulting
        k would be imaginary and physical cross section is 0.
        Higher-level logic should check k > 0.
    z_ion : float
        Ionic charge seen by the scattering electron at infinity.
        If z_ion=0 (Neutral target), uses Plane Wave (Bessel) matching.
        If z_ion>0 (Ionic target), uses Coulomb Wave matching (required for
        correct long-range phase for ions as in the article).
    tail_fraction : float
        Fraction (0<tail_fraction<1) of the largest-r interval of the grid
        to use for extracting the asymptotic phase. Default 0.1 → last 10%
        of grid points.
        We assume that in that region U_j(r) is already negligible.
    rtol, atol : float
        Relative / absolute tolerances for solve_ivp. We keep them fairly
        strict because χ_l will later enter delicate radial integrals.

    Returns
    -------
    cw : ContinuumWave
        Dataclass with l, k_au, normalized chi_of_r, phase_shift.

    Raises
    ------
    ValueError
        If E_eV <= 0 or if shapes mismatch.
    RuntimeError
        If the ODE solver fails or the tail fit is degenerate.

    Notes
    -----
    The resulting χ_l(k,r) is *real* and asymptotically matches a real
    sine with phase shift. This matches the real distorted waves χ^{(+)}
    / χ^{(-)} in the DWBA amplitude formulas in the article
    (they differ by incoming/outgoing boundary conditions, i.e. sign of
    the infinitesimal imaginary part, but for the *magnitude of matrix
    elements* and for σ we only need the properly normalized radial parts).
    """
    if E_eV <= 0.0:
        raise ValueError("solve_continuum_wave: channel energy must be >0 eV (open channel).")

    r = grid.r
    U_arr = U_channel.U_of_r
    if U_arr.shape != r.shape:
        raise ValueError("solve_continuum_wave: U_channel/grid shape mismatch.")

    # compute k in a.u.
    k_au = float(k_from_E_eV(E_eV))
    if k_au <= 0.0 or not np.isfinite(k_au):
        raise ValueError("solve_continuum_wave: invalid k from given E_eV.")

    # spline interpolation of U(r) so we can evaluate it at arbitrary r during integration
    U_spline = CubicSpline(r, U_arr, bc_type="natural")

    # ==========================================================================
    # DYNAMIC BYPASS CHECK BASED ON r_m
    # ==========================================================================
    # Instead of fixed L threshold, check if potential is negligible at small r.
    # If r_m (where |2U| << k²) is found very early in the grid,
    # numerical integration is unnecessary - use analytic solution.
    #
    # This makes the bypass decision physical rather than heuristic.
    # ==========================================================================
    
    # Try to find match point - if it's at very small r, potential is negligible everywhere
    idx_match_check, r_m_check = _find_match_point(r, U_arr, k_au, l, threshold=1e-4)
    
    # If r_m is found in the first 10% of grid, potential is negligible → use analytic
    fraction_to_match = idx_match_check / len(r)
    use_analytic_bypass = fraction_to_match < 0.15  # If match point is in first 15% of grid
    
    if use_analytic_bypass:
        logger.debug(f"L={l}: Analytic bypass (r_m={r_m_check:.1f} at {fraction_to_match*100:.0f}% of grid)")
        
        if abs(z_ion) < 1e-3:
            # Neutral: use Riccati-Bessel
            rho_vec = k_au * r
            jl_vals = spherical_jn(l, rho_vec)
            chi_analytic = r * jl_vals * k_au
            return ContinuumWave(l, k_au, chi_analytic, 0.0)
        else:
            # Ionic: use asymptotic Coulomb
            eta = -z_ion / k_au
            coulomb_phase = np.imag(loggamma(l + 1 + 1j * eta))
            rho_vec = k_au * r
            theta = rho_vec + eta * np.log(2.0 * rho_vec + 1e-30) - (l * np.pi / 2.0) + coulomb_phase
            chi_coulomb = np.sin(theta)
            # Zero below turning point (asymptotic invalid there)
            r_turning = np.sqrt(l * (l + 1)) / k_au
            chi_coulomb[r < 0.5 * r_turning] = 0.0
            return ContinuumWave(l, k_au, chi_coulomb, 0.0)


    # The centrifugal barrier l(l+1)/r^2 is huge at small r.
    # The solution is effectively zero until we approach the classical turning point r_c ~ sqrt(l(l+1))/k.
    # Integrating from r_min (e.g. 0.05) when r_c is 12 (for l=100) causes extreme stiffness and failure.
    
    # --- Solver Selection ---
    # Use Johnson log-derivative for stability in tunneling region.
    # Empirically: RK45 becomes unstable around L > 10 due to turning point effects.
    # For low k (low energy), instability starts even earlier.
    #
    # NOW USING NUMEROV AS PRIMARY METHOD - more stable chi propagation
    # ==========================================================================
    
    idx_start = 0
    if l > 5:
        r_turn = np.sqrt(l*(l+1)) / k_au
        r_safe = r_turn * 0.9
        idx_found = np.searchsorted(r, r_safe)
        if idx_found > 0 and idx_found < len(r) - 20: 
            idx_start = idx_found
            
    r_eval = r[idx_start:]
    r0_int = float(r_eval[0])
    
    # ==========================================================================
    # NUMEROV PROPAGATION (Primary Method)
    # ==========================================================================
    # Numerov is O(h⁶) accurate and more stable than log-derivative 
    # reconstruction. Uses Q(r) = l(l+1)/r² + 2U(r) - k².
    # ==========================================================================
    
    # Build Q(r) array for Numerov
    ell = float(l)
    k2 = k_au ** 2
    Q_full = ell * (ell + 1.0) / (r * r) + 2.0 * U_arr - k2
    Q_eval = Q_full[idx_start:]
    
    # Initial conditions: χ ~ r^(l+1) at small r
    # χ[0] = r[0]^(l+1), χ[1] = r[1]^(l+1)
    if idx_start == 0:
        # Start from origin with regular boundary condition
        chi0 = r_eval[0] ** (ell + 1.0)
        chi1 = r_eval[1] ** (ell + 1.0)
    else:
        # Start near turning point - use WKB-like initial conditions
        # In barrier: χ grows exponentially outward
        r0 = r_eval[0]
        r1 = r_eval[1]
        h_init = r1 - r0
        S0 = ell * (ell + 1.0) / (r0 * r0) + 2.0 * U_arr[idx_start] - k2
        
        if S0 > 0:
            # Inside barrier - exponentially growing solution
            kappa = np.sqrt(S0)
            chi0 = 1e-20  # Small amplitude
            chi1 = chi0 * np.exp(kappa * h_init)
        else:
            # Already outside barrier - oscillatory
            chi0 = r0 ** (ell + 1.0) if ell < 10 else 1e-10
            chi1 = r1 ** (ell + 1.0) if ell < 10 else 1e-10 * 1.1
    
    # Propagate using Numerov
    chi_computed, log_scale = _numerov_propagate(
        r_eval, Q_eval, chi0, chi1, 
        renorm_interval=100, renorm_scale=1e50
    )
    
    # Compute derivative using central difference with local step sizes
    dchi_computed = np.zeros_like(chi_computed)
    for i in range(len(chi_computed)):
        dchi_computed[i] = _derivative_5point(chi_computed, r_eval, i)
    
    method_used = "Numerov"
    
    # Place in full grid
    if idx_start > 0:
        chi_raw = np.zeros_like(r, dtype=float)
        chi_raw[idx_start:] = chi_computed
        dchi_raw = np.zeros_like(r, dtype=float)
        dchi_raw[idx_start:] = dchi_computed
    else:
        chi_raw = chi_computed
        dchi_raw = dchi_computed


    # Sanity: remove any global sign if necessary (not physically important).
    # We won't flip sign here because the asymptotic fit will absorb it into δ_l anyway.

    # ==========================================================================
    # PHASE EXTRACTION VIA LOG-DERIVATIVE MATCHING
    # ==========================================================================
    #
    # Instead of fitting over an entire tail region, we match at a single point
    # r_m where the potential is negligible. This is more stable and accurate.
    #
    # Formula: tan(δ_l) = [Y_m · j_l - j_l'] / [n_l' - Y_m · n_l]
    # where Y_m = χ'(r_m) / χ(r_m) is the log-derivative from numerical propagation.
    #
    # ==========================================================================
    
    # --- Step 1: Find optimal match point (must be beyond idx_start) ---
    idx_match, r_m = _find_match_point(r, U_arr, k_au, l, threshold=1e-4, idx_start=idx_start)
    
    # Get log-derivative at match point
    chi_m = chi_raw[idx_match]
    dchi_m = dchi_raw[idx_match]
    
    if abs(chi_m) < 1e-100:
        # =======================================================================
        # FALLBACK CHAIN: Numerov failed -> Try Johnson -> Try RK45
        # =======================================================================
        logger.debug(f"L={l}: Numerov produced χ≈0, trying Johnson fallback")
        
        # --- Fallback 1: Johnson log-derivative method ---
        U_eval = U_arr[idx_start:]
        r_eval_fb = r[idx_start:]
        chi_johnson, dchi_johnson = _johnson_log_derivative_solve(
            r_eval_fb, U_eval, l, k_au, renorm_interval=50
        )
        
        if idx_start > 0:
            chi_raw = np.zeros_like(r, dtype=float)
            chi_raw[idx_start:] = chi_johnson
            dchi_raw = np.zeros_like(r, dtype=float)
            dchi_raw[idx_start:] = dchi_johnson
        else:
            chi_raw = chi_johnson
            dchi_raw = dchi_johnson
        
        chi_m = chi_raw[idx_match]
        dchi_m = dchi_raw[idx_match]
        method_used = "Johnson (fallback)"
        
        if abs(chi_m) < 1e-100:
            # --- Fallback 2: RK45 ---
            logger.debug(f"L={l}: Johnson also produced χ≈0, trying RK45")
            
            r0_int = float(r[idx_start])
            r_eval_rk = r[idx_start:]
            
            if abs(z_ion) < 1e-3:
                y0 = _initial_conditions_high_L(r0_int, l, k_au, z_ion=0.0)
            else:
                y0 = _initial_conditions_high_L(r0_int, l, k_au, z_ion=z_ion)
            
            rhs = _schrodinger_rhs_factory(l=l, U_spline=U_spline, k_au=k_au)
            
            wavelength = 2.0 * np.pi / k_au if k_au > 1e-3 else 1.0
            max_step_val = min(wavelength / 20.0, 0.2)
            
            sol = solve_ivp(
                fun=rhs,
                t_span=(r0_int, float(r_eval_rk[-1])),
                y0=y0,
                t_eval=r_eval_rk,
                method="RK45",
                max_step=max_step_val,
                rtol=rtol,
                atol=atol,
                dense_output=False
            )
            
            if sol.success:
                chi_raw[idx_start:] = sol.y[0, :]
                dchi_raw[idx_start:] = sol.y[1, :]
                method_used = "RK45 (fallback)"
                
                chi_m = chi_raw[idx_match]
                dchi_m = dchi_raw[idx_match]
                
                if abs(chi_m) < 1e-100:
                    logger.warning(f"L={l}: All solvers (Numerov, Johnson, RK45) failed")
                    return None
            else:
                logger.warning(f"L={l}: RK45 fallback also failed (solver error)")
                return None


    
    Y_m = dchi_m / chi_m
    
    # --- Step 2: Extract phase using log-derivative matching ---
    if abs(z_ion) < 1e-3:
        delta_l = _extract_phase_logderiv_neutral(Y_m, k_au, r_m, l)
    else:
        delta_l = _extract_phase_logderiv_coulomb(Y_m, k_au, r_m, l, z_ion)
    
    # --- Step 3: Compute amplitude for normalization ---
    # At match point: χ = A·[ĵ cos(δ) - n̂ sin(δ)]
    # So: A = χ / [ĵ cos(δ) - n̂ sin(δ)]
    rho_m = k_au * r_m
    
    if abs(z_ion) < 1e-3:
        j_hat, _ = _riccati_bessel_jn(l, rho_m)
        n_hat, _ = _riccati_bessel_yn(l, rho_m)
        ref_value = j_hat * np.cos(delta_l) - n_hat * np.sin(delta_l)
    else:
        eta = -z_ion / k_au
        F, _, G, _ = _coulomb_FG_asymptotic(l, eta, rho_m)
        ref_value = F * np.cos(delta_l) - G * np.sin(delta_l)
    
    if abs(ref_value) < 1e-100:
        logger.warning(f"L={l}: Reference value ≈ 0, using |χ(r_m)| for normalization")
        A_amp = abs(chi_m)
    else:
        A_amp = chi_m / ref_value
    
    # --- Diagnostics ---
    # Check phase stability by comparing with slightly shifted match point
    if idx_match > 10:
        idx_alt = idx_match - 5
        chi_alt = chi_raw[idx_alt]
        dchi_alt = dchi_raw[idx_alt]
        if abs(chi_alt) > 1e-100:
            Y_alt = dchi_alt / chi_alt
            r_alt = r[idx_alt]
            if abs(z_ion) < 1e-3:
                delta_alt = _extract_phase_logderiv_neutral(Y_alt, k_au, r_alt, l)
            else:
                delta_alt = _extract_phase_logderiv_coulomb(Y_alt, k_au, r_alt, l, z_ion)
            
            phase_variation = abs(delta_l - delta_alt)
            phase_stable = phase_variation <= 0.05
            if not phase_stable:
                logger.warning(f"Phase unstable for L={l}: δ varies by {phase_variation:.4f} rad")
        else:
            phase_stable = True
    else:
        phase_stable = True
    
    # Log diagnostics
    logger.debug(f"L={l:3d} [{method_used}] r_m={r_m:.1f} Y_m={Y_m:.4f} δ={delta_l:+.4f} A={A_amp:.2e}")

    if not np.isfinite(A_amp) or not np.isfinite(delta_l) or abs(A_amp) < 1e-100:
        logger.warning(f"L={l}: Unreliable wave (A={A_amp:.2e}, δ={delta_l:.4f})")
        return None

    # ==========================================================================
    # ASYMPTOTIC STITCHING
    # ==========================================================================
    # Scale numerical chi to match asymptotic amplitude, then stitch to
    # analytic solution for r > r_m. This eliminates numerical noise in 
    # the tail and ensures proper normalization to δ(k-k').
    # ==========================================================================
    
    # Build asymptotic wave with sqrt(2/π) normalization
    chi_asymptotic = _build_asymptotic_wave(r, k_au, l, delta_l, z_ion, idx_start=0)
    
    # Scale factor: s = chi_as(r_m) / chi_num(r_m)
    # Use robust formula when chi_m might be close to node
    chi_as_m = chi_asymptotic[idx_match]
    chi_num_m = chi_raw[idx_match]
    
    if abs(chi_num_m) > 1e-100:
        # More robust scaling using both chi and chi'
        dchi_as_m = _derivative_5point(chi_asymptotic, r, idx_match)
        dchi_num_m = dchi_raw[idx_match]
        
        # s = [chi_as·chi_num + chi_as'·chi_num'] / [chi_num² + chi_num'²]
        numerator = chi_as_m * chi_num_m + dchi_as_m * dchi_num_m
        denominator = chi_num_m**2 + dchi_num_m**2
        
        if abs(denominator) > 1e-100:
            scale = numerator / denominator
        else:
            scale = chi_as_m / chi_num_m if abs(chi_num_m) > 1e-100 else 1.0
    else:
        logger.warning(f"L={l}: chi_num(r_m) ≈ 0, using asymptotic wave only")
        scale = 1.0
    
    # Build final stitched wavefunction
    chi_final = np.zeros_like(r, dtype=float)
    
    # For r <= r_m: use scaled numerical solution
    chi_final[:idx_match + 1] = scale * chi_raw[:idx_match + 1]
    
    # For r > r_m: use pure analytic asymptotic solution
    chi_final[idx_match + 1:] = chi_asymptotic[idx_match + 1:]
    
    # Done. Package result with match point for split integrals.
    cw = ContinuumWave(
        l=l,
        k_au=k_au,
        chi_of_r=chi_final,
        phase_shift=delta_l,
        r_match=r_m,
        idx_match=idx_match
    )
    return cw
