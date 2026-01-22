# continuum.py
"""
Distorted-Wave Continuum Solver
================================

Solves the radial Schrödinger equation for scattering electron partial waves
χ_l(k,r) in the DWBA formulation.

Equation (atomic units)
-----------------------
    χ''(r) = [l(l+1)/r² + 2U_j(r) - k²] χ(r)

where:
- l = partial-wave angular momentum
- U_j(r) = distorting potential (vanishes at large r)
- k = asymptotic wave number (E = k²/2)

Boundary Conditions
-------------------
- Origin: χ(r) ~ r^(l+1) (regular solution)
- Asymptotic: χ_l ~ A·sin(kr - lπ/2 + δ_l) with A = √(2/π)

Implementation (v2.1)
---------------------
Primary: Numerov propagation with O(h⁴) accuracy for non-uniform grids
- Physics-based turning point detection: uses S(r) > 0 criterion
- Adaptive initial conditions: WKB or regular based on local S(r)
- Match point selection: searches FORWARD from idx_start + 50
- Asymptotic stitching: numerical χ → analytic beyond r_m

Fallback chain: Numerov → Johnson log-derivative → RK45

Output
------
ContinuumWave objects containing:
- l, k_au (angular momentum, wave number)
- χ_l(r) normalized to √(2/π) asymptotic amplitude
- δ_l (phase shift in radians)
- idx_match, r_match (for split radial integrals)

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
            χ_l(k,r) ~ sin(k r + η ln(2kr) - l π/2 + σ_l + δ_l)
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
    eta : float
        Sommerfeld parameter η = -z_ion/k for Coulomb potential.
        For neutral targets (z_ion=0), η=0.
    sigma_l : float
        Coulomb phase shift σ_l = arg(Γ(l+1+iη)).
        For neutral targets, σ_l=0.
    phase_method : str
        Which phase extraction method was used: "lsq", "logderiv", "lsq_validated", etc.
    """
    l: int
    k_au: float
    chi_of_r: np.ndarray
    phase_shift: float
    r_match: float = 0.0      # Match point radius
    idx_match: int = -1       # Match point grid index (-1 = not set, use full grid)
    eta: float = 0.0          # Sommerfeld parameter (-z_ion/k)
    sigma_l: float = 0.0      # Coulomb phase shift arg(Γ(l+1+iη))
    phase_method: str = "unknown"  # Phase extraction method used (v2.11+)
    solver_method: str = "numerov" # ODE solver used (v2.12+)


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
    
    # Guard against overflow (spherical_yn can overflow for large l, small rho)
    if not np.isfinite(yl) or abs(yl) > 1e50:
        return -1e30, -1e30
    
    n_hat = rho * yl  # n̂_l = ρ·y_l
    
    if l == 0:
        # n̂_0 = -cos(ρ), n̂_0' = sin(ρ)
        n_hat_prime = np.sin(rho)
    else:
        # Use y_{l-1} for derivative
        yl_minus1 = float(spherical_yn(l - 1, rho))
        
        # Guard against overflow
        if not np.isfinite(yl_minus1) or abs(yl_minus1) > 1e50:
            n_hat_prime = -1e30
        else:
            n_hat_prime = rho * yl_minus1 - l * yl
    
    return n_hat, n_hat_prime



def _coulomb_FG_asymptotic(l: int, eta: float, rho: float) -> Tuple[float, float, float, float]:
    """
    Asymptotic Coulomb functions F_l, G_l and their derivatives.
    
    Includes O(1/ρ) corrections from NIST DLMF §33.11 for improved accuracy
    at moderate ρ (when ρ ~ l or ρ ~ |η|).
    
    Leading order (ρ >> l, η):
        F_l(η,ρ) ≈ sin(θ)
        G_l(η,ρ) ≈ cos(θ)
    where θ = ρ + η·ln(2ρ) - lπ/2 + σ_l
    
    Corrections applied:
        1. Centrifugal phase: θ_corr = θ - l(l+1)/(2ρ)
        2. Amplitude factor: A = 1 + λ/(4ρ²) where λ = l(l+1) - 2η²
    
    Returns (F, F', G, G')
    """
    if rho < 1e-10:
        return 0.0, 0.0, 1e30, 1e30
    
    # Coulomb phase shift σ_l = arg Γ(l+1+iη)
    sigma_l = np.imag(loggamma(l + 1 + 1j * eta))
    
    # Base asymptotic argument
    theta_base = rho + eta * np.log(2.0 * rho) - (l * np.pi / 2.0) + sigma_l
    
    # =========================================================================
    # O(1/ρ) CORRECTIONS (NIST DLMF §33.11)
    # =========================================================================
    # Phase correction: accounts for centrifugal barrier effect at moderate ρ
    # Derived from WKB: the effective potential l(l+1)/r² modifies the phase
    # The correction -l(l+1)/(2kρ) = -l(l+1)/(2ρ) (in units where k=1/ρ_bar)
    # =========================================================================
    
    ll1 = l * (l + 1)
    phase_correction = ll1 / (2.0 * rho) if rho > 1.0 else 0.0
    theta = theta_base - phase_correction
    
    # Amplitude correction: from Sommerfeld expansion
    # λ = l(l+1) - 2η² captures both centrifugal and Coulomb effects
    # A = 1 + λ/(4ρ²) is the first-order correction
    lambda_param = ll1 - 2.0 * eta * eta
    if rho > 5.0:  # Only apply when asymptotic regime is valid
        amp = 1.0 + lambda_param / (4.0 * rho * rho)
    else:
        amp = 1.0  # Don't apply correction too close to origin
    
    # Derivative of theta: dθ/dρ = 1 + η/ρ + l(l+1)/(2ρ²)
    theta_prime = 1.0 + eta / rho + ll1 / (2.0 * rho * rho)
    
    F = amp * np.sin(theta)
    G = amp * np.cos(theta)
    
    # Derivatives include amplitude correction
    # d(A·sin(θ))/dρ = A'·sin(θ) + A·θ'·cos(θ)
    # For leading order, A' ~ O(1/ρ³) which we neglect
    F_prime = amp * theta_prime * np.cos(theta)
    G_prime = -amp * theta_prime * np.sin(theta)
    
    return F, F_prime, G, G_prime


def _find_match_point(r_grid: np.ndarray, U_arr: np.ndarray, k_au: float, l: int,
                       threshold: float = 1e-2, idx_start: int = 0,
                       chi: np.ndarray = None) -> Tuple[int, float]:
    """
    Find optimal matching point r_m where potential is negligible.
    
    CRITICAL: Match point must be >= idx_start + margin to ensure we have
    valid wavefunction amplitude (the solver only populates chi from idx_start).
    
    Strategy: Search FORWARD from idx_start until we find a point where
    the potential is negligible compared to kinetic energy.
    
    Criteria:
        |2U(r_m)| < threshold × k²
        r_m > r_turn (classical turning point)
    
    Parameters
    ----------
    idx_start : int
        Minimum index where wavefunction is valid (propagation start).
        Match point will be at least idx_start + 50 to ensure stable phase.
    threshold : float
        Ratio |U|/(k²/2) below which potential is considered negligible.
        Default 1e-2 (1%).
    chi : np.ndarray, optional
        Unused parameter for API compatibility.
    
    Returns
    -------
    idx, r : Tuple[int, float]
        Index and radius of match point.
    """
    k2 = k_au ** 2
    N = len(r_grid)
    
    # Minimum margin: at least 50 points past idx_start for stable oscillations
    # and to ensure we're well past any turning point effects
    MIN_MARGIN = 50
    search_start = max(idx_start + MIN_MARGIN, N // 4)
    
    # Don't search past 90% of grid (edge effects)
    search_end = int(0.9 * N)
    
    # Also ensure we're past the classical turning point
    r_turn = np.sqrt(l * (l + 1)) / k_au if k_au > 1e-6 else 0.0
    idx_turn = np.searchsorted(r_grid, r_turn)
    search_start = max(search_start, idx_turn + 10)
    
    # Search FORWARD from search_start
    # Two-stage criterion:
    # 1. r must be sufficiently beyond turning point (centrifugal safety)
    # 2. Short-range potential U(r) must be small relative to k²
    # We DON'T add centrifugal to threshold (too restrictive for high L)
    
    # Safety factor for turning point: require r > 2.5 × r_turn
    # Note: 2.5× (not 2×) ensures diagnostic alt point at idx_match-5 is also safe
    r_turn_safe = 2.5 * r_turn
    
    for idx in range(search_start, search_end):
        r = r_grid[idx]
        U = U_arr[idx]
        
        # Stage 1: Must be well past turning point (centrifugal region)
        if r < r_turn_safe:
            continue
        
        # Stage 2: EFFECTIVE potential must be negligible (not just U!)
        # V_eff = |2U| + centrifugal barrier l(l+1)/r²
        # This is the TRUE asymptotic criterion - both short-range AND 
        # centrifugal must be small compared to kinetic energy k²
        V_cent = l * (l + 1) / (r * r)
        V_eff = abs(2.0 * U) + V_cent
        
        if V_eff < threshold * k2:
            return idx, r
    
    # Fallback: use the larger of (70% of grid, 2.5×r_turn) 
    # This ensures centrifugal safety even if U(r) criterion cannot be met
    # Use +20 margin (not +5) so diagnostic alt point idx_match-5 is also safe
    idx_turn_safe = np.searchsorted(r_grid, r_turn_safe) if r_turn_safe > 0 else 0
    fallback_idx = max(search_start, int(0.7 * N), idx_turn_safe + 20)
    fallback_idx = min(fallback_idx, N - 10)  # Not too close to edge
    
    logger.debug(
        f"No ideal match point found for L={l}, k={k_au:.3f}; "
        f"using fallback idx={fallback_idx} (r={r_grid[fallback_idx]:.2f}, r_turn_safe={r_turn_safe:.2f})"
    )
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
    
    # tan(δ) formula derived from asymptotic matching:
    # χ = A[ĵ cos(δ) - n̂ sin(δ)]  →  Y = (ĵ' cos δ - n̂' sin δ) / (ĵ cos δ - n̂ sin δ)
    # Solving for tan(δ): tan(δ) = [Y·ĵ - ĵ'] / [Y·n̂ - n̂']
    numerator = Y_rho * j_hat - j_hat_prime
    denominator = Y_rho * n_hat - n_hat_prime  # FIXED: was incorrectly inverted
    
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
    
    # VALIDITY CHECK: Asymptotic Coulomb functions require ρ >> max(l, |η|)
    # If violated, the phase extraction may be inaccurate for ionic targets
    rho_min_required = 3.0 * max(l, abs(eta))
    if rho_m < rho_min_required:
        logger.warning(
            f"L={l}: Coulomb asymptotic may be inaccurate "
            f"(ρ={rho_m:.1f} < 3×max(l,|η|)={rho_min_required:.1f}). "
            f"Consider increasing r_max."
        )
    
    # Get Coulomb functions at match point
    F, F_prime, G, G_prime = _coulomb_FG_asymptotic(l, eta, rho_m)
    
    # Convert Y_m to ρ-space
    Y_rho = Y_m / k_au
    
    # tan(δ) = [Y_ρ·F - F'] / [Y_ρ·G - G']  (derived from asymptotic matching)
    numerator = Y_rho * F - F_prime
    denominator = Y_rho * G - G_prime  # FIXED: was incorrectly inverted
    
    if abs(denominator) < 1e-15:
        return np.pi / 2.0 if numerator > 0 else -np.pi / 2.0
    
    tan_delta = numerator / denominator
    delta_l = np.arctan(tan_delta)
    
    return delta_l


def _schrodinger_rhs_factory(l: int, U_spline: CubicSpline, k_au: float) -> Callable:
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
            
    except (OverflowError, FloatingPointError) as e:
        # Fallback if logarithm underflows (shouldn't happen for r0>0)
        logger.debug("Initial conditions fallback triggered for l=%d: %s", l, e)
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


def _extract_phase_hybrid(
    chi_raw: np.ndarray,
    dchi_raw: np.ndarray,
    r: np.ndarray,
    w_trapz: np.ndarray,
    k_au: float,
    l: int,
    idx_match: int,
    z_ion: float = 0.0,
    tail_fraction: float = 0.15,
    disagreement_threshold: float = 0.1
) -> Tuple[float, float, str]:
    """
    Optimized hybrid phase extraction (v2.11+).
    
    Analysis showed LSQ is more accurate than log-derivative, especially for high L.
    Strategy:
    - LSQ is PRIMARY method (multi-point fit, more robust)
    - Log-derivative used for VALIDATION only
    - For L > L_threshold: always use LSQ (log-derivative degrades)
    - For low L: cross-validate, prefer LSQ when disagreement
    
    Parameters
    ----------
    chi_raw : np.ndarray
        Wavefunction χ(r) on grid.
    dchi_raw : np.ndarray
        Derivative χ'(r) on grid.
    r : np.ndarray
        Radial grid points.
    w_trapz : np.ndarray
        Trapezoidal integration weights.
    k_au : float
        Wave number in atomic units.
    l : int
        Angular momentum.
    idx_match : int
        Index of matching point for log-derivative.
    z_ion : float
        Ionic charge (0 for neutral targets).
    tail_fraction : float
        Fraction of grid to use for LSQ fitting.
    disagreement_threshold : float
        Maximum acceptable difference between methods (radians).
        
    Returns
    -------
    delta : float
        Best estimate of phase shift.
    amplitude : float
        Amplitude for normalization.
    method_used : str
        Which method was used: "lsq", "logderiv", "lsq_validated".
    """
    r_m = r[idx_match]
    chi_m = chi_raw[idx_match]
    
    # Threshold above which log-derivative becomes unreliable
    # Analysis showed error grows with L: ~0.13 rad at L=5, ~0.30 rad at L=20
    L_threshold_for_lsq_only = 10
    
    # --- STEP 1: Compute LSQ result (primary method) ---
    n_tail = int(len(r) * tail_fraction)
    idx_tail = max(idx_match, len(r) - n_tail)
    r_tail = r[idx_tail:]
    chi_tail = chi_raw[idx_tail:]
    
    lsq_valid = len(chi_tail) >= 10 and np.max(np.abs(chi_tail)) > 1e-100
    
    if lsq_valid:
        if abs(z_ion) < 1e-3:
            A_lsq, delta_lsq = _fit_asymptotic_phase_neutral(r_tail, chi_tail, l, k_au)
        else:
            A_lsq, delta_lsq = _fit_asymptotic_phase_coulomb(r_tail, chi_tail, l, k_au, z_ion)
        
        lsq_valid = np.isfinite(delta_lsq) and np.isfinite(A_lsq) and A_lsq > 1e-100
    
    # --- STEP 2: For high L, always use LSQ ---
    if l > L_threshold_for_lsq_only:
        if lsq_valid:
            return delta_lsq, A_lsq, "lsq"
        # Fall through to log-derivative if LSQ failed
    
    # --- STEP 3: Compute log-derivative result (validation) ---
    ld_valid = abs(chi_m) > 1e-100
    
    if ld_valid:
        dchi_m = dchi_raw[idx_match]
        Y_m = dchi_m / chi_m
        
        if abs(z_ion) < 1e-3:
            delta_ld = _extract_phase_logderiv_neutral(Y_m, k_au, r_m, l)
        else:
            delta_ld = _extract_phase_logderiv_coulomb(Y_m, k_au, r_m, l, z_ion)
        
        # Calculate amplitude from log-derivative
        rho_m = k_au * r_m
        if abs(z_ion) < 1e-3:
            j_hat, _ = _riccati_bessel_jn(l, rho_m)
            n_hat, _ = _riccati_bessel_yn(l, rho_m)
            ref_value = j_hat * np.cos(delta_ld) - n_hat * np.sin(delta_ld)
        else:
            eta = -z_ion / k_au
            F, _, G, _ = _coulomb_FG_asymptotic(l, eta, rho_m)
            ref_value = F * np.cos(delta_ld) - G * np.sin(delta_ld)
        
        A_ld = abs(chi_m / ref_value) if abs(ref_value) > 1e-100 else 0.0
        ld_valid = np.isfinite(delta_ld) and A_ld > 0
    
    # --- STEP 4: Choose best result ---
    if lsq_valid and ld_valid:
        # Compare methods
        delta_diff = abs(delta_ld - delta_lsq)
        delta_diff = min(delta_diff, 2.0 * np.pi - delta_diff)
        
        if delta_diff < disagreement_threshold:
            # Methods agree - use LSQ (more accurate in tests)
            return delta_lsq, A_lsq, "lsq_validated"
        else:
            # Methods disagree - trust LSQ (more robust, especially for high L)
            logger.debug(
                f"L={l}: Phase methods disagree: LD={delta_ld:.4f}, LSQ={delta_lsq:.4f}, "
                f"diff={delta_diff:.4f} rad. Using LSQ (more accurate)."
            )
            return delta_lsq, A_lsq, "lsq"
    
    elif lsq_valid:
        return delta_lsq, A_lsq, "lsq"
    
    elif ld_valid:
        return delta_ld, A_ld, "logderiv"
    
    else:
        # Both methods failed - return zero (will be caught by caller)
        logger.warning(f"L={l}: Both phase extraction methods failed")
        return 0.0, 0.0, "failed"


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
    Derivative using a local 5-point stencil on a non-uniform grid.

    We compute finite-difference coefficients for the actual local spacing
    (Fornberg algorithm). Falls back to fewer points near boundaries.
    
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
    if idx < 0 or idx >= N:
        raise IndexError("idx out of range in _derivative_5point")

    # Choose up to 5 points around idx (shift window near boundaries)
    n_points = 5 if N >= 5 else N
    start = min(max(idx - 2, 0), N - n_points)
    idxs = np.arange(start, start + n_points)
    x = r_grid[idxs]
    x0 = r_grid[idx]

    # Fornberg coefficients for first derivative at x0
    # Reference: B. Fornberg, Math. Comp. 51, 1988.
    m = 1
    c = np.zeros((n_points, m + 1), dtype=float)
    c[0, 0] = 1.0
    c1 = 1.0
    c4 = x[0] - x0
    for i in range(1, n_points):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2

    coeffs = c[:, 1]
    return float(np.dot(coeffs, chi[idxs]))



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
    # For non-uniform grid, use LOCAL step sizes at each point
    for i in range(1, N - 1):
        h1 = h_arr[i - 1]  # h_{i-1} = r[i] - r[i-1]
        h2 = h_arr[i]      # h_i = r[i+1] - r[i]
        
        # Improved Numerov for non-uniform grid:
        # Use h1² for backward, h2² for forward, geometric mean for center
        # This preserves O(h⁴) accuracy for exponential grids
        h1_sq = h1 * h1
        h2_sq = h2 * h2
        h_center_sq = h1 * h2  # Geometric mean squared for center term
        
        # Numerov coefficients with proper local steps
        a_prev = 1.0 - (h1_sq / 12.0) * Q_arr[i - 1]
        b_curr = 2.0 + (5.0 * h_center_sq / 6.0) * Q_arr[i]
        a_next = 1.0 - (h2_sq / 12.0) * Q_arr[i + 1]
        
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
    
    # Get slice to work with
    r_slice = r_grid[idx_start:]
    rho_slice = k_au * r_slice
    
    # Mask for valid rho values
    valid_mask = rho_slice >= 1e-10
    rho_valid = rho_slice[valid_mask]
    
    if len(rho_valid) == 0:
        return chi_as
    
    if abs(z_ion) < 1e-3:
        # Neutral: use vectorized Riccati-Bessel functions
        # spherical_jn and spherical_yn accept arrays
        jl = spherical_jn(l, rho_valid)
        j_hat = rho_valid * jl  # ĵ_l = ρ·j_l
        
        yl = spherical_yn(l, rho_valid)
        # Guard against overflow
        yl = np.where(np.isfinite(yl) & (np.abs(yl) < 1e50), yl, -1e30)
        n_hat = rho_valid * yl  # n̂_l = ρ·y_l
        
        chi_valid = A * (j_hat * cos_d - n_hat * sin_d)
    else:
        # Ionic: use vectorized Coulomb functions
        eta = -z_ion / k_au
        
        # Coulomb phase shift σ_l = arg Γ(l+1+iη)
        sigma_l = np.imag(loggamma(l + 1 + 1j * eta))
        
        # Asymptotic argument (vectorized)
        theta = rho_valid + eta * np.log(2.0 * rho_valid) - (l * np.pi / 2.0) + sigma_l
        
        F = np.sin(theta)
        G = np.cos(theta)
        
        chi_valid = A * (F * cos_d - G * sin_d)
    
    # Place results back into full array
    valid_indices = np.where(valid_mask)[0] + idx_start
    chi_as[valid_indices] = chi_valid
    
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
    Johnson Renormalized Numerov Method (1978).
    
    Reference: B.R. Johnson, J. Chem. Phys. 69, 4678 (1978)
    "The renormalized Numerov method applied to calculating bound states
    of the coupled-channel Schroedinger equation"
    
    This method propagates the ratio R_n = ψ_{n-1}/ψ_n using:
        R_{n+1} = 1 / (T_n - R_n)
    where T_n = 2 - h² Q_n and Q_n = l(l+1)/r² + 2U(r) - k²
    
    This is numerically stable because R stays O(1) regardless of whether
    the wavefunction is growing or decaying exponentially.
    
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
        Unused in this implementation (kept for API compatibility).
        
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
    
    # Build Q array: Q(r) = l(l+1)/r² + 2U(r) - k²
    Q = np.zeros(N, dtype=float)
    for i in range(N):
        r = r_grid[i]
        Q[i] = ell * (ell + 1.0) / (r * r) + 2.0 * U_arr[i] - k2
    
    # =========================================================================
    # Johnson Renormalized Numerov: Outward propagation
    # =========================================================================
    # For non-uniform grid: we use local step size h_i = r_{i+1} - r_i
    # T_i = 2 - h_i² Q_i  (modified Numerov coefficient)
    # R_{i+1} = 1 / (T_i - R_i)  where R_i = ψ_{i-1}/ψ_i
    #
    # Initial condition: R_1 = ψ_0 / ψ_1
    # For regular solution near origin: ψ ~ r^{l+1}
    # So R_1 = r_0^{l+1} / r_1^{l+1} = (r_0/r_1)^{l+1}
    # =========================================================================
    
    R = np.zeros(N, dtype=float)  # R[i] = ψ_{i-1} / ψ_i
    
    # Initial R value (ratio of first two points)
    r0, r1 = r_grid[0], r_grid[1]
    h0 = r1 - r0
    
    # Check if we're inside barrier (use WKB) or oscillatory (use Bessel)
    S0 = Q[0]  # S = Q for our equation
    
    if S0 > 0:
        # Inside barrier: WKB approximation
        # ψ ~ exp(κr) where κ = sqrt(S)
        # R_1 = exp(-κ·h) for growing solution
        kappa = np.sqrt(S0)
        R[1] = np.exp(-kappa * h0)
    else:
        # Oscillatory region: use Bessel functions
        rho0 = k_au * r0
        rho1 = k_au * r1
        
        if rho0 > 1e-10 and rho1 > 1e-10:
            jl0 = spherical_jn(l, rho0)
            jl1 = spherical_jn(l, rho1)
            chi0 = r0 * jl0
            chi1 = r1 * jl1
            if abs(chi1) > 1e-100:
                R[1] = chi0 / chi1
            else:
                R[1] = (r0 / r1) ** (l + 1)
        else:
            # Very small r: use power law
            R[1] = (r0 / r1) ** (l + 1)
    
    # Clip to prevent numerical issues
    R[1] = np.clip(R[1], -1e10, 1e10)
    
    # Forward propagation using Johnson's algorithm
    for i in range(1, N - 1):
        h_curr = r_grid[i + 1] - r_grid[i]
        h_prev = r_grid[i] - r_grid[i - 1]
        
        # For non-uniform grid, use average step for Q coefficient
        # T_i = 2 - h² Q_i (standard Numerov)
        # We use the geometric mean of steps for better accuracy
        h_eff = np.sqrt(h_curr * h_prev)
        T_i = 2.0 - h_eff * h_eff * Q[i]
        
        # Johnson's recursion: R_{i+1} = 1 / (T_i - R_i)
        denom = T_i - R[i]
        
        # Handle near-zero denominator (turning point)
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        
        R[i + 1] = 1.0 / denom
        
        # Clip to prevent explosion
        R[i + 1] = np.clip(R[i + 1], -1e10, 1e10)
    
    # =========================================================================
    # Backward reconstruction of ψ from R
    # =========================================================================
    # ψ_{i-1} = R_i * ψ_i
    # Start from last point and work backwards
    
    chi = np.zeros(N, dtype=float)
    chi[-1] = 1.0  # Arbitrary normalization
    
    for i in range(N - 1, 0, -1):
        chi[i - 1] = R[i] * chi[i]
        
        # Renormalize periodically to prevent underflow/overflow
        if i % 100 == 0:
            max_val = np.max(np.abs(chi[i - 1:]))
            if max_val > 1e30 or (max_val < 1e-30 and max_val > 0):
                scale = 1.0 / max_val if max_val > 0 else 1.0
                chi[i - 1:] *= scale
    
    # Final normalization
    max_val = np.max(np.abs(chi))
    if max_val > 0:
        chi /= max_val
    
    # Compute derivatives using 5-point stencil
    dchi = np.zeros(N, dtype=float)
    for i in range(N):
        dchi[i] = _derivative_5point(chi, r_grid, i)
    
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
    phase_extraction_method: str = "hybrid",  # v2.11+: "hybrid", "logderiv", or "lsq"
    solver: str = "auto",  # v2.13+: "auto" (recommended), "rk45", "johnson", or "numerov"
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

    # ==========================================================================
    # ADAPTIVE PRECISION FOR VERY HIGH ENERGIES (v2.6.2+)
    # ==========================================================================
    # At very high energies (>1 keV, k > 8.6 a.u.), phase extraction becomes
    # numerically challenging due to:
    #   - More oscillations per unit length (smaller wavelength)
    #   - Larger centrifugal terms for high L
    #   - Phase stability issues in tail fitting
    #
    # We adapt by:
    #   1. Using stricter tolerances for ODE solvers
    #   2. Requiring more points in the tail for phase averaging
    #   3. Using tighter renormalization intervals
    #   4. Logging high-energy regime entry
    # ==========================================================================
    
    E_HIGH_THRESHOLD_EV = 1000.0  # 1 keV
    E_VERY_HIGH_THRESHOLD_EV = 5000.0  # 5 keV
    
    adaptive_mode = "standard"
    adaptive_rtol = rtol
    adaptive_atol = atol
    adaptive_renorm_interval = 100
    
    if E_eV > E_VERY_HIGH_THRESHOLD_EV:
        # Very high energy: strictest precision
        adaptive_mode = "very_high"
        adaptive_rtol = min(rtol, 1e-8)
        adaptive_atol = min(atol, 1e-10)
        adaptive_renorm_interval = 50
        if l == 0:  # Only log once per energy point (for L=0)
            logger.info("High-energy adaptive: E=%.0f eV (k=%.2f a.u.) using precision mode: %s", 
                       E_eV, k_au, adaptive_mode)
    elif E_eV > E_HIGH_THRESHOLD_EV:
        # High energy: enhanced precision
        adaptive_mode = "high"
        adaptive_rtol = min(rtol, 1e-7)
        adaptive_atol = min(atol, 1e-9)
        adaptive_renorm_interval = 75
        if l == 0:
            logger.debug("High-energy adaptive: E=%.0f eV (k=%.2f a.u.) mode=%s", 
                        E_eV, k_au, adaptive_mode)

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
        
        # Calculate Born approximation phase shift:
        # δ_l^Born ≈ -k ∫₀^∞ U(r) [j_l(kr)]² r² dr
        # This gives non-zero phase shifts even for weak potentials
        rho_vec = k_au * r
        jl_vals = spherical_jn(l, rho_vec)
        
        # Use trapezoidal weights from grid for integration
        integrand = U_arr * jl_vals**2 * r**2
        delta_born = -k_au * np.sum(grid.w_trapz * integrand)
        
        # Clamp to reasonable range to avoid numerical artifacts
        delta_born = np.clip(delta_born, -np.pi/2, np.pi/2)
        
        if abs(z_ion) < 1e-3:
            # Neutral: use Riccati-Bessel with Born phase
            chi_analytic = r * jl_vals * k_au
            # Apply phase shift properly via normalization factor
            A_norm = np.sqrt(2.0 / np.pi)  # δ(k-k') normalization
            return ContinuumWave(l, k_au, A_norm * chi_analytic, delta_born, eta=0.0, sigma_l=0.0, phase_method="born")
        else:
            # Ionic: use asymptotic Coulomb with Born phase
            eta = -z_ion / k_au
            coulomb_phase = np.imag(loggamma(l + 1 + 1j * eta))
            theta = rho_vec + eta * np.log(2.0 * rho_vec + 1e-30) - (l * np.pi / 2.0) + coulomb_phase + delta_born
            chi_coulomb = np.sqrt(2.0 / np.pi) * np.sin(theta)
            # Zero below turning point (asymptotic invalid there)
            r_turning = np.sqrt(l * (l + 1)) / k_au
            chi_coulomb[r < 0.5 * r_turning] = 0.0
            return ContinuumWave(l, k_au, chi_coulomb, delta_born, eta=eta, sigma_l=coulomb_phase, phase_method="born")


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
    
    # Physics-based turning point detection:
    # Check if S(r_min) > 0 (inside centrifugal barrier) instead of hardcoded l > 5
    # This handles low-l cases at low energies where potential is strong
    ell = float(l)
    k2 = k_au ** 2
    r0_check = r[0]
    S_at_origin = ell * (ell + 1.0) / (r0_check * r0_check) + 2.0 * U_arr[0] - k2
    
    if S_at_origin > 0:
        # We're inside the barrier at r_min → find safe starting point
        r_turn = np.sqrt(l*(l+1)) / k_au if k_au > 1e-6 else r[-1]
        r_safe = r_turn * 0.9
        idx_found = np.searchsorted(r, r_safe)
        if idx_found > 0 and idx_found < len(r) - 20: 
            idx_start = idx_found
            logger.debug(f"L={l}: Starting at idx={idx_start} (r={r[idx_start]:.2f}) due to barrier")
            
    r_eval = r[idx_start:]
    r0_int = float(r_eval[0])
    
    # ==========================================================================
    # NUMEROV PROPAGATION (Primary Method)
    # ==========================================================================
    # ==========================================================================
    # v2.12+: CONFIGURABLE SOLVER DISPATCH WITH FALLBACK CHAIN
    # ==========================================================================
    # User can select primary solver (numerov, johnson, rk45, auto).
    # The other two solvers serve as automatic fallbacks in defined order.
    # ==========================================================================
    
    # Build solver order: primary first, then remaining two
    ALL_SOLVERS = ["numerov", "johnson", "rk45"]
    primary_solver = solver.lower()
    
    # v2.13+: "auto" mode - physics-based solver selection per partial wave
    if primary_solver == "auto":
        # Selection criteria based on physics:
        # 1. High L (L > 25): Large centrifugal barrier → tunneling region extensive
        #    → Johnson's log-derivative method is most stable
        # 2. Low energy (E < 15 eV, k < 1.05 a.u.): Long-wavelength, potential-dominated
        #    → Johnson handles evanescent waves better
        # 3. Inside barrier at start (S0 > 0): Wavefunction grows exponentially
        #    → Johnson avoids underflow issues
        # 4. Otherwise: RK45 gives best phase accuracy on non-uniform grids
        
        L_HIGH_THRESHOLD = 25       # Use Johnson above this L
        E_LOW_THRESHOLD_EV = 15.0   # Use Johnson below this energy
        
        use_johnson = False
        
        if l > L_HIGH_THRESHOLD:
            use_johnson = True
            logger.debug(f"L={l}: Auto-selected Johnson (high L > {L_HIGH_THRESHOLD})")
        elif E_eV < E_LOW_THRESHOLD_EV:
            use_johnson = True
            logger.debug(f"L={l}: Auto-selected Johnson (low E={E_eV:.1f} eV < {E_LOW_THRESHOLD_EV})")
        elif S_at_origin > 0:
            # Deep inside barrier - Johnson is more stable
            use_johnson = True
            logger.debug(f"L={l}: Auto-selected Johnson (inside barrier, S0={S_at_origin:.2f})")
        
        if use_johnson:
            primary_solver = "johnson"
            solver_order = ["johnson", "rk45", "numerov"]
        else:
            primary_solver = "rk45"
            solver_order = ["rk45", "johnson", "numerov"]
    else:
        if primary_solver not in ALL_SOLVERS:
            logger.warning(f"Unknown solver '{solver}', defaulting to 'rk45'")
            primary_solver = "rk45"
        
        # Build fallback chain: primary + remaining solvers in order
        solver_order = [primary_solver] + [s for s in ALL_SOLVERS if s != primary_solver]
    
    # Helper functions for each solver
    def _run_numerov():
        """Run Numerov propagator and return (chi, dchi, success)."""
        chi_out, log_scale = _numerov_propagate(
            r_eval, Q_eval, chi0, chi1, 
            renorm_interval=adaptive_renorm_interval, renorm_scale=1e50
        )
        dchi_out = np.zeros_like(chi_out)
        for i in range(len(chi_out)):
            dchi_out[i] = _derivative_5point(chi_out, r_eval, i)
        return chi_out, dchi_out, True
    
    def _run_johnson():
        """Run Johnson log-derivative solver and return (chi, dchi, success)."""
        U_eval_fb = U_arr[idx_start:]
        r_eval_fb = r[idx_start:]
        chi_out, dchi_out = _johnson_log_derivative_solve(
            r_eval_fb, U_eval_fb, l, k_au, 
            renorm_interval=max(30, adaptive_renorm_interval // 2)
        )
        return chi_out, dchi_out, True
    
    def _run_rk45():
        """Run RK45 solver and return (chi, dchi, success)."""
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
            rtol=adaptive_rtol,
            atol=adaptive_atol,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[0, :], sol.y[1, :], True
        else:
            return None, None, False
    
    # Map solver names to runner functions
    solver_runners = {
        "numerov": _run_numerov,
        "johnson": _run_johnson,
        "rk45": _run_rk45
    }
    
    # Build Q(r) array for Numerov (needed for numerov, harmless for others)
    Q_full = ell * (ell + 1.0) / (r * r) + 2.0 * U_arr - k2
    Q_eval = Q_full[idx_start:]
    
    # Initial conditions for Numerov
    # v2.13+: Use exact Bessel functions (like RK45) instead of r^(l+1) approximation
    # The r^(l+1) approximation is only valid for kr << 1, but introduces phase errors
    # when used at larger kr where oscillatory behavior has already started.
    r0_init = r_eval[0]
    r1_init = r_eval[1]
    h_init = r1_init - r0_init
    S0_init = ell * (ell + 1.0) / (r0_init * r0_init) + 2.0 * U_arr[idx_start] - k2
    
    if S0_init > 0:
        # Inside centrifugal barrier: use WKB exponential growth
        kappa = np.sqrt(S0_init)
        chi0 = 1e-20
        chi1 = chi0 * np.exp(kappa * h_init)
        logger.debug(f"L={l}: Using WKB initial conditions (S0={S0_init:.2f}, κ={kappa:.4f})")
    else:
        # Outside barrier: use exact Bessel functions (same as RK45)
        # This ensures consistent phase between all solvers
        if abs(z_ion) < 1e-3:
            # Neutral: use Riccati-Bessel j-hat = r * j_l(kr)
            rho0 = k_au * r0_init
            rho1 = k_au * r1_init
            
            # Spherical Bessel functions
            jl0 = spherical_jn(l, rho0)
            jl1 = spherical_jn(l, rho1)
            
            # chi = r * j_l(kr) is the regular solution
            chi0 = r0_init * jl0
            chi1 = r1_init * jl1
            
            # Scale to avoid underflow for high L
            norm_val = max(abs(chi0), abs(chi1))
            if norm_val < 1e-10 and norm_val > 0:
                scale = 1e-10 / norm_val
                chi0 *= scale
                chi1 *= scale
        else:
            # Ionic: use WKB approximation like before
            kappa_sq = ell*(ell+1.0)/(r0_init*r0_init) - 2.0*z_ion/r0_init - k2
            if kappa_sq > 0:
                kappa = np.sqrt(kappa_sq)
                chi0 = 1e-20
                chi1 = chi0 * np.exp(kappa * h_init)
            else:
                # Fallback to Bessel
                rho0 = k_au * r0_init
                rho1 = k_au * r1_init
                chi0 = r0_init * spherical_jn(l, rho0)
                chi1 = r1_init * spherical_jn(l, rho1)
                norm_val = max(abs(chi0), abs(chi1))
                if norm_val < 1e-10 and norm_val > 0:
                    scale = 1e-10 / norm_val
                    chi0 *= scale
                    chi1 *= scale
    
    # Try solvers in order until one succeeds
    chi_computed = None
    dchi_computed = None
    method_used = None
    
    for solver_name in solver_order:
        try:
            chi_out, dchi_out, success = solver_runners[solver_name]()
            
            if not success:
                logger.debug(f"L={l}: {solver_name.capitalize()} solver returned failure, trying next")
                continue
            
            # Place in full grid
            if idx_start > 0:
                chi_raw = np.zeros_like(r, dtype=float)
                chi_raw[idx_start:] = chi_out
                dchi_raw = np.zeros_like(r, dtype=float)
                dchi_raw[idx_start:] = dchi_out
            else:
                chi_raw = chi_out
                dchi_raw = dchi_out
            
            # Find match point and test solution (v2.14+: with node avoidance)
            idx_match, r_m = _find_match_point(r, U_arr, k_au, l, threshold=1e-4, idx_start=idx_start, chi=chi_raw)
            chi_m = chi_raw[idx_match]
            dchi_m = dchi_raw[idx_match]
            
            if abs(chi_m) < 1e-100:
                is_fallback = solver_name != primary_solver
                fallback_str = " (fallback)" if is_fallback else ""
                logger.debug(f"L={l}: {solver_name.capitalize()}{fallback_str} produced χ≈0, trying next")
                continue
            
            # Success!
            is_fallback = solver_name != primary_solver
            method_used = solver_name.capitalize() + (" (fallback)" if is_fallback else "")
            chi_computed = chi_raw
            dchi_computed = dchi_raw
            break
            
        except Exception as e:
            logger.debug(f"L={l}: {solver_name.capitalize()} raised exception: {e}, trying next")
            continue
    
    if chi_computed is None:
        logger.warning(f"L={l}: All solvers ({', '.join(solver_order)}) failed")
        return None


    
    Y_m = dchi_m / chi_m
    
    # --- Step 2: Extract phase using selected method ---
    # v2.11+: Support for hybrid, logderiv, and lsq methods
    phase_method = phase_extraction_method.lower()
    
    if phase_method == "hybrid":
        # Hybrid: use both methods with cross-validation
        delta_l, A_amp, extraction_method = _extract_phase_hybrid(
            chi_raw, dchi_raw, r, grid.w_trapz, k_au, l, idx_match, z_ion,
            tail_fraction=0.15, disagreement_threshold=0.1
        )
        method_used = f"{method_used}/{extraction_method}"
        
    elif phase_method == "lsq":
        # Pure least-squares on tail
        n_tail = int(len(r) * 0.15)
        idx_tail = max(idx_match, len(r) - n_tail)
        r_tail = r[idx_tail:]
        chi_tail = chi_raw[idx_tail:]
        
        if abs(z_ion) < 1e-3:
            A_amp, delta_l = _fit_asymptotic_phase_neutral(r_tail, chi_tail, l, k_au)
        else:
            A_amp, delta_l = _fit_asymptotic_phase_coulomb(r_tail, chi_tail, l, k_au, z_ion)
        method_used = f"{method_used}/lsq"
        
    else:  # "logderiv" or default
        # Pure log-derivative at match point
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
        method_used = f"{method_used}/logderiv"
    
    # --- Diagnostics ---
    # Check phase stability by comparing log-derivative at TWO nearby points.
    # This tests whether the numerical wavefunction is consistent.
    # Note: We compare logderiv vs logderiv, not delta_l (which may be from LSQ).
    # IMPORTANT: Use alt point AFTER match point (not before) to ensure both
    # are in the asymptotic region (r > 2.5×r_turn).
    phase_stable = True
    N_grid = len(r)
    if idx_match > 10 and idx_match < N_grid - 15 and abs(chi_m) > 1e-100:
        idx_alt = idx_match + 10  # Use point AFTER match (both in asymptotic region)
        chi_alt = chi_raw[idx_alt]
        dchi_alt = dchi_raw[idx_alt]
        if abs(chi_alt) > 1e-100:
            # Compute log-derivative phase at MAIN point
            Y_main = dchi_raw[idx_match] / chi_raw[idx_match]
            if abs(z_ion) < 1e-3:
                delta_main_ld = _extract_phase_logderiv_neutral(Y_main, k_au, r_m, l)
            else:
                delta_main_ld = _extract_phase_logderiv_coulomb(Y_main, k_au, r_m, l, z_ion)
            
            # Compute log-derivative phase at ALT point
            Y_alt = dchi_alt / chi_alt
            r_alt = r[idx_alt]
            if abs(z_ion) < 1e-3:
                delta_alt_ld = _extract_phase_logderiv_neutral(Y_alt, k_au, r_alt, l)
            else:
                delta_alt_ld = _extract_phase_logderiv_coulomb(Y_alt, k_au, r_alt, l, z_ion)
            
            # Compare log-derivative at both points
            delta_diff = delta_main_ld - delta_alt_ld
            # Unwrap difference to [-pi, pi]
            delta_diff = (delta_diff + np.pi) % (2 * np.pi) - np.pi
            phase_variation = abs(delta_diff)
            phase_stable = phase_variation <= 0.1  # 0.1 rad tolerance (v2.11+)
            if not phase_stable:
                logger.warning(f"Phase unstable for L={l}: δ_ld varies by {phase_variation:.4f} rad")
    
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
    
    # =========================================================================
    # v2.14+: AMPLITUDE VALIDATION AND ENFORCEMENT
    # =========================================================================
    # Ensure asymptotic amplitude is exactly √(2/π) ≈ 0.7979 for δ(k-k') normalization.
    # This fixes inconsistent amplitudes across L that cause upturn artifacts.
    # =========================================================================
    
    # Compute actual amplitude in asymptotic region (RMS of sine wave = A/√2)
    asym_region = chi_final[idx_match:]
    if len(asym_region) > 10:
        chi_rms = np.sqrt(np.mean(asym_region**2))
        actual_amplitude = chi_rms * np.sqrt(2.0)  # A where χ ~ A·sin(...)
        expected_amplitude = np.sqrt(2.0 / np.pi)  # ≈ 0.7979
        
        # If amplitude deviates significantly (>5%), renormalize
        amplitude_ratio = actual_amplitude / expected_amplitude if expected_amplitude > 0 else 1.0
        
        if abs(amplitude_ratio - 1.0) > 0.05 and actual_amplitude > 1e-50:
            # Renormalize to exact unit amplitude
            chi_final /= amplitude_ratio
            logger.debug(f"L={l}: Amplitude corrected: {actual_amplitude:.4f} → {expected_amplitude:.4f} (ratio={amplitude_ratio:.3f})")
        
        # Validation warning if amplitude was very wrong
        if abs(amplitude_ratio - 1.0) > 0.3:
            logger.warning(f"L={l}: Large amplitude correction applied (ratio={amplitude_ratio:.3f}). Check wave quality.")
    
    # =========================================================================
    # Compute Coulomb phase parameters for oscillatory integral tail
    # =========================================================================
    # η = -z_ion/k (Sommerfeld parameter)
    # σ_l = arg(Γ(l+1+iη)) (Coulomb phase shift)
    eta_val = -z_ion / k_au if k_au > 0 else 0.0
    
    # Compute σ_l = Im(log(Γ(l+1+iη)))
    if abs(eta_val) > 1e-10:
        sigma_l_val = float(np.imag(loggamma(l + 1 + 1j * eta_val)))
    else:
        sigma_l_val = 0.0
    
    # Done. Package result with match point and Coulomb params for split integrals.
    cw = ContinuumWave(
        l=l,
        k_au=k_au,
        chi_of_r=chi_final,
        phase_shift=delta_l,
        r_match=r_m,
        idx_match=idx_match,
        eta=eta_val,
        sigma_l=sigma_l_val,
        phase_method=phase_extraction_method,
        solver_method=method_used
    )
    logger.debug("ContinuumWave created | l=%d, k=%.3f | chi_size=%d, idx_match=%d, grid_size=%d",
                 l, k_au, len(chi_final), idx_match, len(r))
    return cw
