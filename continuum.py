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
from scipy.special import spherical_jn, loggamma


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
        δ_l in radians, extracted from the asymptotic fit.
        Physically this encodes the short-range distortion due to U_j(r).
    """
    l: int
    k_au: float
    chi_of_r: np.ndarray
    phase_shift: float

    @property
    def u_of_r(self) -> np.ndarray:
        """Alias for chi_of_r to ensure compatibility with BoundOrbital interface."""
        return self.chi_of_r


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

    # --- Analytic Bypass for High L (Neutral) ---
    # For very high L (e.g. > 60), the centrifugal barrier L(L+1)/r^2 dominates the potential U(r).
    # The solution is effectively a free particle wave (Bessel function).
    # Numerical integration is unstable due to underflow/noise domination.
    # We return the exact analytic solution j_l(kr).
    if l > 60 and abs(z_ion) < 1e-3:
        # u(r) = r * j_l(k*r)
        rho_vec = k_au * r
        # spherical_jn(l, z) is efficient for vector z in scipy
        jl_vals = spherical_jn(l, rho_vec)
        # Normalize to Unit Amplitude: u ~ (1/k) sin. So multiply by k.
        chi_analytic = r * jl_vals * k_au
        
        # Phase shift is 0 relative to plane wave

        return ContinuumWave(l, k_au, chi_analytic, 0.0)

    # --- Optimization: For high l, skip deep tunneling region ---

    # The centrifugal barrier l(l+1)/r^2 is huge at small r.
    # The solution is effectively zero until we approach the classical turning point r_c ~ sqrt(l(l+1))/k.
    # Integrating from r_min (e.g. 0.05) when r_c is 12 (for l=100) causes extreme stiffness and failure.
    
    idx_start = 0
    if l > 5:
        r_turn = np.sqrt(l*(l+1)) / k_au
        # Start at a fraction of turning point (e.g. 0.3) where wavefunction is still tiny but ODE is stable
        r_safe = r_turn * 0.9
        
        # Find index in grid
        idx_found = np.searchsorted(r, r_safe)
        if idx_found > 0 and idx_found < len(r) - 20: 
            idx_start = idx_found
            
    # Define integration sub-grid
    r_eval = r[idx_start:]
    
    # initial conditions at the first grid point of evaluation
    r0_int = float(r_eval[0])
    
    if idx_start > 0:
        # Verify 90% of turning point - might imply we are in a region where U is small?
        # At turning point L(L+1)/r^2 ~ k^2. 
        # Potential V(r) is usually small compared to centrifugal there.
        # So Bessel/Coulomb initialization is physically justified.
        y0 = _initial_conditions_high_L(r0_int, l, k_au, z_ion)
    else:
        y0 = _initial_conditions_regular(r0_int, l)


    # define RHS of ODE system
    rhs = _schrodinger_rhs_factory(l=l, U_spline=U_spline, k_au=k_au)

    # integrate outward up to r_max
    # r_max = float(r[-1]) # Not needed explicitly if using t_eval

    # Determine max_step to avoid aliasing oscillations (lambda = 2pi/k)
    # We want at least 10-20 steps per wavelength.
    # lambda ~ 2*pi / k_au.
    if k_au > 1e-3:
        wavelength = 2.0 * np.pi / k_au
        max_step_val = wavelength / 20.0
    else:
        max_step_val = 0.1 # default safe step
        
    # Cap max_step to avoid being too small or too large
    # For high L, we might need smaller steps initially, but RK45 adapts. 
    # Just ensure we don't miss oscillations.
    max_step_val = min(max_step_val, 0.2) 

    sol = solve_ivp(
        fun=rhs,
        t_span=(r0_int, float(r_eval[-1])),
        y0=y0,
        t_eval=r_eval,          
        method="RK45",
        max_step=max_step_val,
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    if not sol.success:
        raise RuntimeError(f"solve_continuum_wave: ODE solver failed: {sol.message}")

    # Extract χ(r) from solution and pad with zeros if needed
    chi_computed = sol.y[0, :]  # shape (N_eval,)
    
    if idx_start > 0:
        chi_raw = np.zeros_like(r, dtype=float)
        chi_raw[idx_start:] = chi_computed
    else:
        chi_raw = chi_computed

    # Sanity: remove any global sign if necessary (not physically important).
    # We won't flip sign here because the asymptotic fit will absorb it into δ_l anyway.

    # --- Fit asymptotic tail to get amplitude and phase shift ---

    N = r.size
    n_tail = max(20, int(np.floor(tail_fraction * N)))
    if n_tail >= N // 2:
        # tail can't be the entire grid; enforce upper bound
        n_tail = max(20, N // 4)

    tail_slice = slice(N - n_tail, N)
    r_tail = r[tail_slice]
    chi_tail = chi_raw[tail_slice]

    # Decide matching strategy
    if abs(z_ion) < 1e-3:
        A_amp, delta_l = _fit_asymptotic_phase_neutral(r_tail, chi_tail, l, k_au)
    else:
        A_amp, delta_l = _fit_asymptotic_phase_coulomb(r_tail, chi_tail, l, k_au, z_ion)

    if not np.isfinite(A_amp) or not np.isfinite(delta_l) or A_amp == 0.0:
        # Wave is unreliable (small amplitude or poor fit) - return None
        # Caller should handle this by skipping this partial wave
        return None

    # Renormalize χ so asymptotic amplitude is 1
    chi_norm = chi_raw / A_amp

    
    # Done. Package result.
    cw = ContinuumWave(

        l=l,
        k_au=k_au,
        chi_of_r=chi_norm,
        phase_shift=delta_l
    )
    return cw
