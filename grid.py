# grid.py
#
# Radial grid definition, unit conversions, and basic quadrature weights.
# This module sets the numerical "geometry" for the entire DWBA pipeline.
#
# Conventions:
# - All radial distances r are in atomic units (bohr).
# - All energies are in atomic units (hartree) unless explicitly stated.
# - All wave numbers k are in atomic units (bohr^{-1}).
#
# Physics background:
# In atomic units (Hartree atomic system), we have:
#   ħ = 1
#   m_e = 1
#   e^2 = 1
# so that the Schrödinger radial equation in the article has the form
#   [-1/2 d^2/dr^2 + l(l+1)/(2 r^2) + V(r)] u(r) = E u(r)
# and the free-electron kinetic energy is E = k^2 / 2.
#
# We'll keep all subsequent solvers in these units so that we reproduce
# exactly the equations as written in the paper.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ---- Physical constants ----

# 1 hartree in eV
HARTREE_TO_EV = 27.211386245988  # CODATA-like; high precision
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


@dataclass(frozen=True)
class RadialGrid:
    """
    Radial grid container.

    Attributes
    ----------
    r : np.ndarray
        Radial coordinate values in atomic units (bohr), shape (N,).
        Monotonically increasing, strictly positive.
    dr : np.ndarray
        Local radial spacing dr[i] = r[i+1] - r[i], shape (N,).
        Last element dr[-1] is set equal to dr[-2] for convenience,
        so dr has same length as r. This is useful for quick finite
        differences / integrations where we just need a local scale.
    w_trapz : np.ndarray
        Trapezoidal-rule weights for integrating a function f(r).
    w_simpson : np.ndarray
        Simpson's rule weights (O(h^4)) for higher accuracy integration.
        Requires N to be odd; if N is even, last interval uses trapezoidal correction.
    """
    r: np.ndarray
    dr: np.ndarray
    w_trapz: np.ndarray
    w_simpson: np.ndarray


def ev_to_au(E_eV: float | np.ndarray) -> float | np.ndarray:
    """
    Convert energy from electronvolts [eV] to Hartree atomic units [a.u.].

    E_au = E_eV / 27.211386...

    Parameters
    ----------
    E_eV : float or np.ndarray
        Energy in eV.

    Returns
    -------
    float or np.ndarray
        Energy in Hartree (a.u.).
    """
    return np.asarray(E_eV) * EV_TO_HARTREE


def au_to_ev(E_au: float | np.ndarray) -> float | np.ndarray:
    """
    Convert energy from Hartree atomic units [a.u.] to electronvolts [eV].

    E_eV = E_au * 27.211386...

    Parameters
    ----------
    E_au : float or np.ndarray
        Energy in Hartree.

    Returns
    -------
    float or np.ndarray
        Energy in eV.
    """
    return np.asarray(E_au) * HARTREE_TO_EV


def k_from_E_eV(E_eV: float | np.ndarray) -> float | np.ndarray:
    """
    Convert electron kinetic energy (lab electron) in eV
    to wave number k [a.u.].

    In atomic units:
        E (a.u.) = k^2 / 2
    =>  k = sqrt(2 E_au)

    This is exactly how the projectile electron momentum is handled
    in the article (the incoming and outgoing distorted waves χ(k,r)
    are labeled by k, with energies related by E = k^2/2).

    Parameters
    ----------
    E_eV : float or np.ndarray
        Kinetic energy of the free/scattered electron in eV.
        Must be >= 0. If it's 0 or tiny, k -> 0.

    Returns
    -------
    float or np.ndarray
        Wave number k in atomic units (1/bohr).
    """
    E_au = ev_to_au(E_eV)
    # numerical safety: any slight negative due to rounding -> clip to 0
    E_au = np.maximum(E_au, 0.0)
    return np.sqrt(2.0 * E_au)


def _trapz_weights_nonuniform(r: np.ndarray) -> np.ndarray:
    """
    Build trapezoidal-rule weights for integrating ∫ f(r) dr
    on an arbitrary strictly increasing grid r[0] < r[1] < ... < r[N-1].

    For nonuniform spacing, the trapezoidal rule says:
        ∫ f(r) dr ≈ Σ_i w[i] f(r[i])
    with
        w[0]   = 0.5 * (r[1]-r[0])
        w[i]   = 0.5 * (r[i+1]-r[i-1])   for 0 < i < N-1
        w[N-1] = 0.5 * (r[N-1]-r[N-2])

    This gives us a reusable "weight vector" we can dot with f(r)
    to approximate 1D radial integrals. We'll use it for:
    - normalizations of bound states,
    - computing Hartree potentials from |Φ|^2,
    - radial integrals in DWBA kernels.

    Parameters
    ----------
    r : np.ndarray, shape (N,)
        Strictly increasing radial grid in a.u.

    Returns
    -------
    w : np.ndarray, shape (N,)
        Trapezoidal weights for that grid.
    """
    N = r.size
    w = np.zeros_like(r)
    if N < 2:
        raise ValueError("Need at least 2 grid points for trapezoidal weights.")
    if not np.all(np.diff(r) > 0):
        raise ValueError("r grid must be strictly increasing.")

    # end-points
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    # interior
    if N > 2:
        w[1:-1] = 0.5 * (r[2:] - r[:-2])
    return w

def _simpson_weights_nonuniform(r: np.ndarray) -> np.ndarray:
    """
    Build Simpson's rule weights for nonuniform grid.
    Approximate O(h^4).
    """
    N = r.size
    w = np.zeros(N)
    if N < 3:
        return _trapz_weights_nonuniform(r) # Fallback
        
    diffs = np.diff(r) # h_i = r[i+1] - r[i]
    
    # We iterate over pairs of intervals (triplets of points)
    # i=0,1,2 -> interval 0 and 1.
    # i=2,3,4 -> interval 2 and 3.
    # If N is even, we have N-1 intervals (odd count).
    # We can fit pairs up to N-2 (leaving 1 interval).
    
    # Logic: loop stride 2.
    for i in range(0, N-2, 2):
        h1 = diffs[i]
        h2 = diffs[i+1]
        
        # Coefficients for parabola over r[i], r[i+1], r[i+2]
        # derived from Lagrange polynomials integration
        alpha = (2*h1**3 - h2**3 + 3*h1*h2*(h1+h2)) / (6*h1*(h1+h2)) # Wait, simple?
        # Standard Simpson nonuniform:
        # Integral = (h1+h2)/6 * [ (2 - h2/h1) f0 + (h1+h2)^2/(h1h2) f1 + (2 - h1/h2) f2 ]
        
        c0 = (h1+h2)/6.0 * (2.0 - h2/h1)
        c1 = (h1+h2)/6.0 * ( (h1+h2)**2 / (h1*h2) )
        c2 = (h1+h2)/6.0 * (2.0 - h1/h2)
        
        w[i]   += c0
        w[i+1] += c1
        w[i+2] += c2
        
    # Handle remainder if N is even (N-1 intervals is odd number of intervals, so last one is left over)
    if N % 2 == 0:
        # Last interval (N-2 to N-1) using Trapezoidal (or 2nd order) correction
        h_last = diffs[-1]
        w[-2] += 0.5 * h_last
        w[-1] += 0.5 * h_last
        
    return w



def make_r_grid(
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 4000,
    kind: str = "exp"
) -> RadialGrid:
    """
    Construct the global radial grid to be used everywhere in the DWBA code.

    Requirements from the physics / the article:
    -------------------------------------------
    - We need high resolution near the origin because:
        * V_A+(r) ~ -Z_eff/r is singular-ish near r=0.
        * the centrifugal barrier l(l+1)/(2r^2) is huge at small r.
      The bound-state wavefunctions and distorted waves vary fastest there.

    - We need to extend to large r (hundreds of bohr) because:
        * bound orbitals of weakly bound states (Rydberg-like) extend far,
        * distorted continuum waves χ_l(k,r) must be propagated out
          to where the potential is negligible so we can match
          to asymptotic scattering behavior.

    - We need a monotone increasing r-grid usable for:
        * solving radial Schrödinger equations (Numerov-like methods),
        * building Hartree-type potentials via radial integrals,
        * computing DWBA radial matrix elements (double integrals in r1,r2).

    Design:
    -------
    We'll build a quasi-exponential / stretched grid:
        r[i] = r_min * (r_max/r_min)^(i/(N-1))
    which is geometric (log-spaced). This packs many points near r_min
    and still reaches r_max smoothly. It behaves well for Coulomb-like
    and atomic-type problems.

    Alternatives (linear piecewise, etc.) are possible, but a pure
    geometric grid is a good starting point for atomic scattering:
    - near r=0: fine spacing ~ r_min * (ratio)^(small)
    - at large r: spacing grows ~ r itself, which is acceptable
      because χ and V(r) vary slowly there.

    We expose parameters so you can refine later if needed.

    Parameters
    ----------
    r_min : float
        Smallest radius in a.u. Must be > 0.
        Values ~1e-5 a.u. avoid division by zero in centrifugal term.
    r_max : float
        Largest radius in a.u. Should be big enough that
        U_j(r_max) ~ 0 and χ is essentially asymptotic.
        200 a.u. is conservative for light ions up to keV energies.
    n_points : int
        Number of radial points. Must be >= 2.
        ~2000-4000 is reasonable for stable Numerov + smooth integrals.
        We default to 4000 because later we'll do double integrals
        over r1,r2 and we want reasonable resolution.
    kind : str
        Currently only "exp" (pure exponential / geometric spacing)
        is implemented. Hook left here in case we add hybrid grids.

    Returns
    -------
    RadialGrid
        Dataclass with:
        - r        : (N,) radii in bohr,
        - dr       : (N,) local spacing,
        - w_trapz  : (N,) trapezoid weights for ∫dr.

    Raises
    ------
    ValueError
        If parameters are not physically meaningful.
    """
    if r_min <= 0.0:
        raise ValueError("r_min must be > 0 to avoid singularities at r=0.")
    if r_max <= r_min:
        raise ValueError("r_max must be > r_min.")
    if n_points < 2:
        raise ValueError("Need at least 2 grid points.")
    if kind != "exp":
        raise ValueError(f"Unknown grid kind '{kind}'. Only 'exp' supported now.")

    # geometric / exponential spacing in r
    # r[i] = r_min * (r_max/r_min)^(i/(N-1))
    idx = np.linspace(0.0, 1.0, n_points)
    ratio = r_max / r_min
    r = r_min * (ratio ** idx)

    # basic sanity
    if not np.all(np.diff(r) > 0):
        raise RuntimeError("Generated r grid is not strictly increasing. Check parameters.")

    # local spacing dr[i] ~ r[i+1]-r[i]; last element copied from previous
    dr = np.empty_like(r)
    dr[:-1] = r[1:] - r[:-1]
    dr[-1] = dr[-2]

    # trapezoidal weights for ∫ f(r) dr
    w_trapz = _trapz_weights_nonuniform(r)
    w_simpson = _simpson_weights_nonuniform(r)

    return RadialGrid(r=r, dr=dr, w_trapz=w_trapz, w_simpson=w_simpson)


# ---- Convenience helpers for later stages ----

def integrate_trapz(f: np.ndarray, grid: RadialGrid) -> float:
    """
    Approximate ∫ f(r) dr on the given radial grid using precomputed
    trapezoidal weights.

    This is used a LOT:
    - normalization of bound orbitals ∫ |u(r)|^2 dr = 1,
      because u(r) is the radial function in the standard reduced form.
    - computing expectation values.
    - building Hartree potentials from |Φ|^2.

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Function sampled on grid.r
    grid : RadialGrid
        The radial grid (must match f.shape).

    Returns
    -------
    float
        Approximation to ∫ f(r) dr.
    """
    if f.shape != grid.r.shape:
        raise ValueError("integrate_trapz: f and grid.r must have same shape.")
    return float(np.sum(grid.w_trapz * f))
