# distorting_potential.py
#
# Construction of the distorted-wave scattering potentials U_j(r)
# used in DWBA, following the article.
#
# The article defines (in atomic units):
#
#   U_j(r) = V_{A+}(r) + ∫ |Φ_j(r2)|^2 / |r - r2| d^3 r2,
#
# where Φ_j is the bound-state wavefunction of the active electron
# in the *initial* (j=i) or *final* (j=f) target state.
#
# Physically:
# - V_{A+}(r) is the effective single-active-electron core potential
#   (the ionic core without the active electron).
# - The integral term is the electrostatic (Hartree) potential generated
#   by the active electron itself. We take only the spherically averaged
#   (monopole) contribution, which is exactly what the paper does.
#
# For a spherically symmetric charge density ρ(r2), the Coulomb potential is:
#
#   V_H(r) = (1/r) ∫_0^r ρ(r2) 4π r2^2 dr2
#          +       ∫_r^∞ ρ(r2) 4π r2   dr2
#
# In our single-active-electron description,
#
#   Φ_j(r,Ω) = u_j(r)/r * Y_{l m}(Ω),
#
# where u_j(r) is the reduced radial wavefunction solved in bound_states.py,
# normalized so that ∫ |u_j(r)|^2 dr = 1.
#
# If we average |Φ_j|^2 over angles (which is what we want for a central,
# spherically averaged potential), the resulting radial probability density
# per unit r is:
#
#   P(r) = |u_j(r)|^2          [dimension: 1/length]
#
# and the corresponding *3D* density is
#
#   ρ(r) = |Φ_j(r)|^2 = |u_j(r)|^2 / (4π r^2).
#
# Plugging ρ into the Hartree formula simplifies beautifully to:
#
#   V_H(r)
#    = (1/r) ∫_0^r |u_j(r2)|^2 dr2
#      +      ∫_r^∞ |u_j(r2)|^2 / r2  dr2
#
# (atomic units, Coulomb interaction is +1/|r1 - r2|)
#
# This expression is what we will compute numerically on our radial grid.
#
# IMPORTANT:
#  - Because u_j(r) is normalized (∫ |u_j|^2 dr = 1), for large r:
#        V_H(r) -> 1/r.
#    Then:
#        U_j(r) = V_{A+}(r) + V_H(r) -> -1/r + 1/r = 0
#    So U_j(r) is short-range. This matches the article's requirement
#    that the distorted scattering wave χ_l(k,r) can asymptotically match
#    a free spherical wave, not a Coulomb wave.
#
#
# What this file provides:
#
# - cumulative_trapz_forward / cumulative_trapz_reverse:
#       robust cumulative integrators on our *nonuniform* radial grid.
#
# - hartree_potential_from_orbital:
#       computes V_H(r) from a BoundOrbital.
#
# - U_distorting:
#       builds U_j(r) = V_core(r) + V_H(r).
#
# - build_distorting_potentials:
#       convenience helper that, given two orbitals (initial and final),
#       returns U_i(r) and U_f(r) on the common grid.
#
#
# All inputs/outputs are in atomic units, consistent with the rest of the code.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from grid import RadialGrid
from bound_states import BoundOrbital
from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)


@dataclass(frozen=True)
class DistortingPotential:
    """
    Container for a distorted-wave potential U_j(r).

    Attributes
    ----------
    U_of_r : np.ndarray
        The full distorted potential U_j(r) = V_core(r) + V_H(r),
        in Hartree, sampled on the same radial grid.
        By construction U_of_r -> 0 for large r.
    V_hartree_of_r : np.ndarray
        The Hartree (electron-electron) part V_H(r), in Hartree.
        This should behave ~ +1/r at large r, because the active electron
        carries total charge -1 in atomic units.
    """
    U_of_r: np.ndarray
    V_hartree_of_r: np.ndarray


def _cumulative_trapz_forward(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute forward cumulative integral:
        F[i] = ∫_{r[0]}^{r[i]} f(r') dr'
    using trapezoidal rule on a nonuniform grid.

    Implementation:
        F[0] = 0
        F[i] = F[i-1] + 0.5*(f[i-1] + f[i]) * (r[i] - r[i-1])

    Parameters
    ----------
    r : np.ndarray, shape (N,)
        Strictly increasing radii (bohr).
    f : np.ndarray, shape (N,)
        Function values at those radii.

    Returns
    -------
    F : np.ndarray, shape (N,)
        Forward cumulative integral.
        F[0] = 0 by definition.
    """
    if r.shape != f.shape:
        raise ValueError("cumulative_trapz_forward: shape mismatch.")
    N = r.size
    F = np.zeros_like(r, dtype=float)
    for i in range(1, N):
        dr = r[i] - r[i - 1]
        F[i] = F[i - 1] + 0.5 * (f[i - 1] + f[i]) * dr
    return F


def _cumulative_trapz_reverse(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute reverse cumulative integral:
        G[i] = ∫_{r[i]}^{r[-1]} g(r') dr'
    using trapezoidal rule on a nonuniform grid.

    Implementation:
        G[-1] = 0
        G[i] = G[i+1] + 0.5*(g[i+1] + g[i]) * (r[i+1] - r[i])

    Parameters
    ----------
    r : np.ndarray, shape (N,)
        Strictly increasing radii (bohr).
    g : np.ndarray, shape (N,)
        Function values at those radii.

    Returns
    -------
    G : np.ndarray, shape (N,)
        Reverse cumulative integral.
        G[-1] = 0 by definition.

    Notes
    -----
    We approximate ∫_{r}^{∞} (...) dr' by ∫_{r}^{r_max} (...),
    assuming the orbital density vanishes by r_max. With a large enough
    r_max (e.g. 200 bohr+), that is perfectly fine numerically.
    """
    if r.shape != g.shape:
        raise ValueError("cumulative_trapz_reverse: shape mismatch.")
    N = r.size
    G = np.zeros_like(r, dtype=float)
    for i in range(N - 2, -1, -1):
        dr = r[i + 1] - r[i]
        G[i] = G[i + 1] + 0.5 * (g[i + 1] + g[i]) * dr
    return G


def hartree_potential_from_orbital(
    grid: RadialGrid,
    orbital: BoundOrbital
) -> np.ndarray:
    """
    Compute the Hartree potential V_H(r) generated by the active electron
    in the given bound orbital, following the spherically averaged
    single-active-electron picture in the article.

    Formula in atomic units:
        V_H(r) = (1/r) ∫_0^r |u(r')|^2 dr'  +  ∫_r^∞ |u(r')|^2 / r' dr'

    where u(r) is the reduced radial wavefunction from BoundOrbital,
    normalized so that ∫ |u(r)|^2 dr = 1.

    Interpretation:
    - The first term is like the Coulomb field from the "inner" charge
      enclosed within radius r.
    - The second term is the contribution from the "outer shell"
      beyond r.
    - Because ∫ |u|^2 dr = 1, we have V_H(r) → 1/r at large r.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid (bohr).
    orbital : BoundOrbital
        Bound state (u_of_r, energy_au, etc.) from bound_states.solve_bound_states.

    Returns
    -------
    V_H : np.ndarray, shape (N,)
        Hartree potential in Hartree, sampled at grid.r.

    Raises
    ------
    ValueError
        If orbital/u_of_r shape doesn't match grid.
    """
    r = grid.r
    u = orbital.u_of_r
    if u.shape != r.shape:
        raise ValueError("hartree_potential_from_orbital: mismatched grid/orbital shapes.")

    # Probability density in r-space for the reduced radial function u(r)
    # is |u(r)|^2 (dimension 1/length). We've normalized u so that
    # integral |u(r)|^2 dr = 1 using integrate_trapz in bound_states.
    u2 = np.abs(u) ** 2  # |u|^2

    # Precompute forward integral:
    #   Q_inner(r) = ∫_0^r |u(r')|^2 dr'
    Q_inner = _cumulative_trapz_forward(r, u2)

    # Precompute reverse integral:
    #   Q_outer(r) = ∫_r^∞ |u(r')|^2 / r' dr'
    # We'll approximate ∞ by r_max = r[-1].
    # We need g(r') = |u(r')|^2 / r'
    # Guard against r=0: grid.r[0] > 0 by construction, so safe.
    g = u2 / r
    Q_outer = _cumulative_trapz_reverse(r, g)

    # Now build V_H(r):
    #   V_H(r) = Q_inner(r)/r + Q_outer(r)
    # Guard against division by zero at r ~ 0 using same assumption:
    # our grid never includes r=0 exactly (r_min ~ 1e-5).
    V_H = Q_inner / r + Q_outer

    # sanity: V_H should be finite and positive, ~1/r at large r
    if not np.all(np.isfinite(V_H)):
        raise ValueError("Hartree potential contains non-finite values.")

    return V_H



def furness_mccarthy_exchange_potential(
    grid: RadialGrid,
    rho_spher: np.ndarray,
    E_beam_au: float,
    V_static: np.ndarray
) -> np.ndarray:
    """
    Compute the local equivalent exchange potential V_ex(r) using the
    Furness-McCarthy (FM) approximation.

    Ref: Furness, J. B., & McCarthy, I. E. (1973). J. Phys. B, 6, 2280.
    
    Formula:
        V_ex(r) = 0.5 * [ E - V_stat(r) - sqrt( (E - V_stat(r))^2 + 4π ρ(r) ) ]
    
    where:
    - E is the total energy of the scattering electron (E_beam_au).
    - V_stat(r) is the static potential seen by the electron (Core + Hartree).
      V_stat(r) = V_core(r) + V_H(r).
    - ρ(r) is the electron density of the bound state.
      NOTE: The density normally used is the *total* electron density of the target.
      In our SAE approx for H/He+, this is just the density of the 1 bound electron.
      ρ(r) = |Φ(r)|^2 = |u(r)/r|^2.
      
    This approximation yields a local potential that mimics the non-local exchange operator.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid.
    rho_spher : np.ndarray
        Spherical electron density rho(r) = |u(r)/r|^2.
        Units: 1/bohr^3.
    E_beam_au : float
        Energy of the scattering electron in Hartree.
        Can be negative (for bound states) or positive (continuum).
        Typically use k^2/2.
    V_static : np.ndarray
        The total static potential (V_core + V_Hartree) in Hartree.

    Returns
    -------
    V_ex : np.ndarray
        Exchange potential in Hartree.
    """
    # Helper for numerical safety inside sqrt
    # square_term = (E - V_stat)^2 + 4pi * rho
    
    diff = E_beam_au - V_static
    term_sq = diff**2 + 4.0 * np.pi * rho_spher
    
    # V_ex = 0.5 * (diff - sqrt(term_sq))
    V_ex = 0.5 * (diff - np.sqrt(term_sq))
    
    return V_ex

def polarization_potential(
    grid: RadialGrid,
    alpha_d: float,
    rc: float
) -> np.ndarray:
    """
    Compute the polarization potential V_pol(r) using the Buckingham form.
    
    Formula:
        V_pol(r) = - (alpha_d / (2 * r^4)) * [ 1 - exp( - (r/rc)^6 ) ]
        
    Parameters
    ----------
    grid : RadialGrid
        Radial grid.
    alpha_d : float
        Static dipole polarizability (a.u.).
    rc : float
        Cutoff radius (a.u.).
        
    Returns
    -------
    V_pol : np.ndarray
        Polarization potential in Hartree.
    """
    r = grid.r
    r_saf = np.maximum(r, 1e-12)
    
    # Buckingham cutoff factor
    C_r = 1.0 - np.exp( - (r_saf / rc)**6 )
    
    V_pol = - (alpha_d / (2.0 * r_saf**4)) * C_r
    
    return V_pol



def U_distorting(
    V_core_array: np.ndarray,
    V_H_array: np.ndarray,
    orbital: Optional[BoundOrbital] = None,
    grid: Optional[RadialGrid] = None,
    E_beam_au: Optional[float] = None,
    exchange_type: str = 'fumc'
) -> np.ndarray:
    """
    Build the distorted potential U_j(r) for a given channel j.
    
    Supports local exchange model:
    - 'fumc': Furness-McCarthy

        U_j(r) = V_{A+}(r) + V_H^{(j)}(r) + V_ex^{(j)}(r)

    with:
    - V_{A+}(r) from the SAE core potential (potential_core.V_core_on_grid),
    - V_H^{(j)}(r) from hartree_potential_from_orbital.

    Physical meaning:
    - U_i(r) is what the *incident* electron "sees" in the entrance channel,
      i.e. target in its initial state Φ_i.
    - U_f(r) is what the *scattered* electron "sees" in the exit channel,
      i.e. target in the final state Φ_f.

    Important property:
    - At large r, V_{A+}(r) ~ -1/r and V_H(r) ~ +1/r,
      so U_j(r) -> 0. This is exactly the condition in the article
      that lets us treat the asymptotic scattered waves χ_l(k,r)
      as essentially free spherical waves with short-range phase shifts,
      not long-range Coulomb waves.

    Parameters
    ----------
    V_core_array : np.ndarray
        Core potential V_{A+}(r) in Hartree.
    V_H_array : np.ndarray, shape (N,)
        Hartree potential V_H(r).
    orbital : BoundOrbital, optional
        The bound orbital active in this channel. Required for Exchange.
    grid : RadialGrid, optional
        Required for Exchange calculation (to convert u->rho).
    E_beam_au : float, optional
        Scattering energy E = k^2/2. Required for Furness-McCarthy.
    exchange_type : str
        'fumc' (Furness-McCarthy, default).

    Returns
    -------
    U_array : np.ndarray, shape (N,)
        Distorting potential in Hartree.

    Raises
    ------
    ValueError
        If shapes mismatch or U has non-finite values.
    """
    if V_core_array.shape != V_H_array.shape:
        raise ValueError("U_distorting: shape mismatch.")
    
    V_static = V_core_array + V_H_array
    
    if orbital is not None and grid is not None:
        # --- Compute Exchange ---
        # 1. Density rho(r) = |u(r)/r|^2
        r = grid.r
        # Avoid division by zero at r=0 logic (our grid starts > 0 usually)
        u = orbital.u_of_r
        # Safety for tiny r
        r_saf = np.maximum(r, 1e-12)
        rho = (np.abs(u) / r_saf)**2
        
        V_ex = np.zeros_like(r)
        
        if exchange_type == 'fumc' and E_beam_au is not None:
            # Furness-McCarthy needs Energy and Static Potential
            V_ex = furness_mccarthy_exchange_potential(grid, rho, E_beam_au, V_static)
        
        U = V_static + V_ex
    else:
        # Static only
        U = V_static

    if not np.all(np.isfinite(U)):
        # Provide fallback or error?
        # Sometimes FM can develop issues if E is very negative?
        # But here usually it's fine.
        raise ValueError("U_distorting: produced non-finite values.")
        
    return U


def build_distorting_potentials(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    orbital_initial: BoundOrbital,
    orbital_final: BoundOrbital,
    k_i_au: float = 0.5,
    k_f_au: float = 0.5,
    use_exchange: bool = False,  # Deprecated - always False per article
    use_polarization: bool = False
) -> Tuple[DistortingPotential, DistortingPotential]:
    """
    Construct U_i(r) and U_f(r) following Article Eq. 456-463.
    
    Uses STATIC potentials: U_j = V_core + V_Hartree
    
    Exchange is NOT included in potentials - it's treated perturbatively 
    in the T-matrix via amplitude g (Article standard approach).
      
    If use_polarization=True, adds polarization potential.
    
    Note: use_exchange parameter is deprecated and ignored. Exchange in
    potentials was removed as it caused double-counting with T-matrix exchange.
    
    Convenience helper:
    Given the core potential V_{A+}(r) and two bound orbitals
    (initial Φ_i and final Φ_f), construct BOTH distorted-wave
    channel potentials:
        U_i(r)  for the entrance channel,
        U_f(r)  for the exit channel.

    Steps:
    1. Compute V_H from orbital_initial  -> V_H_i(r).
    2. Compute V_H from orbital_final    -> V_H_f(r).
    3. Build U_i = V_core + V_H_i
       and U_f = V_core + V_H_f.

    This matches the article's definition where the incoming electron
    sees U_i and the outgoing electron sees U_f.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid.
    V_core_array : np.ndarray, shape (N,)
        Core potential V_{A+}(r) [Hartree] on that same grid.
    orbital_initial : BoundOrbital
        Bound orbital representing the active electron in the *initial*
        target state Φ_i.
    orbital_final : BoundOrbital
        Bound orbital representing the active electron in the *final*
        target state Φ_f.

    Returns
    -------
    (U_i, U_f) : tuple(DistortingPotential, DistortingPotential)
        U_i, U_f are dataclasses containing both U_of_r and the
        Hartree part V_hartree_of_r for diagnostic / reuse.

    Raises
    ------
    ValueError
        If shapes mismatch or inputs inconsistent.

    Notes
    -----
    - The only assumption here is that orbital_initial.u_of_r and
      orbital_final.u_of_r are defined on the SAME grid as V_core_array.
      This is guaranteed if everything was generated consistently
      using the same RadialGrid.
    - Physically, in DWBA you solve distorted-wave scattering states
      χ_l(k,r) in U_i for the incoming channel, and in U_f for the
      outgoing channel. We'll do to to build χ in the next module.
    """

    r = grid.r
    if V_core_array.shape != r.shape:
        raise ValueError("build_distorting_potentials: V_core_array/grid mismatch.")

    # energies
    E_i = 0.5 * k_i_au**2
    E_f = 0.5 * k_f_au**2

    # Hartree potentials
    V_H_i = hartree_potential_from_orbital(grid, orbital_initial)
    V_H_f = hartree_potential_from_orbital(grid, orbital_final)

    # Distorting potentials - ALWAYS use Static (Article Eq. 456-463)
    # U_j = V_core + V_H_j
    # Note: use_exchange parameter is deprecated and ignored
    U_i_arr = U_distorting(V_core_array, V_H_i)
    U_f_arr = U_distorting(V_core_array, V_H_f)
        
    if use_polarization:
        # Add Polarization Potential V_pol
        # Need to estimate alpha_d and rc for initial and final states.
        # Heuristic:
        # alpha_d approx (9/2) * (n^6 / Z^4) ?? Wait, simplistic.
        # Ground state H (n=1, Z=1): 4.5.
        # Ground state He+ (n=1, Z=2): 4.5/16 = 0.28.
        # n-scaling is strong (n^6 or n^4).
        # rc approx 1.5 * <r>.
        # <r> = (3n^2 - l(l+1))/(2Z).
        
        # Helper to estimate
        def estimate_pol_params(orb: BoundOrbital, Z_eff: float):
            # We don't have n,l inside orbital directly? We do: orb.n_index is n-l.
            # But the caller knows n,l.
            # Actually, let's approximate <r> from the orbital itself!
            # <r> = Integrate r * |u|^2 dr.
            r_vec = grid.r
            u2 = np.abs(orb.u_of_r)**2
            expectation_r = np.trapz(r_vec * u2, r_vec)
            
            # rc
            rc_val = 1.3 * expectation_r # Slightly tighter than 1.5
            
            # alpha_d
            # Use Closure approx scaling: alpha ~ (4/9) * <r^2>^2 ? No.
            # alpha ~ 2 * <r^2>.
            # Let's compute <r^2>.
            expectation_r2 = np.trapz(r_vec**2 * u2, r_vec)
            # Unsold uses alpha approx 4.5 a0^3 for H.
            # Approx relation: alpha_d approx (2/3?) * <r^2 / deltaE>.
            # Empirical for H-like: alpha = 4.5 * (a0/Z)^3 * n^k ?
            # Let's use simple scaling from H 1s: alpha = 4.5 * (expectation_r / 1.5)**3 ?
            # Or better: alpha = 4.5 * (expectation_r2 / 3.0)**2 ?
            # Let's use <r^2>^2 scaling. <r^2>_H1s = 3.
            # alpha_d_H1s = 4.5.
            # So alpha_d = 0.5 * (<r^2>)^? 
            # Actually a known strong bound is alpha >= 16/9 <r^2>^2 / N.
            # Let's take alpha_d = 0.5 * (expectation_r2)**(1.5)? 
            # Robust fallback: alpha_d = 4.5 * expectation_r**3.
            # (1.5**3 = 3.375. 4.5).
            alpha_val = 1.3 * (expectation_r**3) 
            
            return alpha_val, rc_val
            
        Z_eff_est = 1.0 # Effective charge? Hard to guess. Use geometrical estimation.
        
        a_i, rc_i = estimate_pol_params(orbital_initial, 1.0)
        a_f, rc_f = estimate_pol_params(orbital_final, 1.0)
        
        V_pol_i = polarization_potential(grid, a_i, rc_i)
        V_pol_f = polarization_potential(grid, a_f, rc_f)
        
        U_i_arr += V_pol_i
        U_f_arr += V_pol_f

    U_i = DistortingPotential(U_of_r=U_i_arr, V_hartree_of_r=V_H_i)
    U_f = DistortingPotential(U_of_r=U_f_arr, V_hartree_of_r=V_H_f)

    return U_i, U_f

def inspect_distorting_potential(
    grid: RadialGrid,
    U: DistortingPotential,
    n_preview: int = 5
) -> Tuple[float, float]:
    """
    diagnostic helper, analogous to inspect_core_potential().

    Checks the behavior of the distorting potential at small r and large r.

    From the article's perspective:
    - at very large r, we require U(r) -> 0,
    - close to the nucleus, U(r) approx V_core(r) + large positive Hartree part,
      so the sign might be non-trivial.

    Parameters
    ----------
    grid : RadialGrid
        Radial grid.
    U : DistortingPotential
        Result from build_distorting_potentials(...).
    n_preview : int
        Number of first/last points to use for estimation.

    Returns
    -------
    (U_small, U_large) : tuple of floats
        Approximate value of U(r) at the smallest and largest grid r.
        This provides a quick sanity check.

    Notes
    -----
    Physically we expect:
    - |U_large| << |V_core_large|, it should go to zero asymptotically.
    """
    r = grid.r
    n_preview = max(1, min(n_preview, r.size))

    U_small = float(U.U_of_r[0])
    U_large = float(U.U_of_r[-1])

    return U_small, U_large
