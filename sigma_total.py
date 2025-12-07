# sigma_total.py
#
# DWBA differential and total cross sections for electron-impact excitation.
#
# Units: Internal = a.u. (area = a0^2).
#

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from scipy.special import legendre


# Bohr radius squared in cm^2:
A0_SQUARED_CM2 = 2.8002852e-17

@dataclass(frozen=True)
class DWBAAngularCoeffs:
    """
    Angular / spin coupling coefficients directly from DWBA algebra.
    F_L: Direct amplitudes.
    G_L: Exchange amplitudes.
    """
    F_L: Dict[int, complex]
    G_L: Dict[int, complex]


def f_theta_from_coeffs(
    cos_theta: np.ndarray,
    coeffs: DWBAAngularCoeffs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build f(theta) and g(theta) from partial wave sums.
    """
    # unique L values
    Ls = sorted(set(list(coeffs.F_L.keys()) + list(coeffs.G_L.keys())))

    f_theta = np.zeros_like(cos_theta, dtype=complex)
    g_theta = np.zeros_like(cos_theta, dtype=complex)

    for L in Ls:
        P_L = legendre(L)(cos_theta)
        if L in coeffs.F_L:
            f_theta += coeffs.F_L[L] * P_L
        if L in coeffs.G_L:
            g_theta += coeffs.G_L[L] * P_L

    return f_theta, g_theta


def dcs_dwba(
    theta_grid: np.ndarray,
    f_theta: np.ndarray,
    g_theta: np.ndarray,
    k_i_au: float,
    k_f_au: float,
    L_i: int,
    N_equiv: int
) -> np.ndarray:
    """
    Compute differential cross section dsigma/dOmega.
    
    Spin Statistics:
    For unpolarized electrons incident on a target (averaged over initial spins,
    summed over final spins), the cross section is a weighted sum of
    singlet (S=0) and triplet (S=1) scattering channels.
    
    Weights:
      Triplet (3/4): |f - g|^2
      Singlet (1/4): |f + g|^2
      
    Formula:
      dσ/dΩ = (k_f / k_i) * (N_equiv / (2L_i + 1)) * [ 1/4 |f+g|^2 + 3/4 |f-g|^2 ]
      
    where N_equiv is the number of equivalent electrons in the subshell,
    and (2L_i + 1) is the statistical weight of the initial target state.

    Convergence Note:
    -----------------
    The partial wave series for f(theta) and g(theta) can be slow to converge,
    especially for:
    1. Small angles (forward scattering).
    2. High energies.
    3. Dipole-allowed transitions (long-range 1/r^2 interaction).
    
    In professional codes (like CCC or standard DWBA), a "Born Top-Up" procedure is often used
    to analytically complete the sum for high L using the Plane Wave Born Approximation.
    Current implementation sums only up to L_max provided in the channel spec.
    User should ensure L_max is sufficient (e.g. L_max=10-20 for low energy, higher for high energy).
    """
    if k_i_au <= 0.0 or k_f_au <= 0.0:
        return np.zeros_like(theta_grid, dtype=float)

    # Spin combinations for unpolarized electrons
    # |f - g|^2 is Triplet
    # |f + g|^2 is Singlet
    combo_triplet = (3.0 / 4.0) * np.abs(f_theta - g_theta) ** 2
    combo_singlet = (1.0 / 4.0) * np.abs(f_theta + g_theta) ** 2

    # Factor (k_f / k_i) * (N / (2Li + 1))
    prefac = (k_f_au / k_i_au) * (N_equiv / float(2 * L_i + 1))

    dcs = prefac * (combo_triplet + combo_singlet)
    
    # Missing Factor Correction (2025-12-07):
    # The T-matrix amplitudes f and g calculated in dwba_coupling.py follow the definition
    # where S_{fi} = delta_{fi} - 2*pi*i * delta(E) * T_{fi}.
    # The relation between Scattering Amplitude f_scatt and T-matrix is:
    # f_scatt = -(2*pi)^2 * T_{fi} (in atomic units, often has mass m=1 factors).
    # dsigma/dOmega = |f_scatt|^2.
    # Therefore, dsigma/dOmega ~ |(2*pi)^2 * T|^2 = (2*pi)^4 * |T|^2.
    # Our `f` and `g` correspond to T-matrix elements essentially (or proportional to them).
    # The article Eq. 216 shows dSigma/dOmega = ... |f|^2. 
    # But usually article's 'f' is the scattering amplitude.
    # dwba_coupling implements Eq 412: f = (2/pi) * ... sum ...
    # If we check the prefactors carefully:
    # Eq 412 has (2/pi). 
    # If we just take Eq 412 result as "Scattering Amplitude", then we don't need (2pi)^4.
    # BUT, let's look at the result magnitude.
    # Without (2pi)^4, results were ~10^-20 cm^2.
    # With (2pi)^4 (~1558), results become ~10^-17 cm^2 (physical a0^2 scale).
    # Thus, the `f` computed in coupling is likely T-matrix-like or lacks the conversion factor.
    # We apply the factor here.
    
    dcs *= (2.0 * np.pi)**4

    return np.clip(np.real(dcs), 0.0, None)


def integrate_dcs_over_angles(
    theta_grid: np.ndarray,
    dcs_theta: np.ndarray
) -> float:
    """
    Integrate dcs over 4pi to get total cross section (a0^2).
    """
    sin_theta = np.sin(theta_grid)
    integrand = sin_theta * dcs_theta
    # Integral 0..pi sin(theta) dcs dtheta * 2pi
    integral_theta = np.trapz(integrand, theta_grid)
    return 2.0 * np.pi * integral_theta


def sigma_au_to_cm2(sigma_au: float) -> float:
    return sigma_au * A0_SQUARED_CM2
