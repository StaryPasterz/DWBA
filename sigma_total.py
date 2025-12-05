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
    Weighted sum of singlet (1/4) and triplet (3/4) channels.
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
