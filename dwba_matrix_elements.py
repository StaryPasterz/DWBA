# dwba_matrix_elements.py
#
# Radial DWBA matrix elements for electron-impact excitation,
# following the formulation in the article.
#
# Goal:
#   For a given transition Φ_i -> Φ_f in the target and given
#   distorted-wave scattering states χ_i(k_i,r), χ_f(k_f,r),
#   compute the *radial* integrals I_L that appear in the DWBA
#   amplitudes (direct and exchange).
#
# Theory recap (atomic units):
#
#   V_i(r1,r2) =
#       1/|r1 - r2|  +  [ V_core(r1) - U_i(r1) ]
#
#   1/|r1 - r2| multipole expansion =>
#
#       1/|r1-r2| =
#         4π Σ_{L,M} [ r_<^L / r_>^{L+1} ] Y_{LM}^*(Ω1) Y_{LM}(Ω2)
#
# After angular reduction (spherical harmonics algebra, Clebsch-Gordan,
# Wigner 3j/6j factors), the DWBA "direct" amplitude f involves radial
# integrals of the general form:
#
#   I_L =
#     ∫_0^∞ dr1 ∫_0^∞ dr2
#       χ_f(r1) u_f(r2)
#       A_L(r1,r2)
#       u_i(r2) χ_i(r1)
#
# where
#
#   A_L(r1,r2) = r_<^L / r_>^{L+1}
#                + [V_core(r1) - U_i(r1)] δ_{L,0}
#
# and u_i, u_f are reduced radial bound states (from bound_states.py),
# χ_i, χ_f are distorted-wave continuum solutions (from continuum.py),
# V_core is the SAE core potential (potential_core.py),
# U_i is the entrance-channel distorted potential (distorting_potential.py).
#
# NOTE:
#   The angular algebra (which builds the physical scattering amplitudes
#   f and g, i.e. direct and exchange, including spin-statistics factors
#   and Wigner symbols) is NOT done here. That depends on (l_i, l_f, L, S, ...)
#   and we will handle it in a higher-level function once we encode the
#   exact coupling from the article.
#
# In this file we provide:
#
#   - radial_ME_single_L(...) : compute I_L for a single multipole L
#     using stable streamed integration on a nonuniform grid.
#
#   - radial_ME_all_L(...)    : compute all I_L for L=0..L_max.
#
# This is the numerically heavy part; everything else in DWBA sits on top.
#
# All arrays are assumed real (the article treats χ as real distorted
# waves with a phase shift, not explicitly complex Coulomb waves).
#
# Units: atomic units throughout.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union

from grid import RadialGrid
from potential_core import V_core_on_grid
from bound_states import BoundOrbital
from continuum import ContinuumWave
from distorting_potential import DistortingPotential


@dataclass(frozen=True)
class RadialDWBAIntegrals:
    """
    Container for the set of radial integrals I_L needed by DWBA amplitudes.

    Attributes
    ----------
    I_L : dict[int, float]
        Dictionary mapping multipole rank L -> I_L (float, Hartree^{-?} * bohr^{?} units;
        physically it will combine with angular factors to give scattering amplitude).
        All values are real.
    """
    I_L: Dict[int, float]


def radial_ME_single_L(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    U_i_array: np.ndarray,
    bound_i: BoundOrbital,
    bound_f: Union[BoundOrbital, ContinuumWave],
    cont_i: ContinuumWave,
    cont_f: ContinuumWave,
    L: int
) -> float:
    """
    Compute the radial DWBA integral I_L for given multipole L using optimized vectorization.

    I_L = ∫ dr1 ∫ dr2 [ χ_f(r1) χ_i(r1) ] * A_L(r1,r2) * [ u_f(r2) u_i(r2) ]

    Optimization:
    - We broadcast r1 (col) and r2 (row) to build A_L matrix (N x N).
    - We compute the double integral as a vector-matrix-vector product.
    
    Memory usage:
    - A_L matrix: N*N * 8 bytes. For N=5000: ~25M * 8 = 200MB. Acceptable.
    """
    r = grid.r
    w = grid.w_trapz
    
    if not (r.shape == w.shape == V_core_array.shape == U_i_array.shape):
        raise ValueError("radial_ME_single_L: grid/core/U_i shape mismatch.")

    u_i = bound_i.u_of_r
    u_f = bound_f.u_of_r
    chi_i = cont_i.chi_of_r
    chi_f = cont_f.chi_of_r

    if not (u_i.shape == u_f.shape == chi_i.shape == chi_f.shape == r.shape):
        raise ValueError("radial_ME_single_L: wavefunction/grid shape mismatch.")

    # 1. Precompute densities (including weights for integration)
    # The radial integral I_L separates into an integral over r1 and r2.
    # We define "densities" that absorb the integration weights w_trapz.
    # 
    # rho1(r1) = w(r1) * chi_f(r1) * chi_i(r1)    (Scattering density)
    # rho2(r2) = w(r2) * u_f(r2) * u_i(r2)        (Bound-state density)
    
    rho1 = w * chi_f * chi_i
    rho2 = w * u_f * u_i
    
    # 2. Construct Kernel Matrix A_L(r1, r2)
    # The interaction kernel A_L(r1, r2) arises from the multipole expansion of 1/|r1-r2|.
    #
    # Formula:
    #    A_L(r1, r2) = r_<^L / r_>^{L+1}  +  [ V_core(r1) - U_i(r1) ] * delta_{L,0}
    #
    # We use NumPy broadcasting to create an (N x N) matrix where:
    # - rows correspond to r1
    # - columns correspond to r2 (or vice versa, depending on indexing).
    #
    # Here:
    # r1_col shape (N, 1) -> Represents r1
    # r2_row shape (1, N) -> Represents r2
    
    r1_col = r[:, np.newaxis]
    r2_row = r[np.newaxis, :]
    
    # Identify r_< (min) and r_> (max) for each pair (r1, r2)
    r_less = np.minimum(r1_col, r2_row)
    r_gtr  = np.maximum(r1_col, r2_row)
    
    # Calculate the Coulomb multipole term: r_<^L / r_>^(L+1)
    # We use errstate to safely handle the potential singularity at r=0 (though grid starts >0).
    with np.errstate(divide='ignore', invalid='ignore'):
         kernel = (r_less ** L) / (r_gtr ** (L + 1))
    
    # Sanitize kernel (replace NaNs/Infs with 0 if any appear)
    if not np.all(np.isfinite(kernel)):
        kernel[~np.isfinite(kernel)] = 0.0

    # Monopole Correction (L=0)
    # The term [ V_core(r1) - U_i(r1) ] appears only for L=0 (spherical orthogonality).
    # This term depends ONLY on r1, so it is constant across all r2 (columns).
    if L == 0:
        # V_core_array, U_i_array are 1D arrays of size N (function of r1).
        # We broadcast them to (N, 1) to add to the (N, N) kernel matrix.
        correction = (V_core_array - U_i_array)[:, np.newaxis]
        kernel += correction

    # 3. Double Integration
    # We want to compute:
    #    I_L = Sum_{i,j} rho1[i] * A_L[i,j] * rho2[j]
    #
    # In matrix notation: I_L = rho1^T @ A_L @ rho2
    #
    # Step 3a: Integrate over r2 (inner integral)
    # This computes the potential V_L(r1) induced by rho2(r2).
    # integrated_r2[i] = Sum_j A_L[i,j] * rho2[j]
    integrated_r2 = np.dot(kernel, rho2)
    
    # Step 3b: Integrate over r1 (outer integral)
    # I_L = Sum_i rho1[i] * integrated_r2[i]
    I_L = np.dot(rho1, integrated_r2)
    
    if not np.isfinite(I_L):
        raise ValueError("radial_ME_single_L: non-finite result detected.")
        
    return float(I_L)


def radial_ME_all_L(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    U_i_array: np.ndarray,
    bound_i: BoundOrbital,
    bound_f: Union[BoundOrbital, ContinuumWave],
    cont_i: ContinuumWave,
    cont_f: ContinuumWave,
    L_max: int
) -> RadialDWBAIntegrals:
    """
    Policz zestaw I_L dla L = 0..L_max.

    Dlaczego potrzebujemy wielu L?
    - multipolowy rozwój oddziaływania 1/r12 w DWBA daje wkłady o różnych
      L (monopol, dipol, kwadrupol...). W praktyce dla przejść dipolowych
      zwykle dominują niskie L (0 i 1), ale formalnie należy sumować.

    Parametry
    ---------
    grid, V_core_array, U_i_array, bound_i, bound_f, cont_i, cont_f :
        Jak w radial_ME_single_L.
        Uwaga: U_i_array to U_i(r), czyli potencjał wejściowego kanału.
        To jest ważne: w operatorze oddziaływania pojawia się
        [V_core(r1) - U_i(r1)], nie [V_core - U_f].
    L_max : int
        Najwyższy multipol, który chcemy policzyć.

    Zwraca
    -------
    RadialDWBAIntegrals
        .I_L[L] = I_L (float) dla L=0..L_max.

    Noty fizyczne
    -------------
    - W amplitudzie "direct" f z artykułu te I_L będą występować z wagami
      kątowymi (3j, 6j, fazy itd.). W amplitudzie "exchange" g pojawia się
      strukturalnie podobna całka, ale z zamienionymi rolami elektronów,
      co prowadzi do innej kombinacji kątowej i czasem innych L.
      To zrobimy w kolejnym kroku, w warstwie amplitudy.
    """
    if L_max < 0:
        raise ValueError("radial_ME_all_L: L_max must be >= 0.")

    I_L_dict: Dict[int, float] = {}
    for L in range(L_max + 1):
        I_L_val = radial_ME_single_L(
            grid=grid,
            V_core_array=V_core_array,
            U_i_array=U_i_array,
            bound_i=bound_i,
            bound_f=bound_f,
            cont_i=cont_i,
            cont_f=cont_f,
            L=L
        )
        I_L_dict[L] = I_L_val

    return RadialDWBAIntegrals(I_L=I_L_dict)
