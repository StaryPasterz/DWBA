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
from typing import Dict, List

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


def _kernel_row_A_L(
    r1_val: float,
    r_grid: np.ndarray,
    L: int,
    V_core_r1: float,
    U_i_r1: float
) -> np.ndarray:
    """
    Build A_L(r1, r2_j) for *fixed* r1 and all r2_j on the grid,
    without allocating the full N x N kernel.

    Theory:
        A_L(r1,r2) = r_<^L / r_>^{L+1} + [V_core(r1) - U_i(r1)] δ_{L,0}

    where:
        r_< = min(r1, r2),
        r_> = max(r1, r2).

    For a given r1:
        For each r2:
            if r2 <= r1:
                r_< = r2, r_> = r1
            else:
                r_< = r1, r_> = r2

    We vectorize this for speed.

    Parameters
    ----------
    r1_val : float
        Fixed radius r1.
    r_grid : np.ndarray, shape (N,)
        All radii r2.
    L : int
        Multipole rank.
    V_core_r1 : float
        V_core(r1) in Hartree.
    U_i_r1 : float
        U_i(r1) in Hartree. (Entrance-channel distorted potential.)
        Appears only in monopole correction.

    Returns
    -------
    Arow : np.ndarray, shape (N,)
        Values of A_L(r1, r2_j) for all j.

    Notes
    -----
    Units: in atomic units, Coulomb kernel 1/|r1-r2| has dimensions 1/length.
    After angular reduction and integration with u,χ (which themselves have
    dimensions ~ length^(1/2)), final amplitude is dimensionless up to
    standard scattering normalization factors. We'll assemble those later.
    """
    r2 = r_grid

    # r_< and r_> arrays
    r_less = np.minimum(r1_val, r2)
    r_gtr = np.maximum(r1_val, r2)

    # Coulomb multipole kernel r_<^L / r_>^(L+1)
    with np.errstate(divide='ignore', invalid='ignore'):
        base_kernel = (r_less ** L) / (r_gtr ** (L + 1))

    # numeric safety near r=0: our grid never includes r=0, so no true singularity;
    # but if r1_val is extremely small, r_gtr can also be extremely small.
    # In practice r_min ~ 1e-5, so it's fine.
    # Still, replace any NaN/inf from roundoff with 0.
    bad = ~np.isfinite(base_kernel)
    if np.any(bad):
        base_kernel[bad] = 0.0

    if L == 0:
        # add [V_core(r1)-U_i(r1)] term
        correction = (V_core_r1 - U_i_r1)
        Arow = base_kernel + correction
    else:
        Arow = base_kernel

    return Arow


def radial_ME_single_L(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    U_i_array: np.ndarray,
    bound_i: BoundOrbital,
    bound_f: BoundOrbital,
    cont_i: ContinuumWave,
    cont_f: ContinuumWave,
    L: int
) -> float:
    """
    Compute the radial DWBA integral I_L for given multipole L.

    Discretized version of:
        I_L =
        ∫ dr1 ∫ dr2
        χ_f(r1) u_f(r2) A_L(r1,r2) u_i(r2) χ_i(r1)

    with
        A_L(r1,r2) = r_<^L / r_>^{L+1}
                     + [V_core(r1) - U_i(r1)] δ_{L,0}

    Numerically we evaluate:
        I_L ≈ Σ_i w_i χ_f[i] χ_i[i]
                  * [ Σ_j w_j u_f[j] u_i[j] A_L(r_i, r_j) ]

    gdzie:
        - w_i, w_j to wagi całkowania trapezowego z grid.w_trapz,
        - wszystko jest próbnikowane na tej samej siatce radialnej.

    Zwracamy liczbę rzeczywistą (float64).

    Parametry
    ---------
    grid : RadialGrid
        Siatka radialna (r, w_trapz).
    V_core_array : np.ndarray, shape (N,)
        V_{A+}(r) w Hartree na siatce grid.r.
    U_i_array : np.ndarray, shape (N,)
        Potencjał zniekształcający kanału wejściowego U_i(r) w Hartree,
        na tej samej siatce. (Występuje w części δ_{L,0}.)
    bound_i : BoundOrbital
        Orbital związany stanu początkowego Φ_i (u_i(r)).
    bound_f : BoundOrbital
        Orbital związany stanu końcowego Φ_f (u_f(r)).
    cont_i : ContinuumWave
        Fala rozpraszania w kanale wejściowym χ_i(r).
        UWAGA: musi być policzona przy energii wejściowej elektron/pocisk.
    cont_f : ContinuumWave
        Fala rozpraszania w kanale wyjściowym χ_f(r),
        przy energii wyjściowej (po utracie ΔE).
    L : int
        Rząd multipola (L_T w artykule). L = 0,1,2,...

    Zwraca
    -------
    I_L : float
        Wartość radialnej całki dla multipola L.

    Błędy
    -----
    ValueError, jeśli kształty nie pasują albo są NaNy.

    Uwaga fizyczna
    --------------
    To jest *radialna* część amplitudy DWBA. Pełna amplituda (f i g
    w artykule) ma dodatkowo współczynniki kątowe (Clebsch-Gordan,
    Wigner 3j, 6j, fazy (-1)^...), spinowe itp., które zależą
    od konkretnego przejścia (ℓ_i -> ℓ_f). Te współczynniki
    zrobimy później na poziomie wyżej.
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

    # wejściowa część (czynnik od r1):
    # outer_factor[i] = w[i] * χ_f[i] * χ_i[i]
    outer_factor = w * chi_f * chi_i

    # wewnętrzna część (czynnik od r2):
    # inner_base[j] = w[j] * u_f[j] * u_i[j]
    inner_base = w * u_f * u_i

    I_L_accum = 0.0

    # stream po i:
    for i in range(r.size):
        r1 = r[i]
        # zbuduj A_L(r1, r2_j) dla wszystkich j
        Arow = _kernel_row_A_L(
            r1_val=r1,
            r_grid=r,
            L=L,
            V_core_r1=V_core_array[i],
            U_i_r1=U_i_array[i]
        )
        # składamy wewnętrzną sumę po r2:
        # inner_sum_i = Σ_j inner_base[j] * A_L(r1, r2_j)
        inner_sum_i = float(np.dot(inner_base, Arow))

        # dorzucamy do całki z czynnikiem outer_factor[i]
        I_L_accum += outer_factor[i] * inner_sum_i

    # sanity
    if not np.isfinite(I_L_accum):
        raise ValueError("radial_ME_single_L: non-finite result (overflow/NaN).")

    return float(I_L_accum)


def radial_ME_all_L(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    U_i_array: np.ndarray,
    bound_i: BoundOrbital,
    bound_f: BoundOrbital,
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
