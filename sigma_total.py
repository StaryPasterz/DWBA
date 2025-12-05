# sigma_total.py
#
# DWBA differential and total cross sections for electron-impact excitation,
# following the structure given in the article.
#
# This module sits on top of:
#   - radial integrals I_L from dwba_matrix_elements.py
#   - distorted waves χ_i, χ_f (continuum.py)
#   - bound orbitals Φ_i, Φ_f (bound_states.py)
#
# It does three jobs:
#   1. Build scattering amplitudes f(θ), g(θ) from multipole contributions.
#   2. Compute dσ/dΩ(θ) using the DWBA spin-weighted combination.
#   3. Integrate over θ to get total σ_DWBA(E).
#
# Physics recap:
# --------------
# The DWBA amplitudes consist of:
#   - direct term f(θ)
#   - exchange term g(θ)
#
# Each can be expanded in Legendre polynomials:
#   f(θ) = Σ_L F_L P_L(cosθ)
#   g(θ) = Σ_L G_L P_L(cosθ)
#
# The coefficients F_L, G_L are built from:
#   - radial multipole integrals I_L,
#   - angular/spin coupling coefficients (Clebsch-Gordan, 3j/6j, etc.),
#   - possible (-1)^... phase factors (from exchange antisymmetrization).
#
# The article then gives (unpolarized initial beam, spin-1/2 electrons):
#
#   dσ/dΩ = (k_f / k_i)
#           * [ (3/4)|f(θ) - g(θ)|^2
#               + (1/4)|f(θ) + g(θ)|^2 ]
#           * (1 / (2L_i + 1)) * N
#
# where:
#   - k_i, k_f are incoming/outgoing momenta (a.u.),
#   - L_i is the total orbital angular momentum quantum number of the
#     target electron in the initial state (or J_i depending on coupling;
#     practically: degeneracy of the initial level -> 2L_i+1),
#   - N is the number of equivalent electrons in that subshell,
#     i.e. how many electrons can undergo the excitation.
#
# Our code below:
#   - lets you provide F_L and G_L from the angular algebra,
#   - builds f(θ), g(θ),
#   - computes dσ/dΩ(θ),
#   - integrates over θ to produce σ_total.
#
# Note:
# We assume atomic units everywhere internally. The returned cross sections
# are in atomic units of area (a0^2). Conversion to cm^2 is easy:
#   1 a0^2 ≈ 2.8002852e-17 cm^2.
# We'll expose a helper for that.
#
# Requirements:
#   numpy, scipy.special for Legendre polynomials.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from scipy.special import legendre  # P_L(x)


# Bohr radius squared in cm^2:
A0_SQUARED_CM2 = 2.8002852e-17  # (a0 ≈ 0.529177e-8 cm)^2


@dataclass(frozen=True)
class DWBAAngularCoeffs:
    """
    Angular / spin coupling coefficients for a given excitation channel.

    After performing all the Wigner algebra from the article
    (Clebsch-Gordan, 3j, 6j, etc.), the direct and exchange amplitudes
    can be written in the form:

        f(θ) = Σ_{L=0}^{L_max} F_L P_L(cosθ)
        g(θ) = Σ_{L=0}^{L_max} G_L P_L(cosθ)

    where each F_L and G_L is a *complex* number in ogólnej teorii,
    but for purely real distorted waves and purely real radial integrals
    they'll be real. We'll keep complex to be safe.

    This dataclass stores {L -> F_L} and {L -> G_L}.

    Attributes
    ----------
    F_L : dict[int, complex]
        Direct-channel multipole coefficients for f(θ).
    G_L : dict[int, complex]
        Exchange-channel multipole coefficients for g(θ).

    Notes
    -----
    You generate these from:
      - radial integrals I_L,
      - angular momentum coupling factors for the specific transition
        (ℓ_i, ℓ_f, total L,S coupling, etc.) given in the article.
    """
    F_L: Dict[int, complex]
    G_L: Dict[int, complex]


def build_angular_coeffs_placeholder(
    I_L_dict: Dict[int, float],
    phase_direct: float = 1.0,
    phase_exchange: float = 1.0,
    exchange_scale: float = 1.0
) -> DWBAAngularCoeffs:
    """
    TEMPORARY PLACEHOLDER / SCAFFOLD.

    This function maps the radial integrals I_L directly into F_L and G_L
    with trivial weights. This is NOT the final physics, it's a hook.

    Why we need this:
    -----------------
    The true expressions for F_L and G_L in the article involve:
      - Wigner 3j / 6j symbols,
      - Clebsch-Gordan coefficients coupling ℓ_i, ℓ_f, L,
      - spin algebra (direct vs exchange),
      - parity / phase factors (-1)^(ℓ_i+ℓ_f+L) etc.
    Those depend on the specific transition.

    We still want the rest of the pipeline (σ, integration) to be coded
    cleanly now. So here we offer a trivial mapping:
        F_L = phase_direct   * I_L
        G_L = phase_exchange * exchange_scale * I_L

    Later:
      - We'll replace this with a real function that implements eqs. from
        the paper for direct and exchange amplitudes.

    Parameters
    ----------
    I_L_dict : dict[int, float]
        Radial integrals I_L computed in dwba_matrix_elements.radial_ME_all_L.
    phase_direct : float
        Overall phase for direct amplitude.
    phase_exchange : float
        Overall phase for exchange amplitude.
    exchange_scale : float
        Relative scale between direct and exchange channels.

    Returns
    -------
    DWBAAngularCoeffs
        {L -> F_L}, {L -> G_L} usable for f(θ) and g(θ).
    """
    F = {}
    G = {}
    for L, I_L_val in I_L_dict.items():
        F[L] = complex(phase_direct * I_L_val)
        G[L] = complex(phase_exchange * exchange_scale * I_L_val)
    return DWBAAngularCoeffs(F_L=F, G_L=G)


def f_theta_from_coeffs(
    cos_theta: np.ndarray,
    coeffs: DWBAAngularCoeffs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build f(θ) and g(θ) on a grid of scattering angles θ, given the
    multipole expansions:

        f(θ) = Σ_L F_L P_L(cosθ)
        g(θ) = Σ_L G_L P_L(cosθ)

    Parameters
    ----------
    cos_theta : np.ndarray, shape (M,)
        cosθ values at which we evaluate amplitudes.
        Typically cosθ = cos(theta_grid) with θ in [0, π].
    coeffs : DWBAAngularCoeffs
        {L -> F_L} and {L -> G_L}.

    Returns
    -------
    f_theta : np.ndarray, shape (M,)
    g_theta : np.ndarray, shape (M,)

    Notes
    -----
    - We allow complex coefficients, so f,g can be complex arrays.
    - legendre(L) returns a polynomial callable for P_L(x).
    """
    # unique L values
    Ls = sorted(set(list(coeffs.F_L.keys()) + list(coeffs.G_L.keys())))

    # prepare result arrays
    f_theta = np.zeros_like(cos_theta, dtype=complex)
    g_theta = np.zeros_like(cos_theta, dtype=complex)

    for L in Ls:
        P_L = legendre(L)(cos_theta)  # shape (M,)
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
    Compute the differential cross section dσ/dΩ(θ) according to the
    DWBA expression given in the article:

        dσ/dΩ(θ)
        = (k_f / k_i)
          * [ (3/4) |f(θ) - g(θ)|^2  +  (1/4) |f(θ) + g(θ)|^2 ]
          * (1 / (2 L_i + 1))
          * N

    Parametry
    ---------
    theta_grid : np.ndarray, shape (M,)
        Kąty rozpraszania θ w radianach, 0 ≤ θ ≤ π.
    f_theta : np.ndarray, shape (M,)
        Amplituda direct f(θ).
    g_theta : np.ndarray, shape (M,)
        Amplituda exchange g(θ).
    k_i_au : float
        Początkowy pęd elektronu w a.u. (k_i = sqrt(2 E_i_au)).
    k_f_au : float
        Końcowy pęd elektronu w a.u. (k_f = sqrt(2 E_f_au)).
    L_i : int
        Całkowity orbitalny moment (albo J w JJ-coupling) stanu początkowego
        celu, używany do uśrednienia po projekcjach. W prostym modelu
        jednego elektronu można założyć L_i = l_i.
    N_equiv : int
        Liczba równoważnych elektronów w podpowłoce, które mogą być
        wzbudzone tym kanałem (degeneracja obsadzeń).

    Zwraca
    -------
    dcs : np.ndarray, shape (M,)
        dσ/dΩ w jednostkach a0^2 / sr (bo DWBA standardowo daje
        wynik w jednostkach powierzchni na steradian).
        (a0^2 to bohr^2).

    Uwaga fizyczna
    --------------
    - Dokładnie ten czynnik z wagami 3/4 i 1/4 oraz k_f/k_i
      jest standardem DWBA dla niepolaryzowanego rozpraszania
      elektron-elektron i zgadza się z artykułem.
    - (1/(2L_i+1)) to uśrednienie po m_i.
    - N_equiv dodaje degenerację orbitalną tej podpowłoki.
    """
    if theta_grid.shape != f_theta.shape or theta_grid.shape != g_theta.shape:
        raise ValueError("dcs_dwba: theta/f/g shape mismatch.")

    if k_i_au <= 0.0 or k_f_au <= 0.0:
        # Kanał zamknięty => brak przekroju fizycznie
        return np.zeros_like(theta_grid, dtype=float)

    # kombinacje spinowe
    combo_triplet = (3.0 / 4.0) * np.abs(f_theta - g_theta) ** 2
    combo_singlet = (1.0 / 4.0) * np.abs(f_theta + g_theta) ** 2

    prefac = (k_f_au / k_i_au) * (N_equiv / float(2 * L_i + 1))

    dcs = prefac * (combo_triplet + combo_singlet)

    # to powinno być czysto rzeczywiste
    dcs_real = np.real(dcs)

    # wymuś brak ujemnych szczątkowych wartości po zaokrągleniu,
    # bo |...|^2 >=0 => wynik powinien być >=0
    dcs_real = np.clip(dcs_real, a_min=0.0, a_max=None)

    return dcs_real


def integrate_dcs_over_angles(
    theta_grid: np.ndarray,
    dcs_theta: np.ndarray
) -> float:
    """
    Zintegrować dσ/dΩ(θ) po wszystkich kierunkach, żeby dostać
    całkowity przekrój czynny σ:

        σ = ∫ dΩ (dσ/dΩ)
          = 2π ∫_0^π sinθ [dσ/dΩ(θ)] dθ

    Numerowo robimy zwykłą całkę trapezową po θ:
        σ ≈ 2π Σ_k w_k sinθ_k dσ/dΩ(θ_k)

    Parametry
    ---------
    theta_grid : np.ndarray, shape (M,)
        Kąty θ w radianach, rosnące od 0 do π.
    dcs_theta : np.ndarray, shape (M,)
        Wartości dσ/dΩ(θ) w a0^2 / sr.

    Zwraca
    -------
    sigma_total_au : float
        Całkowity przekrój czynny σ w jednostkach a0^2.

    Uwaga:
    ------
    - Wynik jest w jednostkach bohr^2 (a0^2).
    - Do konwersji na cm^2 użyj sigma_au_to_cm2().
    """
    if theta_grid.shape != dcs_theta.shape:
        raise ValueError("integrate_dcs_over_angles: shape mismatch.")

    sin_theta = np.sin(theta_grid)
    integrand = sin_theta * dcs_theta  # shape (M,)

    # całka po θ z wagą sinθ:
    # ∫_0^π sinθ dθ (...) -> trapez na theta_grid
    integral_theta = np.trapz(integrand, theta_grid)

    sigma_total = 2.0 * np.pi * integral_theta
    sigma_total = float(sigma_total)

    # sanity
    if not np.isfinite(sigma_total):
        raise ValueError("integrate_dcs_over_angles: non-finite σ.")

    if sigma_total < 0:
        # numerycznie nie powinno wyjść ujemne; jeśli minimalnie <0 przez
        # numeryczne śmieci, zetnij do 0.
        sigma_total = max(sigma_total, 0.0)

    return sigma_total


def sigma_au_to_cm2(sigma_au: float) -> float:
    """
    Przelicza przekrój czynny z jednostek a0^2 (bohr^2) na cm^2.

    1 a0^2 = ~2.8002852e-17 cm^2.

    Parametry
    ---------
    sigma_au : float
        Przekrój czynny w a0^2.

    Zwraca
    -------
    sigma_cm2 : float
        Przekrój czynny w cm^2.
    """
    return sigma_au * A0_SQUARED_CM2


def compute_sigma_dwba(
    I_L_dict: Dict[int, float],
    k_i_au: float,
    k_f_au: float,
    L_i: int,
    N_equiv: int,
    theta_samples: int = 400
) -> Tuple[float, float]:
    """
    High-level convenience:
    Policz σ_DWBA z danych radialnych I_L i parametrów kanału.

    To robi:
    1. Buduje współczynniki kątowe F_L, G_L.
       (Na razie placeholder liniowy w I_L_dict.)
    2. Liczy f(θ), g(θ) na siatce θ.
    3. Liczy dσ/dΩ(θ).
    4. Całkuje po θ, żeby dostać σ_total.
    5. Zwraca σ_total w a0^2 i w cm^2.

    Parametry
    ---------
    I_L_dict : dict[int, float]
        Radial integrals I_L z radial_ME_all_L(...). To jest czysto radialna część DWBA.
    k_i_au : float
        k_i (a.u.), pęd elektronu przed zderzeniem dla kanału wzbudzenia.
    k_f_au : float
        k_f (a.u.), pęd elektronu po zderzeniu (po utracie energii wzbudzenia).
    L_i : int
        Całkowity moment orbitalny początkowego stanu celu (do uśrednienia 1/(2L_i+1)).
        Dla pojedynczego elektronu w orbitalu l_i można zacząć od L_i = l_i.
        Jeśli używasz JJ-coupling i masz J_i, zamiast L_i wstaw J_i.
    N_equiv : int
        Liczba równoważnych elektronów, które mogą zostać wzbudzone (degeneracja obsadzeń).
    theta_samples : int
        Liczba punktów siatki kątowej θ w [0, π] używanej do całkowania.

    Zwraca
    -------
    (sigma_total_au, sigma_total_cm2) : tuple(float, float)
        σ całkowite w a0^2 oraz w cm^2.

    UWAGA
    -----
    To używa prostego placeholdera do konwersji I_L -> (F_L, G_L),
    czyli zakłada F_L ~ I_L i G_L ~ I_L. To NIE jest pełna fizyka.
    W finalnej wersji:
      - podstawiamy prawdziwe współczynniki kątowe z artykułu,
        wynikające z równań DWBA dla danego przejścia,
        z odpowiednimi znakami i wagami spinowymi.
    """
    # 1. Współczynniki kątowe z radialnych I_L
    coeffs = build_angular_coeffs_placeholder(I_L_dict)

    # 2. Siatka kątowa: θ ∈ [0, π]
    theta_grid = np.linspace(0.0, np.pi, theta_samples)
    cos_theta = np.cos(theta_grid)

    # 3. f(θ), g(θ)
    f_theta, g_theta = f_theta_from_coeffs(cos_theta, coeffs)

    # 4. dσ/dΩ(θ)
    dcs_theta = dcs_dwba(
        theta_grid=theta_grid,
        f_theta=f_theta,
        g_theta=g_theta,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
        L_i=L_i,
        N_equiv=N_equiv
    )

    # 5. Integracja po kącie
    sigma_total_au = integrate_dcs_over_angles(theta_grid, dcs_theta)
    sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)

    return sigma_total_au, sigma_total_cm2
