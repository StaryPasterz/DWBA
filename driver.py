# driver.py
#
# High-level orchestration for computing electron-impact excitation
# cross sections in DWBA, following the article.
#
# Pipeline implemented here:
#
#   1. Build radial grid.
#   2. Build core potential V_{A+}(r).
#   3. Solve bound states for initial and final orbitals (Φ_i, Φ_f).
#   4. Build distorted potentials U_i(r), U_f(r).
#   5. Solve distorted-wave scattering states χ_i(k_i,r), χ_f(k_f,r).
#   6. Compute DWBA radial integrals I_L.
#   7. Build angular amplitudes f(θ), g(θ) and integrate to get σ_DWBA.
#   8. Optionally apply empirical M-Tong scaling.
#
# We return both raw DWBA σ and M-Tong-scaled σ, in a0^2 and cm^2.
#
# Assumptions:
#   - Single-active-electron model (the article's model).
#   - Distorted potentials U_i, U_f are constructed as in the article,
#     so that they go to 0 at large r.
#   - Distorted waves χ are real with asymptotic unit amplitude (continuum.py).
#   - Angular coupling (direct/exchange, Clebsch-Gordan/Wigner factors)
#     is factored out and provided via build_angular_coeffs_for_transition().
#
# Units:
#   - Internally everything is atomic units (hartree for energies, bohr for length).
#   - User-facing energy is in eV.
#   - Cross sections reported in both a0^2 and cm^2.
#
# This module is designed to have NO heavy numerics itself
# (besides a couple of sqrt). It mostly calls the physics modules.


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import time

from grid import (
    RadialGrid,
    make_r_grid,
    ev_to_au,
    k_from_E_eV,
)
from potential_core import (
    CorePotentialParams,
    V_core_on_grid,
)
from bound_states import (
    solve_bound_states,
    BoundOrbital,
)
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential,
)
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from dwba_matrix_elements import (
    radial_ME_all_L,
    RadialDWBAIntegrals,
)
from sigma_total import (
    DWBAAngularCoeffs,
    build_angular_coeffs_placeholder,
    f_theta_from_coeffs,
    dcs_dwba,
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
)
from dwba_coupling import (
    ChannelAngularInfo,
    build_angular_coeffs_for_channel,
)


@dataclass(frozen=True)
class ExcitationChannelSpec:
    """
    Specification of a particular excitation channel Φ_i -> Φ_f
    for which we want σ_DWBA.

    Attributes
    ----------
    l_i : int
        Orbital angular momentum ℓ of the initial bound electron state Φ_i.
        Example: ℓ=1 for p-like.
    l_f : int
        Orbital angular momentum ℓ of the final bound electron state Φ_f.
        (After excitation.)
    n_index_i : int
        Which eigenstate (1 = lowest, 2 = next, ...) within that ℓ to
        treat as Φ_i. See BoundOrbital.n_index.
    n_index_f : int
        Same but for the final state.
    N_equiv : int
        Number of equivalent electrons in that subshell in the initial
        configuration. This is the degeneracy factor N that shows up
        in DWBA cross sections (how many electrons can be excited
        the same way).
    L_max : int
        Maximum multipole rank L to include in the DWBA sum.
        For dipole-like transitions typically L_max=2 or 3 is enough.
    L_i_total : int
        Total orbital angular momentum quantum number used for averaging
        in the cross section formula (1/(2L_i+1)).
        In the simplest LS picture for a single electron this is often l_i.
        In jj-coupling this would be J_i.
    """
    l_i: int
    l_f: int
    n_index_i: int
    n_index_f: int
    N_equiv: int
    L_max: int
    L_i_total: int


@dataclass(frozen=True)
class DWBAResult:
    """
    Final result bundle for one excitation channel and one incident energy.

    Attributes
    ----------
    ok_open_channel : bool
        True jeśli kanał jest energetycznie otwarty (k_f realne, E_f>0).
        False jeśli energia zderzenia była za mała, wtedy σ=0.
    E_incident_eV : float
        Energia początkowa elektronu-projektylu w eV (wejściowa).
    E_excitation_eV : float
        Energia wzbudzenia celu w eV = (E_final_bound - E_initial_bound)*27.211...
        (dodatnia wartość jest progiem; jeśli E_incident_eV < E_excitation_eV,
         kanał jest zamknięty).
    sigma_total_au : float
        Całkowity przekrój czynny DWBA w a0^2.
    sigma_total_cm2 : float
        Ten sam przekrój w cm^2.
    sigma_mtong_au : float
        (Na razie identyczne co sigma_total_au.)
        Miejsce na skalowanie M-Tong opisane w artykule.
    sigma_mtong_cm2 : float
        (Na razie identyczne co sigma_total_cm2.)
    k_i_au : float
        Początkowy pęd elektronu (k_i) w jednostkach atomowych.
    k_f_au : float
        Końcowy pęd elektronu (k_f) w jednostkach atomowych.
    """
    ok_open_channel: bool
    E_incident_eV: float
    E_excitation_eV: float
    sigma_total_au: float
    sigma_total_cm2: float
    sigma_mtong_au: float
    sigma_mtong_cm2: float
    k_i_au: float
    k_f_au: float


def _pick_bound_orbital(orbs: Tuple[BoundOrbital, ...], n_index_wanted: int) -> BoundOrbital:
    """
    Wybierz z listy stan związany o zadanym n_index.
    Jeśli nie istnieje, rzucamy ValueError.
    """
    for o in orbs:
        if o.n_index == n_index_wanted:
            return o
    raise ValueError(f"Requested bound state n_index={n_index_wanted} not found in solve_bound_states output.")


def _compute_excitation_energy_au(orb_i: BoundOrbital, orb_f: BoundOrbital) -> float:
    """
    Energia wzbudzenia celu:
        ΔE_target = E_f - E_i  (obie energie są <0 Hartree dla stanów związanych).
    Jeśli E_f jest mniej ujemna (bliżej zera), to ΔE_target > 0.
    To jest bariera energetyczna, którą musi pokonać elektron pocisk.

    Zwracamy ΔE_target w Hartree.
    """
    return orb_f.energy_au - orb_i.energy_au


def _build_angular_coeffs_for_transition(
    I_L_dict: Dict[int, float],
    chan: ExcitationChannelSpec
) -> DWBAAngularCoeffs:
    """
    Miejsce na właściwą algebrę kątową z artykułu.

    W finalnej wersji:
        - użyjesz l_i, l_f, L, zasady wyboru,
        - zbudujesz F_L (direct) i G_L (exchange) zgodnie z równaniami w artykule,
          czyli uwzględnisz Wigner 3j/6j, (-1)^..., sprzężenie spinowe itd.

    Na razie korzystamy z placeholdera, który po prostu traktuje I_L
    jako F_L i G_L z ewentualnymi prostymi mnożnikami fazowymi.
    To pozwala zrobić pełny pipeline, a w następnym etapie
    wstawimy konkretną fizykę dla danego przejścia.

    Parametry
    ---------
    I_L_dict : dict[int,float]
        Radialne całki I_L.
    chan : ExcitationChannelSpec
        Informacje o przejściu (l_i, l_f, ...). Tu jeszcze ich nie używamy.

    Zwraca
    -------
    DWBAAngularCoeffs
        Słownik {L -> F_L}, {L -> G_L}.
    """
    # TODO: implementować rzeczywiste współczynniki kątowe wg artykułu.
    #       Na razie: użyj placeholdera 1:1.
    coeffs = build_angular_coeffs_placeholder(I_L_dict)
    return coeffs


def _apply_mtong_scaling(
    sigma_total_au: float,
    E_incident_eV: float,
    E_excitation_eV: float
) -> float:
    """
    Hak pod tzw. 'M-Tong scaling' (albo inne dopasowanie kalibracyjne),
    który artykuł stosuje żeby dopasować DWBA do danych eksperymentalnych
    i poprawić zachowanie przy progu.

    Prawdziwa implementacja (zgodnie z artykułem):
    - wprowadza pewien empiryczny współczynnik energio-zależny f(E),
      kalibrowany tak, by pasował do dostępnych danych lub do modelu Tong.
    - Ten współczynnik zwykle zależy m.in. od odległości od progu
      (E_incident - E_excitation), kinematyki kanału itp.
    - Ważne: jest stosowany do SIGMY, nie do elementu macierzowego.

    Ponieważ nie mamy tu jeszcze jawnych parametrów dopasowania,
    zwracamy wartość bez zmiany.

    Parametry
    ---------
    sigma_total_au : float
        Surowy przekrój DWBA w a0^2.
    E_incident_eV : float
        Energia początkowa wiązki (eV).
    E_excitation_eV : float
        Próg wzbudzenia (eV).

    Zwraca
    -------
    sigma_corr_au : float
        (Na razie identyczne sigma_total_au.)
    """
    # TODO: wstawić faktyczną korektę M-Tong z artykułu.
    return sigma_total_au


def compute_total_excitation_cs(
    E_incident_eV: float,
    chan: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-5,
    r_max: float = 200.0,
    n_points: int = 4000,
) -> DWBAResult:
    """
    Główna funkcja high-level:
    Dla podanej energii elektronu pocisku i konkretnego przejścia
    Φ_i(l_i, n_index_i) -> Φ_f(l_f, n_index_f),
    policz całkowity przekrój czynny wzbudzenia w DWBA
    zgodnie z artykułem, oraz jego wersję z ewentualną korekcją M-Tong.

    Kroki (jak w artykule):
    1. Siatka radialna (r_min..r_max).
    2. Potencjał rdzenia V_{A+}(r).
    3. Rozwiązanie stanów związanych Φ_i i Φ_f.
    4. Potencjały zniekształcające U_i, U_f.
    5. Rozwiązanie fal rozpraszania χ_i(k_i,r), χ_f(k_f,r).
    6. Całki radialne I_L.
    7. Amplitudy kątowe f,g → dσ/dΩ → σ.
    8. Skalowanie M-Tong.

    Parametry
    ---------
    E_incident_eV : float
        Energia kinetyczna elektronu pocisku (wejściowa) w eV.
    chan : ExcitationChannelSpec
        Opis kanału wzbudzenia (l_i, l_f, n_indexy, degeneracje, itp.).
    core_params : CorePotentialParams
        Parametry potencjału rdzeniowego V_{A+}(r) (Tong/Lin-type),
        zgodnie z artykułem.
    r_min, r_max : float
        Zakres siatki radialnej [bohr].
        r_max MUSI być na tyle duży, żeby u(r) i U_j(r) wygasły,
        bo w przeciwnym razie fale χ nie będą miały poprawnej asymptotyki.
    n_points : int
        Liczba punktów siatki radialnej.

    Zwraca
    -------
    DWBAResult
        Pudełko z σ_DWBA (a0^2 i cm^2), σ_MTong, energią wzbudzenia itd.

    Rzuca
    -----
    ValueError jeśli kanał jest zamknięty energetycznie (E_incident < próg),
    to zwrócimy σ=0 ale ok_open_channel=False (nie rzucamy, tylko flagujemy).
    """

    t0 = time.perf_counter()
    
    # 1. Grid
    grid: RadialGrid = make_r_grid(r_min=r_min, r_max=r_max, n_points=n_points)
    t1 = time.perf_counter()

    # 2. Core potential
    V_core = V_core_on_grid(grid, core_params)
    t2 = time.perf_counter()

    # 3. Bound states for l_i and l_f
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=max(chan.n_index_i, chan.n_index_i+2))
    states_f = solve_bound_states(grid, V_core, l=chan.l_f, n_states_max=max(chan.n_index_f, chan.n_index_f+2))

    orb_i = _pick_bound_orbital(tuple(states_i), chan.n_index_i)
    orb_f = _pick_bound_orbital(tuple(states_f), chan.n_index_f)
    
    # Energia wzbudzenia celu (Hartree)
    dE_target_au = _compute_excitation_energy_au(orb_i, orb_f)
    # w eV:
    dE_target_eV = dE_target_au / ev_to_au(1.0)  # ev_to_au(1 eV) = 1/27.2114..., więc dzielenie jest ok

    t3 = time.perf_counter()

    # 4. Distorting potentials U_i, U_f
    U_i, U_f = build_distorting_potentials(
        grid=grid,
        V_core_array=V_core,
        orbital_initial=orb_i,
        orbital_final=orb_f
    )
    t4 = time.perf_counter()
    # U_i.U_of_r i U_f.U_of_r to potencjały używane przy rozwiązywaniu χ_i i χ_f.

    # 5. Continuum waves χ_i, χ_f
    # Energia pocisku przed zderzeniem = E_incident_eV.
    # Energia pocisku po zderzeniu = E_incident_eV - dE_target_eV.
    E_final_eV = E_incident_eV - dE_target_eV

    # k_i, k_f w a.u.
    k_i_au = float(k_from_E_eV(E_incident_eV)) if E_incident_eV > 0.0 else 0.0
    k_f_au = float(k_from_E_eV(E_final_eV))    if E_final_eV    > 0.0 else 0.0

    # Kanał jest otwarty tylko jeśli końcowy elektron ma dodatnią energię.
    channel_open = (E_final_eV > 0.0) and (k_i_au > 0.0) and (k_f_au > 0.0)

    if not channel_open:
        # Kanał zamknięty energetycznie: brak wzbudzenia, σ=0.
        return DWBAResult(
            ok_open_channel=False,
            E_incident_eV=E_incident_eV,
            E_excitation_eV=dE_target_eV,
            sigma_total_au=0.0,
            sigma_total_cm2=0.0,
            sigma_mtong_au=0.0,
            sigma_mtong_cm2=0.0,
            k_i_au=k_i_au,
            k_f_au=k_f_au,
        )

    # Rozwiąż χ_i w potencjale U_i, przy energii E_incident_eV
    chi_i: ContinuumWave = solve_continuum_wave(
        grid=grid,
        U_channel=U_i,
        l=chan.l_i,      # UWAGA: w zasadzie częściowa fala pocisku nie MUSI mieć to samo l co orbital celu.
                         # W ogólnej teorii χ ma własne l_scatt. Finalnie będzie trzeba sumować po l_scatt.
                         # Ten driver zakłada na razie pojedynczą dominującą składową l=chan.l_i dla uproszczenia.
        E_eV=E_incident_eV
    )

    # Rozwiąż χ_f w potencjale U_f, przy energii E_final_eV
    chi_f: ContinuumWave = solve_continuum_wave(
        grid=grid,
        U_channel=U_f,
        l=chan.l_f,      # analogiczny komentarz jak wyżej
        E_eV=E_final_eV
    )

    t5 = time.perf_counter()

    # 6. Radial integrals I_L dla L=0..L_max
    radial_ints: RadialDWBAIntegrals = radial_ME_all_L(
        grid=grid,
        V_core_array=V_core,
        U_i_array=U_i.U_of_r,        # U_i(r) wchodzi w operator V_i(r1,r2)
        bound_i=orb_i,
        bound_f=orb_f,
        cont_i=chi_i,
        cont_f=chi_f,
        L_max=chan.L_max
    )
    I_L_dict: Dict[int, float] = radial_ints.I_L

    t6 = time.perf_counter()

    # 7. Angular coefficients F_L, G_L
    chan_info = ChannelAngularInfo(
        l_i=chan.l_i,
        l_f=chan.l_f,
        S_i=float(0.5),  # <-- tymczasowo; tu powinieneś wstawić prawdziwe S_i z konfiguracji celu
        S_f=float(0.5),  # <-- jw.
        L_i=float(chan.L_i_total),  # uproszczone; przy LS-coupling tu trzeba wstawić L_i celu
        L_f=float(chan.l_f),        # jw. placeholder
        J_i=float(chan.L_i_total),  # placeholder; jeśli masz J, wpisz J_i
        J_f=float(chan.l_f),        # placeholder
    )
    coeffs = build_angular_coeffs_for_channel(I_L_dict, chan_info)

    # 8. Zbuduj amplitudy f(θ), g(θ), policz dσ/dΩ(θ), zintegruj  σ_total
    theta_grid = np.linspace(0.0, np.pi, 400)
    cos_theta = np.cos(theta_grid)

    f_theta, g_theta = f_theta_from_coeffs(cos_theta, coeffs)

    dcs_theta = dcs_dwba(
        theta_grid=theta_grid,
        f_theta=f_theta,
        g_theta=g_theta,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
        L_i=chan.L_i_total,
        N_equiv=chan.N_equiv
    )

    sigma_total_au = integrate_dcs_over_angles(theta_grid, dcs_theta)
    sigma_total_cm2 = sigma_au_to_cm2(sigma_total_au)

    # 9. M-Tong scaling hook
    sigma_mtong_au = _apply_mtong_scaling(
        sigma_total_au=sigma_total_au,
        E_incident_eV=E_incident_eV,
        E_excitation_eV=dE_target_eV
    )
    sigma_mtong_cm2 = sigma_au_to_cm2(sigma_mtong_au)

    print("Timing diagnostic:")
    print(f"  grid            {t1-t0:.3f} s")
    print(f"  V_core          {t2-t1:.3f} s")
    print(f"  bound states    {t3-t2:.3f} s")
    print(f"  distorting pots {t4-t3:.3f} s")
    print(f"  continuum waves {t5-t4:.3f} s")
    print(f"  DWBA radial     {t6-t5:.3f} s")

    # 10. Zbierz wynik
    return DWBAResult(
        ok_open_channel=True,
        E_incident_eV=E_incident_eV,
        E_excitation_eV=dE_target_eV,
        sigma_total_au=sigma_total_au,
        sigma_total_cm2=sigma_total_cm2,
        sigma_mtong_au=sigma_mtong_au,
        sigma_mtong_cm2=sigma_mtong_cm2,
        k_i_au=k_i_au,
        k_f_au=k_f_au,
    )
