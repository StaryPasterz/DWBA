import numpy as np

from driver import compute_total_excitation_cs, ExcitationChannelSpec, CorePotentialParams
# UWAGA: zakładam że CorePotentialParams jest już zdefiniowane w potential_core.py
#        i zaimportowane do driver.py tak jak wcześniej pokazywaliśmy.

def pretty_print_result(tag, res):
    print(f"=== {tag} ===")
    print(f"ok_open_channel      : {res.ok_open_channel}")
    print(f"E_incident_eV        : {res.E_incident_eV:.6f} eV")
    print(f"E_excitation_eV (thr): {res.E_excitation_eV:.6f} eV")
    print(f"k_i_au               : {res.k_i_au:.6f} a.u.")
    print(f"k_f_au               : {res.k_f_au:.6f} a.u.")
    print(f"sigma_total_au       : {res.sigma_total_au:.6e} a0^2")
    print(f"sigma_total_cm2      : {res.sigma_total_cm2:.6e} cm^2")
    print(f"sigma_mtong_au       : {res.sigma_mtong_au:.6e} a0^2")
    print(f"sigma_mtong_cm2      : {res.sigma_mtong_cm2:.6e} cm^2")
    print()

def main():
    # 1. Parametry efektywnego potencjału rdzeniowego.
    #
    # Tu MUSISZ wstawić coś sensownego zgodnie z tym jak zdefiniowałeś CorePotentialParams.
    # Przykład minimalny (jeśli masz coś w stylu: Z_eff + screening coefficients):
    #
    #   CorePotentialParams(
    #       Z_eff=2.0,
    #       a1=2.0, b1=1.0,
    #       a2=0.5, b2=0.2,
    #       ...
    #   )
    #
    # Jeżeli Twój CorePotentialParams wygląda inaczej, po prostu podaj prawidłowe pola.
    #
    # Ważne warunki fizyczne:
    #   - V_core(r) musi być ujemny blisko jądra (przyciąga elektron),
    #   - V_core(r) -> 0 przy r -> ∞.
    #
    # Jeśli tak NIE jest, bound_states i continuum się rozjadą.
    #
    # Poniżej placeholder. DOSTOSUJ do swojego CorePotentialParams.
    core_params = CorePotentialParams(
    a1=5.0,
    a2=2.0,
    a3=1.0,
    a4=0.5,
    a5=0.2,
    a6=0.1,
)


    # 2. Kanał wzbudzenia testowy: p -> s, degeneracja 5, L_max=2
    test_channel = ExcitationChannelSpec(
        l_i=1,
        l_f=0,
        n_index_i=1,
        n_index_f=1,
        N_equiv=5,
        L_max=2,
        L_i_total=1,
    )

    # 3. Energia wiązki [eV]
    E_incident = 200.0

    # 4. Pierwszy test siatki: domyślne 200 bohr / 4000 punktów
    res1 = compute_total_excitation_cs(
        E_incident_eV=E_incident,
        chan=test_channel,
        core_params=core_params,
        r_min=1e-5,
        r_max=100.0,
        n_points=1000,
    )
    pretty_print_result("grid_200bohr_4000pts", res1)

    '''# 5. Drugi test siatki: większy zasięg i gęstsza siatka
    res2 = compute_total_excitation_cs(
        E_incident_eV=E_incident,
        chan=test_channel,
        core_params=core_params,
        r_min=1e-5,
        r_max=300.0,
        n_points=6000,
    )
    pretty_print_result("grid_300bohr_6000pts", res2)'''

    # 6. Różnica względna przekroju
    if res1.sigma_total_au > 0 and res2.sigma_total_au > 0:
        rel_diff = abs(res2.sigma_total_au - res1.sigma_total_au) / res1.sigma_total_au
    else:
        rel_diff = np.nan
    print(f"Relative diff σ_total_au between grids: {rel_diff:.3e}")

if __name__ == "__main__":
    main()
import numpy as np

from driver import compute_total_excitation_cs, ExcitationChannelSpec, CorePotentialParams
# UWAGA: zakładam że CorePotentialParams jest już zdefiniowane w potential_core.py
#        i zaimportowane do driver.py tak jak wcześniej pokazywaliśmy.

def pretty_print_result(tag, res):
    print(f"=== {tag} ===")
    print(f"ok_open_channel      : {res.ok_open_channel}")
    print(f"E_incident_eV        : {res.E_incident_eV:.6f} eV")
    print(f"E_excitation_eV (thr): {res.E_excitation_eV:.6f} eV")
    print(f"k_i_au               : {res.k_i_au:.6f} a.u.")
    print(f"k_f_au               : {res.k_f_au:.6f} a.u.")
    print(f"sigma_total_au       : {res.sigma_total_au:.6e} a0^2")
    print(f"sigma_total_cm2      : {res.sigma_total_cm2:.6e} cm^2")
    print(f"sigma_mtong_au       : {res.sigma_mtong_au:.6e} a0^2")
    print(f"sigma_mtong_cm2      : {res.sigma_mtong_cm2:.6e} cm^2")
    print()

def main():
    # 1. Parametry efektywnego potencjału rdzeniowego.
    #
    # Tu MUSISZ wstawić coś sensownego zgodnie z tym jak zdefiniowałeś CorePotentialParams.
    # Przykład minimalny (jeśli masz coś w stylu: Z_eff + screening coefficients):
    #
    #   CorePotentialParams(
    #       Z_eff=2.0,
    #       a1=2.0, b1=1.0,
    #       a2=0.5, b2=0.2,
    #       ...
    #   )
    #
    # Jeżeli Twój CorePotentialParams wygląda inaczej, po prostu podaj prawidłowe pola.
    #
    # Ważne warunki fizyczne:
    #   - V_core(r) musi być ujemny blisko jądra (przyciąga elektron),
    #   - V_core(r) -> 0 przy r -> ∞.
    #
    # Jeśli tak NIE jest, bound_states i continuum się rozjadą.
    #
    # Poniżej placeholder. DOSTOSUJ do swojego CorePotentialParams.
    core_params = CorePotentialParams(
        Z_eff=2.0,
        a1=2.0, b1=1.0,
        a2=0.5, b2=0.2,
        a3=0.0, b3=1.0, # możesz mieć więcej parametrów, zależy jak to zdefiniowałeś
    )

    # 2. Kanał wzbudzenia testowy: p -> s, degeneracja 5, L_max=2
    test_channel = ExcitationChannelSpec(
        l_i=1,
        l_f=0,
        n_index_i=1,
        n_index_f=1,
        N_equiv=5,
        L_max=2,
        L_i_total=1,
    )

    # 3. Energia wiązki [eV]
    E_incident = 200.0

    # 4. Pierwszy test siatki: domyślne 200 bohr / 4000 punktów
    res1 = compute_total_excitation_cs(
        E_incident_eV=E_incident,
        chan=test_channel,
        core_params=core_params,
        r_min=1e-5,
        r_max=200.0,
        n_points=4000,
    )
    pretty_print_result("grid_200bohr_4000pts", res1)

    # 5. Drugi test siatki: większy zasięg i gęstsza siatka
    res2 = compute_total_excitation_cs(
        E_incident_eV=E_incident,
        chan=test_channel,
        core_params=core_params,
        r_min=1e-5,
        r_max=300.0,
        n_points=6000,
    )
    pretty_print_result("grid_300bohr_6000pts", res2)

    # 6. Różnica względna przekroju
    if res1.sigma_total_au > 0 and res2.sigma_total_au > 0:
        rel_diff = abs(res2.sigma_total_au - res1.sigma_total_au) / res1.sigma_total_au
    else:
        rel_diff = np.nan
    print(f"Relative diff σ_total_au between grids: {rel_diff:.3e}")

if __name__ == "__main__":
    main()
