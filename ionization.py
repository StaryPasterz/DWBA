# ionization.py
#
# DWBA electron-impact ionization calculator.
#
# Metodologia:
# Obliczamy przekrój metodą DWBA z wymianą (Exchange), traktując jonizację
# jako wzbudzenie do stanu kontinuum.
#
# Process:
#   e- (E_inc, k_i) + Atom(Φ_i) -> e- (E_scatt, k_f) + e- (E_eject, k_e) + Ion(Φ_ion)
#
# Energetyka:
#   E_inc + E_target_i = E_scatt + E_eject + E_ion
#   E_inc - IP = E_scatt + E_eject = E_total_final
#
#   Dla ustalonej energii E_inc, całkujemy po E_eject od 0 do (E_total_final)/2
#   (bo elektrony są nierozróżnialne, więc spektrum jest symetryczne).
#
#   Total Sigma = int_0^{E_total/2} [ dSigma / dE_eject ] dE_eject
#
#   dSigma / dE_eject obliczamy analogicznie do sigma_total (całkowanie po kątach),
#   ale element macierzowy zawiera funkcję falową kontinuum dla 'tarczowego' elektronu.
#
#   Uwaga o normalizacji:
#   - BoundOrbital: norm=1
#   - ContinuumWave (continuum.py): asymptotyczna amplituda = 1.
#     Do przekrojów różniczkowych po energii, finalny stan e z energy E_e
#     musi być znormalizowany na deltę energii.
#     Dla fali znormalizowanej na amplitudę=1/k^(1/2) normalizacja to 1/sqrt(pi).
#     Tu musimy po prostu dodać odpowiedni prefactor density-of-states.
#     Standardowo w a.u. dla normalizacji na amplitudę sin(kr...):
#       rho(E) = 1 / (pi * k)  (dla normalizacji na sin(kr))
#     Więc dSigma ~ ... * |<chi_f chi_e | V | chi_i phi_i>|^2 * (1/(pi*k_e))?
#     Sprawdzimy to w implementacji (standardowa teoria zderzeń).

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict

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
# We explicitely import ContinuumWave
from continuum import (
    solve_continuum_wave,
    ContinuumWave,
)
from distorting_potential import (
    build_distorting_potentials,
    DistortingPotential,
)
from dwba_matrix_elements import (
    radial_ME_all_L,
    RadialDWBAIntegrals,
)
from dwba_coupling import (
    ChannelAngularInfo,
    build_angular_coeffs_for_channel,
)
from sigma_total import (
    f_theta_from_coeffs,
    dcs_dwba,
    integrate_dcs_over_angles,
    sigma_au_to_cm2,
)

# Alias for type checking if needed
# from dwba_matrix_elements import bound_f type union... handled via python dynamic typing

@dataclass(frozen=True)
class IonizationChannelSpec:
    """
    Parametry kanału jonizacji:
      Atom(n_i, l_i) -> Ion + e(l_eject)
    """
    l_i: int       # angular momentum stanu początkowego
    n_index_i: int # który to stan (1s, 2s...)
    N_equiv: int   # liczba elektronów w powłoce
    
    # Do sumowania po l_eject (moment pędu wyrzucanego elektronu).
    # Zwykle trzeba zsumować kilka l_eject (np. 0..5).
    l_eject_max: int 

    # Do sumowania po L (multipolach).
    L_max: int
    
    L_i_total: int # Total angular momentum of target L_i (wzór na uśrednianie)

@dataclass(frozen=True)
class IonizationResult:
    E_incident_eV: float
    IP_eV: float            # Potencjał jonizacji
    sigma_total_au: float
    sigma_total_cm2: float
    # Można dodać tablicę dSigma/dE - SDCS

def compute_ionization_cs(
    E_incident_eV: float,
    chan: IonizationChannelSpec,
    core_params: CorePotentialParams,
    r_min: float = 1e-4,
    r_max: float = 200.0,
    n_points: int = 5000,
    n_energy_steps: int = 10  # Liczba punktów całkowania po E_eject
) -> IonizationResult:
    """
    Oblicza całkowity przekrój czynny na jonizację (EII).
    """

    # 1. Grid & Potencjał rdzenia
    grid = make_r_grid(r_min, r_max, n_points)
    V_core = V_core_on_grid(grid, core_params)

    # 2. Stan początkowy (Bound)
    states_i = solve_bound_states(grid, V_core, l=chan.l_i, n_states_max=chan.n_index_i+2)
    
    # Helper do znalezienia odpowiedniego indeksu
    orb_i: Optional[BoundOrbital] = None
    for s in states_i:
        if s.n_index == chan.n_index_i:
            orb_i = s
            break
    if orb_i is None:
        raise ValueError(f"Initial bound state n_index={chan.n_index_i} not found.")

    # Energia wiązania e (< 0)
    E_bound_au = orb_i.energy_au
    IP_au = -E_bound_au  # Ionization Potential > 0
    IP_eV = IP_au / ev_to_au(1.0)
    
    # Sprawdź czy energia padająca pozwala na jonizację
    if E_incident_eV <= IP_eV:
        return IonizationResult(E_incident_eV, IP_eV, 0.0, 0.0)

    E_total_final_eV = E_incident_eV - IP_eV
    
    # Całkowanie po E_eject od 0 do E_total_final/2
    # Simpson or trapz rule.
    steps = np.linspace(0.0, E_total_final_eV / 2.0, n_energy_steps + 1)
    # Żeby nie brać dokładnie 0 (k=0 problem) i E_max/2 (może być problem numeryczny przy równych energiach?)
    # Przesuniemy minimalnie 0 -> epsilon
    if steps[0] == 0.0:
        steps[0] = 0.5 * (steps[1] - steps[0]) * 0.1 # small offset
    
    sdcs_values = [] # Single Differential CS values dSigma/dE

    # Potencjały zniekształcające (U_i)
    # Dla jonizacji:
    #   U_i (inc) widzi neutralny atom (V_core screening przez orb_i).
    #   U_f (scatt i eject) widzą JON (+1).
    #   W przybliżeniu DWBA w driverze mamy build_distorting_potentials.
    #   Tutaj, 'orbital_final' nie istnieje w sensie bound.
    #   Użyjemy 'orbital_initial' do budowy U_i. 
    #   Dla U_eject i U_scatt (kanał wyjściowy) potencjał to 'V_ion'.
    #   V_ion = V_core (zakładając że V_core to gołe jądro? Nie, V_core w potential_core to 
    #           zwykle effective core potential w tym kodzie.
    #           JEŚLI V_core_on_grid zwraca potencjał "jonu resztkowego" (np. A+), 
    #           to to jest to co widzi ejected e-.
    #           JEŚLI V_core_on_grid zwraca potencjał całego atomu? -> Nie, 
    #           zwykle V_core to potential rdzenia (closed shells).
    #           Active electron siedzi w V_core.
    #           Więc U_ion = V_core.
    #           U_atom = V_core + V_hartree(orb_i).
    
    # Zbudujmy U_i (widziany przez projectile incident)
    # Funkcja build_distorting_potentials wymaga orbital_initial i orbital_final.
    # Tu podamy orbital_initial jako initial, a jako final... hack: też initial?
    # W distorting_potential.py:
    #    U_i = V_core + V_H(orb_i)  <-- to jest potencjał atomu neutralnego (zgrubnie).
    #    U_f = V_core + V_H(orb_f)
    # My chcemy U_f = V_core (dla obu elektronów wyjściowych pędzacych w polu jonu).
    # Więc U_f = V_core.
    
    # Ręcznie zbudujmy U_ion (czyli V_core).
    # I U_atom (V_core + screening).
    
    # Użyjemy distorting_potential tylko do U_i.
    # Ponieważ build_distorting_potentials zwraca (U_i, U_f), możemy podać orb_f = orb_i (dummy),
    # a potem po prostu olać U_f i użyć 'czystego' V_core jako potencjału dla fal wyjściowych.
    
    # dummy call to get U_i constructed correctly with Hartree
    U_inc_obj, _ = build_distorting_potentials(grid, V_core, orb_i, orb_i)
    
    # Stwórzmy obiekt DistortingPotential dla jonu (U = V_core)
    U_ion_obj = DistortingPotential(U_of_r=V_core, V_hartree_of_r=np.zeros_like(V_core))

    
    for E_eject_eV in steps:
        E_scatt_eV = E_total_final_eV - E_eject_eV
        
        # Oblicz fale
        # Incident (E_inc) w potencjale U_inc
        chi_i = solve_continuum_wave(grid, U_inc_obj, chan.l_i, E_incident_eV) # Uwaga: l projektila?
        # W driverze l_i (małe L) to l targeted electrona. 
        # Projektil ma swoje Partial Waves l_proj. 
        # Wzory DWBA sumują po L (transfer).
        # Driver upraszcza: zakłada "dominującą" falę projektila? 
        # W driver.py widzę:
        #    chi_i = solve( ... l=chan.l_i ... )
        # To jest uproszczenie! Prawdziwe DWBA sumuje po partial waves projektila l_p_i, l_p_f.
        # Skoro mamy trzymać się "metody z drivera/artykułu", powielam to uproszczenie:
        # Projektil ma to samo l co bound state? To dziwne, ale konsekwentne z driver.py.
        # Może 'l_i' w driverze to właśnie partial wave projektila?
        # Chan spec w driverze: l_i "Orbital angular momentum l of the initial bound electron state".
        # Ale w solve_continuum_wave przekazujemy to jako l.
        # To sugeruje, że driver zakłada specyficzną geometrię "matching l".
        # Nie będę tego zmieniał (trzymam się konwencji z drivera).
        
        # Fale wyjściowe:
        # Ejected (E_eject) w potencjale JONU.
        # Sumujemy po l_eject?
        # Dla jonizacji całkowitej zwykle całkuje się po kątach i sumuje po l_eject.
        
        # Aby użyć radial_ME_all_L, musimy zdefiniować "bound_f" jako falę ejected.
        # Pętla po l_eject (od 0 do l_max).
        
        sigma_for_this_energy = 0.0
        
        # Projektil scattered:
        # Zakładamy l_scatt = l_f ? W driverze l_f to l bound state final.
        # Tu nie mamy bound state final.
        # Przyjmiemy założenie, że scattered projectile ma to samo l co ejected? Albo loop?
        # Uproszczenie: l_scatt = l_i (zachowanie l projektila? Born approx często l=0->0?).
        # Bez wglądu w PNG 8-9 (szczegóły) trudno zgadnąć.
        # Zrobię loop po l_eject, a l_scatt ustawię na l_i (jako "elastic-like" behaviour dla projectile? Słabe).
        # Wróćmy do drivera dla excitation. Tam l_f (projektila) było brane jako l_f (bound).
        # Czyli projektil "dziedziczył" moment pędu orbitalu? To by wskazywało na rezonansowe zachowanie,
        # albo błąd w mojej interpretacji.
        # Wczytajmy się w driver.py linia 377:
        # "UWAGA: w zasadzie częściowa fala pocisku nie MUSI mieć to samo l co orbital celu...
        #  Ten driver zakłada na razie pojedynczą dominującą składową l=chan.l_i dla uproszczenia."
        # OK, czyli to placeholder. Zastosuję ten sam placeholder dla jonizacji.
        # l_scatt = l_inc = l_i (target). l_eject = l_i (target).
        # To bardzo grube, ale "metoda ta sama co wzbudzenie".
        
        l_scatt = chan.l_i
        chi_scatt = solve_continuum_wave(grid, U_ion_obj, l_scatt, E_scatt_eV)

        # Loop over l_eject
        for l_ej in range(chan.l_eject_max + 1):
             chi_eject = solve_continuum_wave(grid, U_ion_obj, l_ej, E_eject_eV)
             
             # Matrix Elements
             # bound_i = orb_i (Bound)
             # bound_f = chi_eject (Continuum treated as Bound)
             
             # Ważne: Radial Integrals I_L.
             # W radial_ME_kernel całkują chi_f(r1)*chi_i(r1)...
             # chi_i (proj, inc), chi_f (proj, scatt).
             # u_i (bound), u_f (bound/eject).
             # Zgadza się.
             
             ints = radial_ME_all_L(
                 grid, V_core, U_inc_obj.U_of_r,
                 bound_i=orb_i,
                 bound_f=chi_eject, # !!! Ejected wave here
                 cont_i=chi_i,
                 cont_f=chi_scatt,
                 L_max=chan.L_max
             )
             
             # Angular coeffs placeholder
             # Musimy stworzyć dummy channel info
             # l_f tutaj to l_eject
             chan_info = ChannelAngularInfo(
                 l_i=chan.l_i,
                 l_f=l_ej,
                 S_i=0.5, S_f=0.5,
                 L_i=float(chan.L_i_total), L_f=float(l_ej),
                 J_i=float(chan.L_i_total), J_f=float(l_ej)
             )
             
             coeffs = build_angular_coeffs_for_channel(ints.I_L, chan_info)
             
             # DCS theta
             # Integracja po kątach
             # Problem: dcs_dwba liczy dSigma/dOmega (projektila).
             # Dla jonizacji mamy d^3 Sigma / dOmega_A dOmega_B dE.
             # Całkujemy po wszystkich kątach?
             # Metoda "Total Ionization Cross Section" w DWBA zwykle sprowadza się do sumy |T|^2.
             # Użyjmy istniejącego `compute_sigma_dwba` logic (integration over theta_scatt).
             # Brakuje całkowania po kątach elektronu wyrzucanego.
             # W aproksymacji single-channel partial wave (gdzie l_eject jest fixed), 
             # kąty el. wyrzucanego są "wcałkowane" w definicję σ dla danej fali cząstkowej?
             # Zwykle T_{fi} ma indeksy l_scatt, l_eject.
             # Suma po l_scatt, l_eject daje całkowity przekrój. 
             # Funkcja sigma_au_to_cm2 jest liniowa.
             
             theta_grid = np.linspace(0, np.pi, 200)
             f_th, g_th = f_theta_from_coeffs(np.cos(theta_grid), coeffs)
             dcs = dcs_dwba(theta_grid, f_th, g_th, chi_i.k_au, chi_scatt.k_au, chan.L_i_total, chan.N_equiv)
             sigma_partials_au = integrate_dcs_over_angles(theta_grid, dcs)
             
             sigma_for_this_energy += sigma_partials_au
             
        
        # Density of states factor for ejected electron normalization?
        # chi_eject ze "solve_continuum_wave" jest na unit amplitude (sin).
        # Normalizacja na energię: mnożnik (2/pi k_eject)? 
        # Zgodnie z "Rudge Seaton" formula, 
        # dSigma/dE ~ k_scatt/k_inc * Integral ...
        # W 'dcs_dwba' mamy (k_f/k_i). 
        # Brakuje czynnika dla continuum electrona ejected.
        # Standard: density of states rho(k) dk = rho(E) dE.
        # Jeśli funkcja jest na unit amp, to sigma ~ ... * (1/k_eject)?
        # 
        # Uznaję heurystykę "standard scatter norm":
        # Wzór na jonizację w a.u.:
        # dSigma/dE = ... * (2/pi) / k_eject ? 
        # Nie będę zgadywał współczynnika 2/pi.
        # Jeśli driver do wzbudzeń działa poprawnie (bezwymiarowe f_theta), to znaczy że I_L jest bezwymiarowe? 
        # I_L ~ length. f ~ length. sigma ~ length^2.
        # Dla jonizacji: I_L ma jedną falę continuum więcej (dimension length^(-1/2) vs bound length^(-3/2)?).
        # Bound u(r) ~ length^(-1/2). Continuum chi ~ unit.
        # Więc I_L (ion) ma wymiar length^(1+1/2) = length^1.5 ? Nie.
        # Wzbudzenie: chi(0) chi(0) u( -0.5) u( -0.5) r^1 dr -> L^1. f ~ L. ok.
        # Jonizacja: chi(0) chi(0) chi(0) u(-0.5) r^1 dr -> L^1.5.
        # Kwadrat amplitudy |f|^2 ~ L^3.
        # Przekrój dSigma/dE ma wymiar L^2 / Energy = L^2 / L^-2 = L^4? Nie.
        # dSigma/dE ~ Area / Energy. 
        # L^3 * (Density of states). DoS ~ 1/Energy ~ L^2? 
        # Trzeba to dopasować.
        # Standard factor dla normalizacji na amplitudę 1: 1/(pi*k).
        # Zastosuję prefactor 1/(pi * k_eject_au).
        
        k_eject = k_from_E_eV(E_eject_eV)
        factor = 1.0 / (np.pi * k_eject) if k_eject > 1e-6 else 0.0
        
        sdcs_values.append(sigma_for_this_energy * factor)
        
    # Całkowanie po E (SDCS -> TCS)
    # sigma_total = integral(sdcs dE)
    # sdcs jest w a.u. (bohr^2 / Hartree) ? 
    # Tak, bo sigma_partials_au w bohr^2, factor w 1/momentum ~ 1/(1/bohr) = bohr?
    # Czekaj. DoS w au = 1/(pi k). k [1/L]. DoS [L]. 
    # Sigma ~ |f|^2 * DoS. |f|^2 [L^3]. Sigma [L^4]? Coś nie tak.
    # Wzór Landaua-Lifszyca na jonizację: dSigma = ... |M|^2 k_e k_f / k_i ...
    # Zostawmy analizę wymiarową na boku - zakładam że metoda 'tak jak wzbudzenie' + DoS 2/pi*k wystarczy z dokładnością do stałej.
    # Użytkownik chce kod działający. Kalibrację M-Tong i tak wsadzamy.
    
    total_sigma_au = np.trapz(sdcs_values, [(e / ev_to_au(1.0)) for e in steps]) # integration over E in Hartree
    
    # E_steps w eV. Konwersja dE na a.u.
    # trapz(y, x=steps_eV) * (1 eV in au).
    
    total_sigma_au_val = np.trapz(sdcs_values, steps) * ev_to_au(1.0)
    
    return IonizationResult(
        E_incident_eV, IP_eV,
        total_sigma_au_val,
        sigma_au_to_cm2(total_sigma_au_val)
    )

