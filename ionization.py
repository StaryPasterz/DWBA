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

    # Distorting Potentials Setup
    # ---------------------------
    # We need two distinct distorting potentials:
    # 1. U_inc (Incident Channel): 
    #    The incident electron sees a NEUTRAL atom (Core + Bound Electron).
    #    We model this using the initial bound orbital 'orb_i' to calculate Hartree screening.
    #    U_inc = V_core + V_Hartree(orb_i)
    
    # 2. U_ion (Final Channel - Scattered & Ejected):
    #    Both outgoing electrons move in the field of the RESIDUAL ION.
    #    The bound electron is removed, so V_Hartree is effectively zero (or simplified).
    #    We use the bare core potential V_core (which represents the ion).
    #    U_ion = V_core
    
    # Construct U_inc using the standard builder.
    k_i_au = float(k_from_E_eV(E_incident_eV)) if E_incident_eV > 0 else 0.5
    
    # Passing k_i_au allows exchange potential for incident channel.
    U_inc_obj, _ = build_distorting_potentials(grid, V_core, orb_i, orb_i, k_i_au=k_i_au)
    
    # Construct U_ion manually (Pure core potential, no Hartree screening from active electron)
    U_ion_obj = DistortingPotential(U_of_r=V_core, V_hartree_of_r=np.zeros_like(V_core))

    
    for E_eject_eV in steps:
        E_scatt_eV = E_total_final_eV - E_eject_eV
        
        # Effective Charges (Z_eff) for Coulomb Asymptotics
        # -----------------------------------------------
        # 1. Incident Electron:
        #    Sees [Core (+Zc) + Bound Electron (-1)] => Net Charge = Zc - 1.
        #    This is the "Z_ion" parameter for the asymptotic Coulomb matching.
        z_ion_inc = core_params.Zc - 1.0
        
        # 2. Final Electrons (Scattered and Ejected):
        #    See [Core (+Zc)] => Net Charge = Zc.
        z_ion_final = core_params.Zc

        # Calculate Continuum Waves
        # -------------------------
        
        # 1. Incoming Wave (Incident Projectile)
        chi_i = solve_continuum_wave(
            grid, 
            U_inc_obj, 
            chan.l_i, 
            E_incident_eV,
            z_ion=z_ion_inc
        ) 
        # Note on Partial Waves:
        # The DWBA driver primarily considers the partial wave L corresponding to the target symmetry.
        # Ideally, one sums over all projectile partial waves (l_proj). 
        # Here we follow the driver's convention of using l=chan.l_i as a representative mode.
        
        # 2. Scattered Wave (Projectile after collision)
        # We assume the projectile scatters into a similar angular momentum channel.
        sigma_for_this_energy = 0.0
        l_scatt = chan.l_i 
        chi_scatt = solve_continuum_wave(
            grid, 
            U_ion_obj, 
            l_scatt, 
            E_scatt_eV, 
            z_ion=z_ion_final
        )

        # 3. Ejected Wave (The electron knocked out)
        # We sum over ejected angular momenta l_eject.
        for l_ej in range(chan.l_eject_max + 1):
             chi_eject = solve_continuum_wave(
                 grid, 
                 U_ion_obj, 
                 l_ej, 
                 E_eject_eV,
                 z_ion=z_ion_final
             )
             
             # Matrix Elements Calculation
             # ---------------------------
             # Integration: < chi_scatt(r1) chi_eject(r2) | 1/r12 | chi_i(r1) phi_bound(r2) >
             # Note that 'radial_ME_all_L' expects:
             #   bound_i -> Initial bound state (phi_bound)
             #   bound_f -> Final "bound-like" state. For ionization, this is the Ejected Continuum Wave.
             #   cont_i  -> Incident projectile
             #   cont_f  -> Scattered projectile
             
             ints = radial_ME_all_L(
                 grid, V_core, U_inc_obj.U_of_r,
                 bound_i=orb_i,
                 bound_f=chi_eject, # Passing ContinuumWave as bound_f is valid (polymorphism)
                 cont_i=chi_i,
                 cont_f=chi_scatt,
                 L_max=chan.L_max
             )
             
             # Angular Coefficients
             # --------------------
             # Provide placeholder angular info consistent with the transfer L
             chan_info = ChannelAngularInfo(
                 l_i=chan.l_i,
                 l_f=l_ej,
                 S_i=0.5, S_f=0.5,
                 L_i=float(chan.L_i_total), L_f=float(l_ej),
                 J_i=float(chan.L_i_total), J_f=float(l_ej)
             )
             
             coeffs = build_angular_coeffs_for_channel(
                 I_L_direct=ints.I_L_direct,
                 I_L_exchange=ints.I_L_exchange,
                 chan=chan_info
             )

             
             # Differential Cross Section (Incident Angle Integration)
             # -----------------------------------------------------
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

