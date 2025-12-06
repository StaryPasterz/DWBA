# dwba_matrix_elements.py
#
# Radial DWBA matrix elements for electron-impact excitation and ionization.
#
# THEORY & PHYSICS
# ----------------
# This module computes the radial integrals I_L involved in the T-matrix amplitudes.
#
# The T-matrix elements for a transition i -> f are defined as:
#
#   T_{fi} = < chi_f(r1) psi_f(r2..N) | V - U_f | A { psi_i(r2..N) chi_i(r1) } >
#
# In the Distorted Wave Born Approximation (DWBA) with a Single-Active-Electron (SAE) model:
# - psi_i, psi_f are target bound states (BoundOrbital).
# - chi_i, chi_f are distorted waves (ContinuumWave) for the projectile.
# - The interaction V is the Coulomb interaction 1/r12.
#
# We compute two types of radial integrals corresponding to Direct and Exchange amplitudes:
#
# 1. DIRECT Integral (I_L^D):
#    Arises from the term where electrons are NOT swapped.
#    I_L^D = Integral[ dr1 dr2  chi_f(r1) chi_i(r1) * (r_<^L / r_>^{L+1}) * u_f(r2) u_i(r2) ]
#    * Corrections: For L=0, we add the orthogonality term [V_core(r1) - U_i(r1)] * chi_f * chi_i * delta(orthog).
#
# 2. EXCHANGE Integral (I_L^E):
#    Arises from the term where the projectile electron is swapped with the target electron.
#    I_L^E = Integral[ dr1 dr2  chi_f(r2) chi_i(r1) * (r_<^L / r_>^{L+1}) * u_f(r1) u_i(r2) ]
#          = Integral[ dr1 dr2  (u_f(r1) chi_i(r1)) * (r_<^L / r_>^{L+1}) * (chi_f(r2) u_i(r2)) ]
#
# UNITS
# -----
# - All inputs and outputs are in HARTREE atomic units.
# - Lengths: Bohr radii (a0).
# - Energies: Hartree (Ha).
# - Probabilities/Integrals: Consistent with wavefunctions normalized to 1 (bound) or unit amplitude (continuum).
#
# IMPLEMENTATION NOTES
# --------------------
# - This module focuses solely on the radial part. Angular factors (3j/6j symbols) are applied in `dwba_coupling.py`.
# - Integration is performed using matrix-vector multiplication for efficiency on non-uniform grids.
# - We utilize the multipole expansion of 1/r12:  Sum_L (r_<^L / r_>^{L+1}) P_L(cos theta).
#



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
    Container for the set of radial integrals needed by DWBA amplitudes.
    
    Attributes
    ----------
    I_L_direct : dict[int, float]
        Direct radial integrals.
        Density pairs: (chi_f * chi_i) and (u_f * u_i).
    I_L_exchange : dict[int, float]
        Exchange radial integrals.
        Density pairs: (chi_f * u_i) and (chi_i * u_f).
    """
    I_L_direct: Dict[int, float]
    I_L_exchange: Dict[int, float]


def _generic_radial_integral(
    grid: RadialGrid,
    rho1: np.ndarray,
    rho2: np.ndarray,
    L: int,
    V_correction: Optional[np.ndarray] = None
) -> float:
    """
    Helper: Compute double radial integral:
       I = ∫ dr1 ∫ dr2 rho1(r1) * A_L(r1,r2) * rho2(r2)
    
    where A_L(r1,r2) = r_<^L / r_>^{L+1} + V_correction(r1) * delta_{L,0}
    """
    r = grid.r
    
    # Kernel Matrix A_L(r1, r2)
    r1_col = r[:, np.newaxis]
    r2_row = r[np.newaxis, :]
    
    r_less = np.minimum(r1_col, r2_row)
    r_gtr  = np.maximum(r1_col, r2_row)
    
    with np.errstate(divide='ignore', invalid='ignore'):
         kernel = (r_less ** L) / (r_gtr ** (L + 1))
    
    if not np.all(np.isfinite(kernel)):
        kernel[~np.isfinite(kernel)] = 0.0

    # Monopole Correction (only for L=0 and if provided)
    if L == 0 and V_correction is not None:
        correction = V_correction[:, np.newaxis]
        kernel += correction

    # Integration
    # inner: int_r2 = kernel @ rho2
    integrated_r2 = np.dot(kernel, rho2)
    # outer: I = rho1 @ int_r2
    I_val = np.dot(rho1, integrated_r2)
    
    if not np.isfinite(I_val):
        # Fallback?
        return 0.0
        
    return float(I_val)


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
    Compute set of I_L for L = 0..L_max for both DIRECT and EXCHANGE terms.

    Direct Integral:
      rho1 = w * chi_f * chi_i
      rho2 = w * u_f * u_i
      Correction (L=0): [V_core - U_i] acting on projectile coord r1.
    
    Exchange Integral:
      rho1 = w * chi_f * u_i    (outgoing electron 1 matches with initial bound 2 -> swapped?)
      rho2 = w * chi_i * u_f    (incoming electron 2 matches with final bound 1)
      
      Wait, precise definition of Exchange integral J_L (or G_L pre-factor):
      Standard "Ochkur" type exchange often involves 1/r12.
      
      Direct:  < chi_f(1) u_f(2) | V | chi_i(1) u_i(2) >
      Exchange: < chi_f(2) u_f(1) | V | chi_i(1) u_i(2) >  (swapped electrons in bra)
      
      Integration coords (r1, r2):
      Direct:
         Part 1 (V_interaction): 1/r12  ->  Standard kernel.
         Densities: (chi_f(1) chi_i(1)) and (u_f(2) u_i(2)).
         Part 2 (One-body potentials): V_core(1) etc.
         
      Exchange:
         Term < chi_f(2) u_f(1) | 1/r12 | chi_i(1) u_i(2) >
           = Int dr1 dr2 [chi_f(2) u_f(1)]* (1/r12) [chi_i(1) u_i(2)]
           = Int dr1 dr2 [u_f(1) chi_i(1)] * (1/r12) * [chi_f(2) u_i(2)]
         
         So for Exchange:
         rho1(r1) = w * u_f(r1) * chi_i(r1)
         rho2(r2) = w * chi_f(r2) * u_i(r2)
         
         Correction: The orthogonality term <u_f|u_i> or <chi_f|chi_i> usually handles 
         one-body parts. In Distorted Wave Static Exchange, the potential U is chosen 
         to cancel many terms.
         Typically, for Exchange, we ONLY compute the 1/r12 part (multipoles).
         The one-body parts [V_core - U] are often assumed to be handled by orthogonality 
         or are small. Bray's article Eq (3) T_ex = < chi_f(2) u_f(1) | (N-1)/r12 - ... | A phi_i chi_i >
         
         We will compute the 1/r12 exchange integral.
         Correction term is usually neglected or separately handled. Here we assume 0 correction for exchange.

    Parameters
    ----------
    grid, V_core_array, U_i_array, bound_i, bound_f, cont_i, cont_f :
        Standard inputs.
    L_max : int
        Highest multipole.

    Returns
    -------
    RadialDWBAIntegrals
        Contains .I_L_direct and .I_L_exchange.
    """
    if L_max < 0:
        raise ValueError("radial_ME_all_L: L_max must be >= 0.")

    u_i = bound_i.u_of_r
    u_f = bound_f.u_of_r
    chi_i = cont_i.chi_of_r
    chi_f = cont_f.chi_of_r
    w = grid.w_trapz

    # --- Precompute densities ---
    
    # Direct Densities
    # rho1_dir(r1) comes from projectile overlap: chi_f * chi_i
    rho1_dir = w * chi_f * chi_i
    # rho2_dir(r2) comes from target overlap: u_f * u_i
    rho2_dir = w * u_f * u_i
    
    # Exchange Densities
    # Int dr1 dr2 [u_f(1) chi_i(1)] * (1/r12) * [chi_f(2) u_i(2)]
    # rho1_ex(r1) = u_f * chi_i
    rho1_ex = w * u_f * chi_i
    # rho2_ex(r2) = chi_f * u_i
    rho2_ex = w * chi_f * u_i
    
    # Correction term for Direct L=0
    # [V_core(r1) - U_i(r1)]
    V_diff = V_core_array - U_i_array

    I_L_dir: Dict[int, float] = {}
    I_L_exc: Dict[int, float] = {}

    for L in range(L_max + 1):
        # Direct
        I_dir = _generic_radial_integral(grid, rho1_dir, rho2_dir, L, V_correction=V_diff)
        I_L_dir[L] = I_dir
        
        # Exchange
        # typically no core-potential correction in the standard 1/r12 exchange integral 
        # (core orthogonality usually assumed or handled elsewhere).
        I_ex = _generic_radial_integral(grid, rho1_ex, rho2_ex, L, V_correction=None)
        I_L_exc[L] = I_ex

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)

