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
# - OPTIMIZATION: We utilize a kernel recurrence relation K_L = K_{L-1} * (r_</r_>) to avoid expensive power operations.
#


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from grid import RadialGrid
from bound_states import BoundOrbital
from continuum import ContinuumWave

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
    OPTIMIZED VERSION: Uses kernel recurrence to avoid O(N^2) power operations.
    """
    if L_max < 0:
        raise ValueError("radial_ME_all_L: L_max must be >= 0.")

    u_i = bound_i.u_of_r
    u_f = bound_f.u_of_r
    chi_i = cont_i.chi_of_r
    chi_f = cont_f.chi_of_r
    w = grid.w_trapz
    r = grid.r

    # --- Precompute densities ---
    rho1_dir = w * chi_f * chi_i
    rho2_dir = w * u_f * u_i
    
    rho1_ex = w * u_f * chi_i
    rho2_ex = w * chi_f * u_i
    
    # Correction term for Direct L=0: [V_core(r1) - U_i(r1)]
    V_diff = V_core_array - U_i_array

    I_L_dir: Dict[int, float] = {}
    I_L_exc: Dict[int, float] = {}

    # --- Kernel Optimization ---
    # Construct base kernel matrices only once
    r1_col = r[:, np.newaxis]
    r2_row = r[np.newaxis, :]
    
    # We use a safe division mask for calculating ratio = r_< / r_>
    ratio = np.empty_like(r1_col * r2_row) # shape N,N
    inv_gtr = np.empty_like(ratio)
    
    r_less = np.minimum(r1_col, r2_row)
    r_gtr  = np.maximum(r1_col, r2_row)
    
    # Safely compute ratio and inv_gtr
    with np.errstate(divide='ignore', invalid='ignore'):
         ratio = r_less / r_gtr
         inv_gtr = 1.0 / r_gtr
         
    # Fix nans/infs if any (at r=0)
    if not np.all(np.isfinite(ratio)):
        ratio[~np.isfinite(ratio)] = 0.0
    if not np.all(np.isfinite(inv_gtr)):
        inv_gtr[~np.isfinite(inv_gtr)] = 0.0
        
    # Recurrence Initialization
    # For L=0: Kernel_0 = 1/r_> = inv_gtr
    # We maintain kernel_L in memory and update it inplace.
    kernel_L = inv_gtr.copy()
    
    for L in range(L_max + 1):
        if L > 0:
            # Recurrence: K_L = K_{L-1} * ratio
            kernel_L *= ratio
        
        # --- Direct Integral ---
        # I = < rho1 | Kernel | rho2 >
        # int_r2 = kernel @ rho2
        
        # dot(matrix, vector) is highly optimized in numpy
        int_r2 = np.dot(kernel_L, rho2_dir)
        I_dir = np.dot(rho1_dir, int_r2)

        # Monopole Correction (L=0 only)
        if L == 0:
            # Add correction term: Int rho1 * V_diff * Int rho2
            # because the correction to Kernel is constant V_diff(r1) along r2 ??
            # NO. The correction in formula is: 
            # I_corr = Int dr1 dr2 rho1(r1) * [V_diff(r1) * delta_L0] * rho2(r2) 
            # Note 1/r12 term is 1/r_>. The correction is purely 1-body operator.
            # But where does it enter?
            # T = < ... | V - U | ... >.
            # V - U = (V_core + 1/r12) - (V_core + V_H + V_ex).
            #       = 1/r12 - V_H - V_ex.
            # Wait, V_diff in my code was V_core - U_i. 
            # U_i = V_core + V_H. So V_diff = -V_H.
            # So we represent (1/r12 - V_H).
            # 1/r12 = sum_L (pow... P_L).
            # V_H = integral...
            # The monopolar part of 1/r12 is V_H(r).
            # So for L=0, the term is (1/r_> - V_H(r1)).
            # My code calculates 1/r_> part via kernel.
            # I need to SUBTRACT V_H part.
            # So Correction adds Integral[ rho1(r1) * (-V_H(r1)) * rho2(r2) ] ?
            # Wait, rho2 = u_f u_i. Integral rho2 ~ delta_fi (orthonormality).
            # If f != i, integral rho2 is 0. So V_H term vanishes?
            # Yes, for inelastic transition i!=f, the static potential term terms vanish by orthogonality
            # IF the states are orthogonal.
            # So strictly speaking, for excitation, V_diff correction is 0.
            # BUT, we might want to keep it general.
            # original code added V_diff.
            # V_diff = V_core - U_i = -V_H_i.
            # So we add < rho1 | -V_H_i | rho2 >.
            # If orthogonality holds, <rho2> = 0, so term is 0.
            # If not (e.g. non-orthogonal basis?), it matters.
            # I will implement it efficiently.
            
            # Correction = Integral[ rho1(r1) * V_diff(r1) ] * Integral[ rho2(r2) ]
            sum_rho2 = np.sum(rho2_dir) 
            corr_val = np.dot(rho1_dir, V_diff) * sum_rho2
            I_dir += corr_val

        I_L_dir[L] = float(I_dir)
        
        # --- Exchange Integral ---
        int_r2_ex = np.dot(kernel_L, rho2_ex)
        I_ex = np.dot(rho1_ex, int_r2_ex)
        
        I_L_exc[L] = float(I_ex)

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)
