# dwba_matrix_elements.py
"""
Radial DWBA Matrix Elements
===========================

Computes radial integrals I_L for the T-matrix amplitudes in DWBA calculations.

Integral Types
--------------
1. **Direct (I_L^D)**:
   ∫∫ dr₁ dr₂ χ_f(r₁) χ_i(r₁) · [r_<^L / r_>^(L+1)] · u_f(r₂) u_i(r₂)
   
2. **Exchange (I_L^E)**:
   ∫∫ dr₁ dr₂ u_f(r₁) χ_i(r₁) · [r_<^L / r_>^(L+1)] · χ_f(r₂) u_i(r₂)

L=0 Correction
--------------
For L=0 monopole, a correction term is added:
    [V_core - U_i] × ∫χ_f·χ_i × ∫u_f·u_i

For excitation: ∫u_f·u_i = 0 (orthogonality) → correction vanishes
For ionization: ∫χ_eject·u_i ≠ 0 → correction contributes

Implementation
--------------
- Uses kernel recurrence K_L = K_{L-1}·(r_</r_>) for efficiency
- GPU acceleration via CuPy when available
- Angular factors applied separately in dwba_coupling.py

Units
-----
All inputs/outputs in Hartree atomic units (a₀, Ha).
"""



from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from grid import RadialGrid
from bound_states import BoundOrbital
from continuum import ContinuumWave

# --- GPU Acceleration Support ---
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

def check_cupy_runtime() -> bool:
    """
    Verifies if CuPy can actually run on this system.
    Detects issues like missing NVRTC DLLs even if import succeeds.
    """
    if not HAS_CUPY:
        return False
    try:
        # Attempt a minimal operation that requires compilation/backend
        x = cp.array([1.0])
        y = x * 2.0
        _ = y.get()
        return True
    except Exception as e:
        print(f"[Warning] GPU initialization failed: {e}")
        return False

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
    Compute radial DWBA integrals I_L for multipoles L = 0 to L_max.
    
    This function evaluates the radial matrix elements that appear in the
    T-matrix for electron-impact excitation and ionization. Both direct
    and exchange terms are computed.
    
    Theory
    ------
    The Coulomb interaction 1/|r₁ - r₂| is expanded in multipoles:
    
        1/|r₁ - r₂| = Σ_L (r_<^L / r_>^{L+1}) P_L(cos θ₁₂)
    
    The radial integrals extract the radial part:
    
    Direct (I_L^D):
        I_L^D = ∫∫ χ_f(r₁)χ_i(r₁) × (r_<^L/r_>^{L+1}) × u_f(r₂)u_i(r₂) dr₁dr₂
    
    Exchange (I_L^E):
        I_L^E = ∫∫ u_f(r₁)χ_i(r₁) × (r_<^L/r_>^{L+1}) × χ_f(r₂)u_i(r₂) dr₁dr₂
    
    Parameters
    ----------
    grid : RadialGrid
        Radial grid with integration weights.
    V_core_array : np.ndarray
        Core potential V_A+(r) on the grid, in Hartree.
    U_i_array : np.ndarray
        Distorting potential U_i(r) for incident channel, in Hartree.
    bound_i : BoundOrbital
        Initial target bound state u_i(r).
    bound_f : BoundOrbital or ContinuumWave
        Final target state u_f(r). For ionization, this is a ContinuumWave.
    cont_i : ContinuumWave
        Incident projectile wave χ_i(r).
    cont_f : ContinuumWave
        Scattered projectile wave χ_f(r).
    L_max : int
        Maximum multipole order to compute (typically 10-20).
        
    Returns
    -------
    RadialDWBAIntegrals
        Container with I_L_direct[L] and I_L_exchange[L] for L = 0..L_max.
        
    Notes
    -----
    Optimization: Uses kernel recurrence K_L = K_{L-1} × (r_</r_>) to avoid
    O(N²) power operations at each L. Total complexity is O(L_max × N²).
    
    For L=0, a correction term involving (V_core - U_i) is added to account
    for orthogonality contributions.
    
    See Also
    --------
    radial_ME_all_L_gpu : GPU-accelerated version using CuPy.
    dwba_coupling.calculate_amplitude_contribution : Uses these integrals.
    """
    if L_max < 0:
        raise ValueError("radial_ME_all_L: L_max must be >= 0.")

    u_i = bound_i.u_of_r
    u_f = bound_f.u_of_r
    chi_i = cont_i.chi_of_r
    chi_f = cont_f.chi_of_r
    # High-Accuracy Optimization: Use Simpson's weights (O(h^4)) suitable for oscillatory functions
    w = grid.w_simpson 
    r = grid.r


    # --- Precompute densities ---
    rho1_dir = w * chi_f * chi_i
    rho2_dir = w * u_f * u_i
    
    rho1_ex = w * u_f * chi_i
    rho2_ex = w * chi_f * u_i
    
    # Correction term for L=0: [V_core(r1) - U_i(r1)] from A_0 in the article.
    V_diff = V_core_array - U_i_array
    overlap_tol = 1e-12

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

        # L=0 correction from (V_core - U_i) term in A_0.
        # For orthogonal bound states, the overlap integral is ~0.
        if L == 0:
            sum_rho2 = np.sum(rho2_dir)
            if abs(sum_rho2) > overlap_tol:
                corr_val = np.dot(rho1_dir, V_diff) * sum_rho2
                I_dir += corr_val

        I_L_dir[L] = float(I_dir)
        
        # --- Exchange Integral ---
        int_r2_ex = np.dot(kernel_L, rho2_ex)
        I_ex = np.dot(rho1_ex, int_r2_ex)
        
        if L == 0:
            # Exchange correction term for (V_core - U_i) delta_{L,0}.
            sum_rho2_ex = np.sum(rho2_ex)
            if abs(sum_rho2_ex) > overlap_tol:
                corr_val_ex = np.dot(rho1_ex, V_diff) * sum_rho2_ex
                I_ex += corr_val_ex
        
        I_L_exc[L] = float(I_ex)

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)


def radial_ME_all_L_gpu(
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
    GPU Accelerated Version of radial_ME_all_L using CuPy.
    Uses GPU for broadcasting and heavy matrix-vector multiplications.
    """
    if not HAS_CUPY:
        raise RuntimeError("radial_ME_all_L_gpu called but cupy is not installed.")

    # 1. Transfer Data to GPU
    # Arrays are 1D (size N). Transfer is fast (few KB).
    r_gpu = cp.asarray(grid.r)
    # High-Accuracy Optimization: Use Simpson's weights (O(h^4))
    w_gpu = cp.asarray(grid.w_simpson)
    
    u_i_gpu = cp.asarray(bound_i.u_of_r)
    u_f_gpu = cp.asarray(bound_f.u_of_r)
    chi_i_gpu = cp.asarray(cont_i.chi_of_r)
    chi_f_gpu = cp.asarray(cont_f.chi_of_r)
    
    V_core_gpu = cp.asarray(V_core_array)
    U_i_gpu = cp.asarray(U_i_array)

    # 2. Precompute Densities on GPU
    rho1_dir = w_gpu * chi_f_gpu * chi_i_gpu
    rho2_dir = w_gpu * u_f_gpu * u_i_gpu
    rho1_ex = w_gpu * u_f_gpu * chi_i_gpu
    rho2_ex = w_gpu * chi_f_gpu * u_i_gpu
    
    V_diff = V_core_gpu - U_i_gpu
    overlap_tol = 1e-12

    # 3. Kernel Construction on GPU
    # Broadcasting to create N x N matrices
    r_col = r_gpu[:, None]
    r_row = r_gpu[None, :]
    
    r_less = cp.minimum(r_col, r_row)
    r_gtr  = cp.maximum(r_col, r_row)
    
    # Avoid div/0
    # On GPU we can set eps or mask
    eps = 1e-30
    ratio = r_less / (r_gtr + eps)
    # Correct for diagonal r=0 case if r_gtr=0? 
    # Usually grid.r[0] > 0. If 0, handle?
    # grid.r starts at r_min > 0 usually.
    
    kernel_L = 1.0 / (r_gtr + eps)
    
    I_L_dir: Dict[int, float] = {}
    I_L_exc: Dict[int, float] = {}

    for L in range(L_max + 1):
        if L > 0:
            kernel_L *= ratio
        
        # Direct Integral: < rho1 | K | rho2 >
        # cp.dot(matrix, vector)
        int_r2 = cp.dot(kernel_L, rho2_dir)
        I_dir = cp.dot(rho1_dir, int_r2)
        
        # L=0 correction from (V_core - U_i) term in A_0.
        if L == 0:
            sum_rho2 = cp.sum(rho2_dir)
            if float(cp.abs(sum_rho2)) > overlap_tol:
                corr_val = cp.dot(rho1_dir, V_diff) * sum_rho2
                I_dir += corr_val
            
        # Exchange Integral
        int_r2_ex = cp.dot(kernel_L, rho2_ex)
        I_ex = cp.dot(rho1_ex, int_r2_ex)
        
        if L == 0:
            # Exchange correction term for (V_core - U_i) delta_{L,0}.
            sum_rho2_ex = cp.sum(rho2_ex)
            if float(cp.abs(sum_rho2_ex)) > overlap_tol:
                corr_val_ex = cp.dot(rho1_ex, V_diff) * sum_rho2_ex
                I_ex += corr_val_ex
        
        # Sync Scalar Results to CPU
        I_L_dir[L] = float(I_dir) # Explicit cast triggers sync
        I_L_exc[L] = float(I_ex)

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)
