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
- Uses direct kernel computation K_L = (r_</r_>)^L × (1/r_>) via exp(L·log(ratio))
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

# --- Oscillatory Integral Support ---
from oscillatory_integrals import (
    check_phase_sampling,
    log_phase_diagnostic,
    oscillatory_kernel_integral_2d,
    _analytical_dipole_tail,
    _analytical_multipole_tail,
)

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
    L_max: int,
    use_oscillatory_quadrature: bool = True
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
    oscillatory_integrals : Module with phase-adaptive quadrature methods.
    """
    if L_max < 0:
        raise ValueError("radial_ME_all_L: L_max must be >= 0.")

    u_i = bound_i.u_of_r
    u_f = bound_f.u_of_r
    
    # Extract wave parameters for oscillatory handling
    k_i = cont_i.k_au
    k_f = cont_f.k_au
    l_i = cont_i.l
    l_f = cont_f.l
    chi_i = cont_i.chi_of_r
    chi_f = cont_f.chi_of_r
    # High-Accuracy Optimization: Use Simpson's weights (O(h^4)) suitable for oscillatory functions
    w = grid.w_simpson 
    r = grid.r

    # ==========================================================================
    # SPLIT INTEGRAL OPTIMIZATION
    # ==========================================================================
    # Use match points from ContinuumWave to limit integration range.
    # Beyond r_m, the wavefunction is essentially free (Bessel/Coulomb) and
    # oscillatory contributions largely cancel. We integrate only [0, r_m].
    # ==========================================================================
    
    # Determine effective integration limit from continuum wave match points
    N_grid = len(r)
    idx_limit = N_grid  # Default: use full grid
    
    # Use minimum of match points (where both waves are in asymptotic regime)
    if hasattr(cont_i, 'idx_match') and cont_i.idx_match > 0:
        idx_limit = min(idx_limit, cont_i.idx_match + 1)
    if hasattr(cont_f, 'idx_match') and cont_f.idx_match > 0:
        idx_limit = min(idx_limit, cont_f.idx_match + 1)
    
    # Physics-based validation: match point should be in reasonable range
    # If match point is in first 10% of grid, potential decays very fast → suspect
    # Only apply minimum if match would exclude too much of the wavefunction
    MIN_FRAC = 0.10  # At least 10% of grid for numerical stability
    if idx_limit < int(N_grid * MIN_FRAC):
        logger.warning(
            "Match point at %.1f%% of grid (idx=%d) is suspiciously early. "
            "Using 10%% minimum for stability.",
            100 * idx_limit / N_grid, idx_limit
        )
        idx_limit = int(N_grid * MIN_FRAC)

    # Apply limit by zeroing weights beyond idx_limit (efficient, no array slicing)
    w_limited = w.copy()
    w_limited[idx_limit:] = 0.0

    # --- Precompute densities with limited weights ---
    rho1_dir = w_limited * chi_f * chi_i
    rho2_dir = w_limited * u_f * u_i
    
    rho1_ex = w_limited * u_f * chi_i
    rho2_ex = w_limited * chi_f * u_i
    
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
        
    # Clamp ratio to <1 to prevent drift at large L; use log-power for stability
    ratio_clamped = np.minimum(ratio, 1.0 - 1e-12)
    log_ratio = np.log(ratio_clamped + 1e-300)
    
    # ==========================================================================
    # PHASE SAMPLING DIAGNOSTIC
    # ==========================================================================
    # Check if grid adequately samples oscillations and log warnings if not.
    # This helps diagnose issues with high partial waves.
    # ==========================================================================
    
    if use_oscillatory_quadrature:
        k_total = k_i + k_f
        max_phase, is_ok, prob_idx = check_phase_sampling(r[:idx_limit], k_total)
        if not is_ok and idx_limit > 10:
            log_phase_diagnostic(r[:idx_limit], k_i, k_f, l_i, l_f)
    
    # Get phase shifts for analytical tail (if available)
    delta_i = cont_i.phase_shift if hasattr(cont_i, 'phase_shift') else 0.0
    delta_f = cont_f.phase_shift if hasattr(cont_f, 'phase_shift') else 0.0
    
    # Get Coulomb parameters for ionic targets
    eta_i = cont_i.eta if hasattr(cont_i, 'eta') else 0.0
    eta_f = cont_f.eta if hasattr(cont_f, 'eta') else 0.0
    sigma_i = cont_i.sigma_l if hasattr(cont_i, 'sigma_l') else 0.0
    sigma_f = cont_f.sigma_l if hasattr(cont_f, 'sigma_l') else 0.0
    
    r_match = r[idx_limit - 1] if idx_limit > 0 else r[-1]
    
    for L in range(L_max + 1):
        if L == 0:
            kernel_L = inv_gtr
        else:
            kernel_L = inv_gtr * np.exp(L * log_ratio)
        
        # --- Direct Integral ---
        # I = < rho1 | Kernel | rho2 >
        # For direct: rho1 = chi_f*chi_i (oscillatory), rho2 = u_f*u_i (bound, smooth)
        
        if use_oscillatory_quadrature and k_total > 0.5:
            # Use Filon + Clenshaw-Curtis method per instruction:
            # "rozbij całkę na przedziały, na których faza robi stały przyrost,
            #  a na podprzedziałach użyj węzłów Clenshaw-Curtis"
            I_dir = oscillatory_kernel_integral_2d(
                rho1_dir, rho2_dir, kernel_L, r, k_i, k_f, idx_limit, method="filon"
            )
        else:
            # Standard fast method
            int_r2 = np.dot(kernel_L, rho2_dir)
            I_dir = np.dot(rho1_dir, int_r2)

        # L=0 correction from (V_core - U_i) term in A_0.
        # For orthogonal bound states, the overlap integral is ~0.
        if L == 0:
            sum_rho2 = np.sum(rho2_dir)
            if abs(sum_rho2) > overlap_tol:
                corr_val = np.dot(rho1_dir, V_diff) * sum_rho2
                I_dir += corr_val
        
        # --- Analytical Multipole Tail for L >= 1 ---
        # The multipole integrals have tail contributions from asymptotic region.
        # For L=1 (dipole), exact Si/Ci formula is used.
        # For L>1, asymptotic expansion with r^(-L) envelope decay.
        #
        # IMPORTANT: Tail approximation assumes bound_f is localized (r₂ << r_m).
        # This is ONLY valid for excitation (bound→bound), NOT for ionization.
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if L >= 1 and use_oscillatory_quadrature and idx_limit < N_grid - 10 and is_excitation:
            # FIX: Correct multipole moment is ∫ r^L × u_f × u_i dr, not ∫ u_f × u_i dr
            # The kernel for r1 >> r2 is: (r2)^L / (r1)^(L+1)
            # So the tail integral involves r2^L weighted overlap
            moment_L = np.sum(w_limited * (r ** L) * u_f * u_i)
            
            if abs(moment_L) > 1e-12:
                tail_contrib = _analytical_multipole_tail(
                    r_match, k_i, k_f, delta_i, delta_f, l_i, l_f, L, moment_L,
                    eta_i, eta_f, sigma_i, sigma_f
                )
                I_dir += tail_contrib

        I_L_dir[L] = float(I_dir)
        
        # --- Exchange Integral ---
        # For exchange: rho1 = u_f*chi_i, rho2 = chi_f*u_i
        # BOTH densities contain one continuum wave → oscillatory inner integral
        
        if use_oscillatory_quadrature and k_total > 0.5:
            # Use Filon + Clenshaw-Curtis for BOTH inner and outer integrals
            # This handles oscillations in exchange densities properly
            I_ex = oscillatory_kernel_integral_2d(
                rho1_ex, rho2_ex, kernel_L, r, k_i, k_f, idx_limit, method="filon_exchange"
            )
        else:
            int_r2_ex = np.dot(kernel_L, rho2_ex)
            I_ex = np.dot(rho1_ex, int_r2_ex)
        
        if L == 0:
            # Exchange correction term for (V_core - U_i) delta_{L,0}.
            sum_rho2_ex = np.sum(rho2_ex)
            if abs(sum_rho2_ex) > overlap_tol:
                corr_val_ex = np.dot(rho1_ex, V_diff) * sum_rho2_ex
                I_ex += corr_val_ex
        
        # NOTE: Exchange analytical tail removed.
        # Exchange integrals now use filon_exchange which handles oscillations
        # in both inner and outer integrals via Clenshaw-Curtis quadrature.
        
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
    L_max: int,
    use_oscillatory_quadrature: bool = True
) -> RadialDWBAIntegrals:
    """
    GPU Accelerated Version of radial_ME_all_L using CuPy.
    
    Uses GPU for broadcasting and heavy matrix-vector multiplications.
    Includes support for oscillatory quadrature improvements:
    - Match point domain splitting
    - Analytical multipole tail for L >= 1
    
    Parameters
    ----------
    grid : RadialGrid
        Radial grid with integration weights.
    V_core_array, U_i_array : np.ndarray
        Core and distorting potentials.
    bound_i, bound_f : BoundOrbital
        Initial and final target states.
    cont_i, cont_f : ContinuumWave
        Incident and scattered projectile waves.
    L_max : int
        Maximum multipole order.
    use_oscillatory_quadrature : bool
        Enable oscillatory improvements (default True).
        
    Returns
    -------
    RadialDWBAIntegrals
        Container with I_L_direct and I_L_exchange.
    """
    if not HAS_CUPY:
        raise RuntimeError("radial_ME_all_L_gpu called but cupy is not installed.")

    r = grid.r
    N_grid = len(r)
    
    # ==========================================================================
    # MATCH POINT DOMAIN SPLITTING (same as CPU version)
    # ==========================================================================
    idx_limit = N_grid
    if hasattr(cont_i, 'idx_match') and cont_i.idx_match > 0:
        idx_limit = min(idx_limit, cont_i.idx_match + 1)
    if hasattr(cont_f, 'idx_match') and cont_f.idx_match > 0:
        idx_limit = min(idx_limit, cont_f.idx_match + 1)
    
    # Physics-based validation (same as CPU)
    MIN_FRAC = 0.10
    if idx_limit < int(N_grid * MIN_FRAC):
        logger.warning(
            "GPU: Match point at %.1f%% of grid (idx=%d) is suspiciously early. "
            "Using 10%% minimum for stability.",
            100 * idx_limit / N_grid, idx_limit
        )
        idx_limit = int(N_grid * MIN_FRAC)
    
    # Extract wave parameters for analytical tail
    k_i = cont_i.k_au
    k_f = cont_f.k_au
    l_i = cont_i.l
    l_f = cont_f.l
    delta_i = cont_i.phase_shift if hasattr(cont_i, 'phase_shift') else 0.0
    delta_f = cont_f.phase_shift if hasattr(cont_f, 'phase_shift') else 0.0
    
    # Get Coulomb parameters for ionic targets
    eta_i = cont_i.eta if hasattr(cont_i, 'eta') else 0.0
    eta_f = cont_f.eta if hasattr(cont_f, 'eta') else 0.0
    sigma_i = cont_i.sigma_l if hasattr(cont_i, 'sigma_l') else 0.0
    sigma_f = cont_f.sigma_l if hasattr(cont_f, 'sigma_l') else 0.0
    
    r_match = r[idx_limit - 1] if idx_limit > 0 else r[-1]
    
    # 1. Transfer Data to GPU (limited to idx_limit for efficiency)
    r_lim = r[:idx_limit]
    r_gpu = cp.asarray(r_lim)
    w_lim = grid.w_simpson[:idx_limit]
    w_gpu = cp.asarray(w_lim)
    
    u_i_lim = bound_i.u_of_r[:idx_limit]
    u_f_lim = bound_f.u_of_r[:idx_limit]
    chi_i_lim = cont_i.chi_of_r[:idx_limit]
    chi_f_lim = cont_f.chi_of_r[:idx_limit]
    
    u_i_gpu = cp.asarray(u_i_lim)
    u_f_gpu = cp.asarray(u_f_lim)
    chi_i_gpu = cp.asarray(chi_i_lim)
    chi_f_gpu = cp.asarray(chi_f_lim)
    
    V_diff_lim = (V_core_array - U_i_array)[:idx_limit]
    V_diff_gpu = cp.asarray(V_diff_lim)

    # 2. Precompute Densities on GPU
    rho1_dir = w_gpu * chi_f_gpu * chi_i_gpu
    rho2_dir = w_gpu * u_f_gpu * u_i_gpu
    rho1_ex = w_gpu * u_f_gpu * chi_i_gpu
    rho2_ex = w_gpu * chi_f_gpu * u_i_gpu
    
    overlap_tol = 1e-12
    
    # NOTE: bound_overlap removed - now moment_L computed inside loop for each L

    # 3. Kernel Construction on GPU
    r_col = r_gpu[:, None]
    r_row = r_gpu[None, :]
    
    r_less = cp.minimum(r_col, r_row)
    r_gtr = cp.maximum(r_col, r_row)
    
    eps = 1e-30
    ratio = r_less / (r_gtr + eps)
    ratio = cp.minimum(ratio, 1.0 - 1e-12)
    inv_gtr = 1.0 / (r_gtr + eps)
    log_ratio = cp.log(ratio + eps)
    
    I_L_dir: Dict[int, float] = {}
    I_L_exc: Dict[int, float] = {}

    for L in range(L_max + 1):
        if L == 0:
            kernel_L = inv_gtr
        else:
            kernel_L = inv_gtr * cp.exp(L * log_ratio)
        
        # --- Direct Integral on GPU ---
        int_r2 = cp.dot(kernel_L, rho2_dir)
        I_dir = float(cp.dot(rho1_dir, int_r2))
        
        # L=0 correction
        if L == 0:
            sum_rho2 = float(cp.sum(rho2_dir))
            if abs(sum_rho2) > overlap_tol:
                corr_val = float(cp.dot(rho1_dir, V_diff_gpu)) * sum_rho2
                I_dir += corr_val
        
        # --- Analytical Multipole Tail for L >= 1 ---
        # IMPORTANT: Only valid for excitation (bound→bound), NOT for ionization
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if L >= 1 and use_oscillatory_quadrature and idx_limit < N_grid - 10 and is_excitation:
            # FIX: Correct multipole moment is ∫ r^L × u_f × u_i dr
            moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
            
            if abs(moment_L) > 1e-12:
                tail_contrib = _analytical_multipole_tail(
                    r_match, k_i, k_f, delta_i, delta_f, l_i, l_f, L, moment_L,
                    eta_i, eta_f, sigma_i, sigma_f
                )
                I_dir += tail_contrib
            
        I_L_dir[L] = I_dir
        
        # --- Exchange Integral on GPU ---
        # NOTE: GPU uses standard dot product for exchange, not filon_exchange.
        # CPU version uses filon_exchange which applies CC to both inner and outer
        # integrals. GPU implementation of this would require kernel interpolation
        # on GPU which adds significant complexity. For consistency, both versions
        # produce identical results when use_oscillatory_quadrature=False.
        int_r2_ex = cp.dot(kernel_L, rho2_ex)
        I_ex = float(cp.dot(rho1_ex, int_r2_ex))
        
        if L == 0:
            sum_rho2_ex = float(cp.sum(rho2_ex))
            if abs(sum_rho2_ex) > overlap_tol:
                corr_val_ex = float(cp.dot(rho1_ex, V_diff_gpu)) * sum_rho2_ex
                I_ex += corr_val_ex
        
        I_L_exc[L] = I_ex

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)

