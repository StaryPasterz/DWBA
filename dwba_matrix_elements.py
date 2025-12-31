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
    # New advanced oscillatory methods
    _kahan_sum_real,
    _kahan_sum_complex,
    dwba_outer_integral_1d,
    compute_outer_integral_oscillatory,
    compute_asymptotic_phase,
    compute_phase_derivative,
    compute_phase_second_derivative,
)

# Type for oscillatory method selection
from typing import Literal
OscillatoryMethod = Literal["legacy", "advanced", "full_split"]

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
    use_oscillatory_quadrature: bool = True,
    oscillatory_method: OscillatoryMethod = "advanced"
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
    use_oscillatory_quadrature : bool
        Enable oscillatory integral methods (default True).
    oscillatory_method : {"legacy", "advanced", "full_split"}
        Method for oscillatory integrals:
        - "legacy": Clenshaw-Curtis on phase-split segments (fastest)
        - "advanced": Legacy CC + Levin/Filon tail contribution (balanced)
        - "full_split": Pure I_in (standard quadrature) + I_out (oscillatory) 
                        separation per instruction (most accurate, slowest)
        Default is "advanced".
        
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
    
    # Physics-based validation: match point should be beyond turning points
    # and where potential is negligible. Classical turning point is at:
    #   r_turn ≈ max(l) / k  (where centrifugal barrier equals kinetic energy)
    # 
    # We need r_m > r_turn to be in the oscillatory region.
    k_eff = min(k_i, k_f) if min(k_i, k_f) > 0.1 else max(k_i, k_f)
    l_max_wave = max(l_i, l_f)
    
    if k_eff > 1e-3:
        r_turn = (l_max_wave + 0.5) / k_eff  # +0.5 for quantum correction
        idx_turn = np.searchsorted(r, r_turn)
        
        # Match point should be beyond turning point with safety margin
        MIN_IDX = max(idx_turn + 20, N_grid // 10)  # At least 20 points past turning
    else:
        # Very low energy: use 10% of grid as fallback
        MIN_IDX = N_grid // 10
    
    if idx_limit < MIN_IDX:
        logger.debug(
            "Match point idx=%d is before turning point (idx=%d). "
            "Extending to MIN_IDX=%d for accuracy.",
            idx_limit, idx_turn if 'idx_turn' in dir() else 0, MIN_IDX
        )
        idx_limit = MIN_IDX
    
    # ==========================================================================
    # ASYMPTOTIC VALIDATION
    # ==========================================================================
    # Check that |U(r_m)| / (k²/2) < threshold ensures match point is truly
    # in the asymptotic region where potential is negligible compared to kinetic.
    # Tightened threshold for better tail accuracy.
    # ==========================================================================
    ASYMPTOTIC_THRESHOLD = 0.05  # |U|/(k²/2) should be < 5%
    r_m_idx = max(0, idx_limit - 1)
    U_at_rm = abs(U_i_array[r_m_idx])
    kinetic_energy = 0.5 * k_i**2  # E_kin = k²/2 in atomic units
    
    if kinetic_energy > 1e-10:  # Avoid division by zero
        ratio = U_at_rm / kinetic_energy
        if ratio > ASYMPTOTIC_THRESHOLD:
            # Try to find a better match point further out
            for try_idx in range(idx_limit, min(idx_limit + 100, N_grid)):
                U_try = abs(U_i_array[try_idx])
                if U_try / kinetic_energy < ASYMPTOTIC_THRESHOLD:
                    idx_limit = try_idx + 1
                    break
            else:
                logger.warning(
                    "Match point r_m=%.2f a₀ may not be in asymptotic region: "
                    "|U(r_m)|/(k²/2) = %.2f > %.2f. Tail contribution may be inaccurate.",
                    r[r_m_idx], ratio, ASYMPTOTIC_THRESHOLD
                )

    # Apply limit by zeroing weights beyond idx_limit (efficient, no array slicing)
    w_limited = w.copy()
    w_limited[idx_limit:] = 0.0

    # --- Precompute densities ---
    # WEIGHTED versions (for standard dot product method - weights include dr)
    rho1_dir_w = w_limited * chi_f * chi_i
    rho2_dir_w = w_limited * u_f * u_i
    
    rho1_ex_w = w_limited * u_f * chi_i
    rho2_ex_w = w_limited * chi_f * u_i
    
    # UNWEIGHTED versions (for filon/CC methods - CC handles weighting internally)
    # FIX: Filon uses CC weights (b-a)/2, so we must NOT pre-multiply by Simpson weights
    rho1_dir_uw = chi_f * chi_i
    rho2_dir_uw = u_f * u_i
    
    rho1_ex_uw = u_f * chi_i
    rho2_ex_uw = chi_f * u_i
    
    # NOTE: Unweighted arrays are NOT zeroed here - filon methods slice to idx_limit
    # internally. The slice operation + CC weighting handles the integration bounds correctly.
    
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
        
        # Check if this is excitation (bound final state) vs ionization (continuum final)
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if use_oscillatory_quadrature and k_total > 0.5:
            if oscillatory_method == "legacy":
                # === LEGACY METHOD ===
                # Use Filon + Clenshaw-Curtis on phase-split segments
                I_dir = oscillatory_kernel_integral_2d(
                    rho1_dir_uw, rho2_dir_uw, kernel_L, r, k_i, k_f, idx_limit, method="filon", w_grid=w
                )
            
            elif oscillatory_method == "full_split":
                # === FULL_SPLIT METHOD (per instruction) ===
                # Complete separation for r1 oscillatory dimension:
                #   I_in [0, r_m]: CC oscillatory quadrature for r1, standard for r2
                #   I_out [r_m, ∞): pure Levin/Filon with sinA×sinB for asymptotic tail
                #
                # Per instruction: "rozbij całkę na przedziały, na których faza robi 
                # stały przyrost, a na podprzedziałach użyj węzłów Clenshaw-Curtis"
                #
                # Key difference from "advanced": 
                # - I_in uses ONLY inner region [0, r_m] for r1
                # - I_out is computed via pure Levin/Filon (not added to legacy)
                # - This is a true domain decomposition, not incremental correction
                
                # --- I_in: CC oscillatory for oscillatory r1, standard for bound r2 ---
                # r2 integral uses full grid (bound states localized)
                # r1 integral uses only [0, r_m] with CC oscillatory quadrature
                I_in = oscillatory_kernel_integral_2d(
                    rho1_dir_uw[:idx_limit], rho2_dir_uw, 
                    kernel_L[:idx_limit, :], r[:idx_limit], k_i, k_f, idx_limit, 
                    method="filon", w_grid=w
                )
                
                # --- I_out: Pure oscillatory [r_m, r_max] via Levin/Filon ---
                # For r1 > r_m, use asymptotic sinA×sinB = ½[cos(A-B) - cos(A+B)]
                I_out = 0.0
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    # Multipole moment from bound states
                    moment_L = np.sum(w_limited * (r ** L) * u_f * u_i)
                    
                    if abs(moment_L) > 1e-12:
                        def make_envelope(mL, Lval):
                            def envelope(r_val):
                                return mL / (r_val ** (Lval + 1)) if r_val > 1e-6 else 0.0
                            return envelope
                        
                        env_func = make_envelope(moment_L, L)
                        
                        I_out = dwba_outer_integral_1d(
                            env_func,
                            k_i, l_i, delta_i, eta_i, sigma_i,
                            k_f, l_f, delta_f, eta_f, sigma_f,
                            r_match, r[-1],
                            delta_phi=np.pi / 4
                        )
                
                I_dir = I_in + I_out
            
            else:  # "advanced" method
                # === ADVANCED METHOD ===
                # I = I_in + I_out
                # I_in [0, r_m]: Use same CC quadrature as legacy (already handles inner region)
                # I_out [r_m, r_max]: Add oscillatory tail via Levin/Filon
                
                # --- I_in: Use legacy CC quadrature for inner region ---
                I_in = oscillatory_kernel_integral_2d(
                    rho1_dir_uw, rho2_dir_uw, kernel_L, r, k_i, k_f, idx_limit, method="filon", w_grid=w
                )
                
                # --- I_out: Outer tail integral on [r_m, r_max] ---
                # For direct integrals: rho1 = χ_f×χ_i (oscillatory), rho2 = u_f×u_i (localized)
                # In the outer region (r > r_m), u_f×u_i → 0 (bound states are localized)
                # The tail contribution comes from χ_f×χ_i × ∫u_f×u_i×r^L dr₂ / r₁^(L+1)
                
                I_out = 0.0
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    # Multipole moment from bound states (localized to inner region)
                    moment_L = np.sum(w_limited * (r ** L) * u_f * u_i)
                    
                    if abs(moment_L) > 1e-12:
                        # Envelope for outer integral: moment_L / r^(L+1)
                        def envelope_L(r_val):
                            return moment_L / (r_val ** (L + 1)) if r_val > 1e-6 else 0.0
                        
                        # Use dwba_outer_integral_1d with sinA×sinB = ½[cos(A-B) - cos(A+B)]
                        I_out = dwba_outer_integral_1d(
                            envelope_L,
                            k_i, l_i, delta_i, eta_i, sigma_i,
                            k_f, l_f, delta_f, eta_f, sigma_f,
                            r_match, r[-1],
                            delta_phi=np.pi / 4
                        )
                
                I_dir = I_in + I_out
        else:
            # Standard fast method - uses WEIGHTED densities (Simpson weights)
            int_r2 = np.dot(kernel_L, rho2_dir_w)
            I_dir = np.dot(rho1_dir_w, int_r2)

        # L=0 correction from (V_core - U_i) term in A_0.
        # For orthogonal bound states, the overlap integral is ~0.
        if L == 0:
            sum_rho2 = np.sum(rho2_dir_w)  # Weighted for integral
            if abs(sum_rho2) > overlap_tol:
                corr_val = np.dot(rho1_dir_w, V_diff) * sum_rho2
                I_dir += corr_val
        
        # --- Analytical Multipole Tail (Legacy method only) ---
        # For advanced method, the tail is handled by dwba_outer_integral_1d above
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if oscillatory_method == "legacy":
            if L >= 1 and use_oscillatory_quadrature and idx_limit < N_grid - 10 and is_excitation:
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
        # NOTE: For exchange, even "advanced" method uses CC quadrature because
        # both densities are oscillatory - the 1D sinA×sinB method doesn't apply.
        
        if use_oscillatory_quadrature and k_total > 0.5:
            # Use Filon + Clenshaw-Curtis for BOTH inner and outer integrals
            # This handles oscillations in exchange densities properly
            I_ex = oscillatory_kernel_integral_2d(
                rho1_ex_uw, rho2_ex_uw, kernel_L, r, k_i, k_f, idx_limit, method="filon_exchange", w_grid=w
            )
        else:
            int_r2_ex = np.dot(kernel_L, rho2_ex_w)
            I_ex = np.dot(rho1_ex_w, int_r2_ex)
        
        if L == 0:
            # Exchange correction term for (V_core - U_i) delta_{L,0}.
            sum_rho2_ex = np.sum(rho2_ex_w)
            if abs(sum_rho2_ex) > overlap_tol:
                corr_val_ex = np.dot(rho1_ex_w, V_diff) * sum_rho2_ex
                I_ex += corr_val_ex
        
        I_L_exc[L] = float(I_ex)

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)


# =============================================================================
# GPU FILON QUADRATURE HELPERS
# =============================================================================
# These functions implement Filon-type phase-split Clenshaw-Curtis quadrature
# for oscillatory integrals on GPU using CuPy. They mirror the CPU implementation
# in oscillatory_integrals.py but use GPU-accelerated operations.
# =============================================================================

# Cached GPU CC weights (initialized on first use)
_GPU_CC_INITIALIZED = False
_GPU_CC_X_REF = None
_GPU_CC_W_REF = None

def _init_gpu_cc_weights():
    """Initialize GPU CC reference weights (called once on first use)."""
    global _GPU_CC_INITIALIZED, _GPU_CC_X_REF, _GPU_CC_W_REF
    if _GPU_CC_INITIALIZED:
        return
    
    # Compute on CPU, transfer to GPU (same as oscillatory_integrals.py)
    CC_N = 5
    theta_ref = np.pi * np.arange(CC_N) / (CC_N - 1)
    x_ref = np.cos(theta_ref)
    
    j_max = (CC_N - 1) // 2
    j_vals = np.arange(1, j_max + 1)
    
    if j_max > 0:
        j_col = j_vals[:, np.newaxis]
        theta_row = theta_ref[np.newaxis, :]
        cos_terms = np.cos(2 * j_col * theta_row)
        denom = 4 * j_vals**2 - 1
        weight_sums = np.sum(cos_terms / denom[:, np.newaxis], axis=0)
        if (CC_N - 1) % 2 == 0:
            j_final = j_max
            cos_final = np.cos(2 * j_final * theta_ref)
            weight_sums += 0.5 * cos_final / (4 * j_final**2 - 1)
    else:
        weight_sums = np.zeros(CC_N)
    
    w_ref = (2.0 / (CC_N - 1)) * (1 - 2 * weight_sums)
    
    _GPU_CC_X_REF = cp.asarray(x_ref)
    _GPU_CC_W_REF = cp.asarray(w_ref)
    _GPU_CC_INITIALIZED = True


def _gpu_filon_direct(rho1_uw, int_r2, r_gpu, w_gpu, k_total, phase_increment=np.pi/2):
    """
    GPU Filon quadrature for direct integral outer loop.
    
    Applies CC on phase-split intervals for the outer integral.
    Inner integral (kernel @ rho2 * w) should already be computed with weights.
    
    Parameters
    ----------
    rho1_uw : cp.ndarray
        Unweighted outer density (chi_f × chi_i).
    int_r2 : cp.ndarray
        Result of inner integral (kernel @ rho2 * w).
    r_gpu : cp.ndarray
        Radial grid on GPU.
    w_gpu : cp.ndarray
        Integration weights on GPU.
    k_total : float
        k_i + k_f for phase calculation.
    phase_increment : float
        Phase change per interval (default π/2).
        
    Returns
    -------
    result : float
        Integral value.
    """
    _init_gpu_cc_weights()
    
    if k_total < 1e-6:
        # Non-oscillatory: standard dot with weights for r₁
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    r_start = float(r_gpu[0])
    r_end = float(r_gpu[-1])
    
    # Generate phase nodes on CPU
    dr = phase_increment / k_total
    n_intervals = max(1, int(np.ceil((r_end - r_start) / dr)))
    phase_nodes = np.linspace(r_start, r_end, n_intervals + 1)
    
    # Skip if too few intervals
    if n_intervals < 2:
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    # Get interval bounds
    a_arr = phase_nodes[:-1]
    b_arr = phase_nodes[1:]
    valid_mask = (b_arr - a_arr) > 1e-12
    a_valid = a_arr[valid_mask]
    b_valid = b_arr[valid_mask]
    n_valid = len(a_valid)
    
    if n_valid == 0:
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    # Compute CC nodes for all intervals
    # r_cc = 0.5 * (b - a) * (x_ref + 1) + a
    half_width = 0.5 * (b_valid - a_valid)
    
    # Transfer to GPU
    a_gpu = cp.asarray(a_valid)
    b_gpu = cp.asarray(b_valid)
    half_width_gpu = cp.asarray(half_width)
    
    # All CC points: shape (n_valid, CC_N)
    all_r = half_width_gpu[:, None] * (_GPU_CC_X_REF + 1) + a_gpu[:, None]
    all_r_flat = all_r.ravel()
    
    # OPTIMIZED: Pure GPU interpolation (no CPU transfers)
    rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, 5)
    int_r2_cc = cp.interp(all_r_flat, r_gpu, int_r2).reshape(n_valid, 5)
    
    # Compute integrand
    integrand = rho1_cc * int_r2_cc
    
    # Weights: w_ref * (b-a)/2
    weights_scaled = _GPU_CC_W_REF * half_width_gpu[:, None]
    
    # Sum
    result = float(cp.sum(integrand * weights_scaled))
    
    return result


def _gpu_filon_exchange(kernel_L, rho1_uw, rho2_uw, r_gpu, w_gpu, k_total, phase_increment=np.pi/2):
    """
    GPU Filon quadrature for exchange integral (CC on both inner and outer).
    
    For exchange, both densities contain oscillatory components, so we apply
    CC quadrature to both the inner (over r2) and outer (over r1) integrals.
    
    Parameters
    ----------
    kernel_L : cp.ndarray
        2D kernel matrix K(r1, r2).
    rho1_uw, rho2_uw : cp.ndarray
        Unweighted densities.
    r_gpu : cp.ndarray
        Radial grid on GPU.
    w_gpu : cp.ndarray
        Integration weights on GPU.
    k_total : float
        k_i + k_f for phase calculation.
    phase_increment : float
        Phase change per interval.
        
    Returns
    -------
    result : float
        Exchange integral value.
    """
    _init_gpu_cc_weights()
    
    n_r = len(r_gpu)
    
    if k_total < 1e-6:
        # Non-oscillatory: standard method with weights
        int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    r_start = float(r_gpu[0])
    r_end = float(r_gpu[-1])
    
    # Generate phase nodes
    dr = phase_increment / k_total
    n_intervals = max(1, int(np.ceil((r_end - r_start) / dr)))
    phase_nodes = np.linspace(r_start, r_end, n_intervals + 1)
    
    if n_intervals < 2:
        int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    a_arr = phase_nodes[:-1]
    b_arr = phase_nodes[1:]
    valid_mask = (b_arr - a_arr) > 1e-12
    a_valid = a_arr[valid_mask]
    b_valid = b_arr[valid_mask]
    n_valid = len(a_valid)
    
    if n_valid == 0:
        int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    half_width = 0.5 * (b_valid - a_valid)
    half_width_gpu = cp.asarray(half_width)
    a_gpu_arr = cp.asarray(a_valid)
    
    all_r = half_width_gpu[:, None] * (_GPU_CC_X_REF + 1) + a_gpu_arr[:, None]
    all_r_flat = all_r.ravel()
    
    # OPTIMIZED: Pure GPU interpolation for rho2
    rho2_cc = cp.interp(all_r_flat, r_gpu, rho2_uw).reshape(n_valid, 5)
    
    # OPTIMIZED: GPU kernel interpolation using searchsorted
    idx_right = cp.searchsorted(r_gpu, all_r_flat)
    idx_right = cp.clip(idx_right, 1, n_r - 1)
    idx_left = idx_right - 1
    
    r_left = r_gpu[idx_left]
    r_right = r_gpu[idx_right]
    weight_right = (all_r_flat - r_left) / (r_right - r_left + 1e-30)
    weight_left = 1.0 - weight_right
    
    # kernel_at_cc: shape (n_r, n_cc_total)
    kernel_at_cc = kernel_L[:, idx_left] * weight_left + kernel_L[:, idx_right] * weight_right
    kernel_interp = kernel_at_cc.reshape(n_r, n_valid, 5)
    
    # Inner integral: for each r1, sum over CC nodes with weights
    inner_integrand = kernel_interp * rho2_cc[None, :, :]
    
    # Weights for inner: w_ref * half_width
    inner_weights = _GPU_CC_W_REF * half_width_gpu[:, None]
    
    # Sum inner: (n_r,)
    int_r2_cc = cp.sum(inner_integrand * inner_weights[None, :, :], axis=(1, 2))
    
    # OPTIMIZED: Pure GPU interpolation for outer integral
    rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, 5)
    int_r2_outer = cp.interp(all_r_flat, r_gpu, int_r2_cc).reshape(n_valid, 5)
    
    outer_integrand = rho1_cc * int_r2_outer
    result = float(cp.sum(outer_integrand * inner_weights))
    
    return result


def radial_ME_all_L_gpu(
    grid: RadialGrid,
    V_core_array: np.ndarray,
    U_i_array: np.ndarray,
    bound_i: BoundOrbital,
    bound_f: Union[BoundOrbital, ContinuumWave],
    cont_i: ContinuumWave,
    cont_f: ContinuumWave,
    L_max: int,
    use_oscillatory_quadrature: bool = True,
    oscillatory_method: OscillatoryMethod = "advanced"
) -> RadialDWBAIntegrals:
    """
    GPU Accelerated Version of radial_ME_all_L using CuPy.
    
    Uses GPU for broadcasting and heavy matrix-vector multiplications.
    Includes support for oscillatory quadrature improvements:
    - Match point domain splitting
    - Analytical multipole tail for L >= 1
    - Asymptotic region validation
    - **Filon/CC quadrature** for oscillatory integrals
    - **Advanced methods**: Levin/Filon with sinA×sinB decomposition (GPU accelerated)
    
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
        Enable oscillatory improvements including Filon/CC (default True).
    oscillatory_method : {"legacy", "advanced", "full_split"}
        Method for oscillatory integrals (same as CPU version).
        
    Returns
    -------
    RadialDWBAIntegrals
        Container with I_L_direct and I_L_exchange.
    """
    if not HAS_CUPY:
        raise RuntimeError("radial_ME_all_L_gpu called but cupy is not installed.")
    
    # Note: GPU now implements Filon/CC for oscillatory integrals

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
    
    # Physics-based validation (same as CPU) - use turning point
    k_i = cont_i.k_au
    k_f = cont_f.k_au
    l_i = cont_i.l
    l_f = cont_f.l
    
    k_eff = min(k_i, k_f) if min(k_i, k_f) > 0.1 else max(k_i, k_f)
    l_max_wave = max(l_i, l_f)
    
    if k_eff > 1e-3:
        r_turn = (l_max_wave + 0.5) / k_eff
        idx_turn = np.searchsorted(r, r_turn)
        MIN_IDX = max(idx_turn + 20, N_grid // 10)
    else:
        MIN_IDX = N_grid // 10
    
    if idx_limit < MIN_IDX:
        logger.debug(
            "GPU: Match point idx=%d is before turning point. Extending to MIN_IDX=%d.",
            idx_limit, MIN_IDX
        )
        idx_limit = MIN_IDX
    
    # ==========================================================================
    # ASYMPTOTIC VALIDATION (same as CPU version)
    # ==========================================================================
    ASYMPTOTIC_THRESHOLD = 0.05  # Tightened to 5%
    r_m_idx = max(0, idx_limit - 1)
    U_at_rm = abs(U_i_array[r_m_idx])
    kinetic_energy = 0.5 * k_i**2
    
    if kinetic_energy > 1e-10:
        ratio_asymp = U_at_rm / kinetic_energy
        if ratio_asymp > ASYMPTOTIC_THRESHOLD:
            # Try to find better match point
            for try_idx in range(idx_limit, min(idx_limit + 100, N_grid)):
                if abs(U_i_array[try_idx]) / kinetic_energy < ASYMPTOTIC_THRESHOLD:
                    idx_limit = try_idx + 1
                    break
            else:
                logger.warning(
                    "GPU: Match point r_m=%.2f a₀ may not be in asymptotic region: "
                    "|U(r_m)|/(k²/2) = %.2f > %.2f. Tail may be inaccurate.",
                    r[r_m_idx], ratio_asymp, ASYMPTOTIC_THRESHOLD
                )
    
    # Extract phase shift parameters for analytical tail (wave params already defined above)
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
    # Weighted (for standard method / L=0 correction)
    rho1_dir_w = w_gpu * chi_f_gpu * chi_i_gpu
    rho2_dir_w = w_gpu * u_f_gpu * u_i_gpu
    rho1_ex_w = w_gpu * u_f_gpu * chi_i_gpu
    rho2_ex_w = w_gpu * chi_f_gpu * u_i_gpu
    
    # Unweighted (for Filon/CC method)
    rho1_dir_uw = chi_f_gpu * chi_i_gpu
    rho2_dir_uw = u_f_gpu * u_i_gpu
    rho1_ex_uw = u_f_gpu * chi_i_gpu
    rho2_ex_uw = chi_f_gpu * u_i_gpu
    
    overlap_tol = 1e-12
    k_total = k_i + k_f

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
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if use_oscillatory_quadrature and k_total > 0.5:
            if oscillatory_method == "legacy":
                # === LEGACY METHOD ===
                # Use GPU Filon + CC - inner integral uses weights for r₂
                int_r2_dir = cp.dot(kernel_L, rho2_dir_uw * w_gpu)
                I_dir = _gpu_filon_direct(rho1_dir_uw, int_r2_dir, r_gpu, w_gpu, k_total)
                
                # Add analytical tail for L >= 1 (legacy only)
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
                    if abs(moment_L) > 1e-12:
                        tail_contrib = _analytical_multipole_tail(
                            r_match, k_i, k_f, delta_i, delta_f, l_i, l_f, L, moment_L,
                            eta_i, eta_f, sigma_i, sigma_f
                        )
                        I_dir += tail_contrib
            
            elif oscillatory_method == "full_split":
                # === FULL_SPLIT METHOD (GPU optimized) ===
                # Same as CPU: CC oscillatory for I_in [0, r_m] + Levin/Filon for I_out
                # 
                # Per instruction: "rozbij całkę na przedziały, na których faza robi 
                # stały przyrost, a na podprzedziałach użyj węzłów Clenshaw-Curtis"
                
                # I_in: GPU CC quadrature with weights for r₂
                int_r2_dir = cp.dot(kernel_L, rho2_dir_uw * w_gpu)
                I_in = _gpu_filon_direct(rho1_dir_uw, int_r2_dir, r_gpu, w_gpu, k_total)
                
                # I_out: CPU Levin/Filon for oscillatory tail (same as CPU)
                I_out = 0.0
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
                    if abs(moment_L) > 1e-12:
                        def make_envelope(mL, Lval):
                            def envelope(r_val):
                                return mL / (r_val ** (Lval + 1)) if r_val > 1e-6 else 0.0
                            return envelope
                        
                        env_func = make_envelope(moment_L, L)
                        
                        # CPU Levin/Filon with sinA×sinB decomposition
                        I_out = dwba_outer_integral_1d(
                            env_func,
                            k_i, l_i, delta_i, eta_i, sigma_i,
                            k_f, l_f, delta_f, eta_f, sigma_f,
                            r_match, float(r_gpu[-1].get()),
                            delta_phi=np.pi / 4
                        )
                
                I_dir = I_in + I_out
            
            else:  # advanced
                # === ADVANCED METHOD ===
                # GPU CC for inner region with weights for r₂
                int_r2_dir = cp.dot(kernel_L, rho2_dir_uw * w_gpu)
                I_in = _gpu_filon_direct(rho1_dir_uw, int_r2_dir, r_gpu, w_gpu, k_total)
                
                # Outer tail via CPU (Levin/Filon more accurate)
                I_out = 0.0
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
                    if abs(moment_L) > 1e-12:
                        def make_envelope(mL, Lval):
                            def envelope(r_val):
                                return mL / (r_val ** (Lval + 1)) if r_val > 1e-6 else 0.0
                            return envelope
                        
                        env_func = make_envelope(moment_L, L)
                        
                        I_out = dwba_outer_integral_1d(
                            env_func,
                            k_i, l_i, delta_i, eta_i, sigma_i,
                            k_f, l_f, delta_f, eta_f, sigma_f,
                            r_match, float(r_gpu[-1].get()),
                            delta_phi=np.pi / 4
                        )
                
                I_dir = I_in + I_out
        else:
            # Standard method (weighted densities)
            int_r2 = cp.dot(kernel_L, rho2_dir_w)
            I_dir = float(cp.dot(rho1_dir_w, int_r2))
        
        # L=0 correction
        if L == 0:
            sum_rho2 = float(cp.sum(rho2_dir_w))
            if abs(sum_rho2) > overlap_tol:
                corr_val = float(cp.dot(rho1_dir_w, V_diff_gpu)) * sum_rho2
                I_dir += corr_val
            
        I_L_dir[L] = I_dir
        
        # --- Exchange Integral on GPU ---
        if use_oscillatory_quadrature and k_total > 0.5:
            # Use GPU Filon + CC for exchange (both inner and outer)
            I_ex = _gpu_filon_exchange(kernel_L, rho1_ex_uw, rho2_ex_uw, r_gpu, w_gpu, k_total)
        else:
            # Standard method
            int_r2_ex = cp.dot(kernel_L, rho2_ex_w)
            I_ex = float(cp.dot(rho1_ex_w, int_r2_ex))
        
        if L == 0:
            sum_rho2_ex = float(cp.sum(rho2_ex_w))
            if abs(sum_rho2_ex) > overlap_tol:
                corr_val_ex = float(cp.dot(rho1_ex_w, V_diff_gpu)) * sum_rho2_ex
                I_ex += corr_val_ex
        
        I_L_exc[L] = I_ex

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)

