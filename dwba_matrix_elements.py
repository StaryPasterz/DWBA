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
import gc
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
    oscillatory_method: OscillatoryMethod = "advanced",
    # Pass config or defaults
    CC_nodes: int = 5,
    phase_increment: float = 1.5708
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
                    method="filon", w_grid=w,
                    n_nodes=CC_nodes, phase_increment=phase_increment
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
                    rho1_dir_uw, rho2_dir_uw, kernel_L, r, k_i, k_f, idx_limit, method="filon", w_grid=w,
                    n_nodes=CC_nodes, phase_increment=phase_increment
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
_GPU_CC_N = 0
_GPU_CC_X_REF = None
_GPU_CC_W_REF = None

def _init_gpu_cc_weights(n_nodes: int = 5):
    """Initialize GPU CC reference weights (called on first use or if N changes)."""
    global _GPU_CC_N, _GPU_CC_X_REF, _GPU_CC_W_REF
    if _GPU_CC_N == n_nodes and _GPU_CC_X_REF is not None:
        return
    
    # Compute on CPU, transfer to GPU (same as oscillatory_integrals.py)
    CC_N = n_nodes
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
    _GPU_CC_N = CC_N


def _generate_gpu_filon_params(r_gpu, k_total, phase_increment=1.5708, n_nodes=5):
    """
    Precompute grid-dependent parameters for GPU Filon quadrature.
    """
    _init_gpu_cc_weights(n_nodes)

    if k_total < 1e-6:
        return None

    r_start = float(r_gpu[0])
    r_end = float(r_gpu[-1])

    dr = phase_increment / k_total
    n_intervals = max(1, int(np.ceil((r_end - r_start) / dr)))
    phase_nodes = np.linspace(r_start, r_end, n_intervals + 1)
    
    if n_intervals < 2:
        return None
        
    a_arr = phase_nodes[:-1]
    b_arr = phase_nodes[1:]
    valid_mask = (b_arr - a_arr) > 1e-12
    a_valid = a_arr[valid_mask]
    b_valid = b_arr[valid_mask]
    n_valid = len(a_valid)
    
    if n_valid == 0:
        return None

    half_width = 0.5 * (b_valid - a_valid)
    
    # Transfer to GPU
    a_gpu = cp.asarray(a_valid)
    # b_gpu unused except for half_width calculation which is done on CPU
    half_width_gpu = cp.asarray(half_width)
    
    # All CC points: shape (n_valid, CC_N)
    all_r = half_width_gpu[:, None] * (_GPU_CC_X_REF + 1) + a_gpu[:, None]
    all_r_flat = all_r.ravel()
    
    # Weights
    weights_scaled = _GPU_CC_W_REF * half_width_gpu[:, None]
    
    # Precompute interpolation indices
    n_r = len(r_gpu)
    idx_right = cp.searchsorted(r_gpu, all_r_flat)
    idx_right = cp.clip(idx_right, 1, n_r - 1)
    
    # Pre-calculate interpolation weights for exchange kernels
    idx_left = idx_right - 1
    r_left = r_gpu[idx_left]
    r_right = r_gpu[idx_right]
    weight_right = (all_r_flat - r_left) / (r_right - r_left + 1e-30)
    weight_left = 1.0 - weight_right
    
    return {
        'n_valid': n_valid,
        'all_r_flat': all_r_flat,
        'weights_scaled': weights_scaled,
        'idx_left': idx_left,
        'idx_right': idx_right,
        'w_left': weight_left,
        'w_right': weight_right,
        'n_nodes': n_nodes
    }

def _gpu_filon_direct(rho1_uw, int_r2, r_gpu, w_gpu, k_total, phase_increment=1.5708, n_nodes=5, precomputed=None):
    """
    GPU Filon quadrature for direct integral outer loop.
    """
    if k_total < 1e-6:
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    if precomputed is None:
        params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, n_nodes)
        if params is None:
             return float(cp.dot(rho1_uw * w_gpu, int_r2))
    else:
        params = precomputed

    n_valid = params['n_valid']
    all_r_flat = params['all_r_flat']
    weights_scaled = params['weights_scaled']
    n_nodes = params['n_nodes']
    # idx_right = params['idx_right'] # unused in direct if using cp.interp, but can be used for optimized interp
    
    # OPTIMIZED: Pure GPU interpolation
    # Note: cp.interp is slow? We can use precomputed indices if we implement custom lerp
    rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, n_nodes)
    int_r2_cc = cp.interp(all_r_flat, r_gpu, int_r2).reshape(n_valid, n_nodes)
    
    integrand = rho1_cc * int_r2_cc
    
    # Sum
    result = float(cp.sum(integrand * weights_scaled))
    return result



def _gpu_filon_exchange(kernel_L, rho1_uw, rho2_uw, r_gpu, w_gpu, k_total, phase_increment=1.5708, n_nodes=5, precomputed=None):
    """
    GPU Filon quadrature for exchange integral (CC on both inner and outer).
    """
    if k_total < 1e-6:
        int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
        return float(cp.dot(rho1_uw * w_gpu, int_r2))
    
    if precomputed is None:
        params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, n_nodes)
        if params is None:
             int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
             return float(cp.dot(rho1_uw * w_gpu, int_r2))
    else:
        params = precomputed
        
    n_valid = params['n_valid']
    all_r_flat = params['all_r_flat']
    weights_scaled = params['weights_scaled']
    idx_left = params['idx_left']
    idx_right = params['idx_right']
    weight_left = params['w_left']
    weight_right = params['w_right']
    n_nodes = params['n_nodes']
    
    n_r = len(r_gpu)
    
    # Kernel interpolation: reconstruct kernel at CC nodes
    # We do this from either full kernel_L or sliced components
    if isinstance(kernel_L, dict):
        # Optimized path: use pre-sliced components
        inv_gtr_at_cc = kernel_L['inv_gtr_at_cc']
        log_ratio_at_cc = kernel_L['log_ratio_at_cc']
        L = kernel_L['L']
        if L == 0:
            kernel_at_cc = inv_gtr_at_cc
        else:
            kernel_at_cc = inv_gtr_at_cc * cp.exp(L * log_ratio_at_cc)
    else:
        # Standard path: slice from full matrix
        kernel_at_cc = kernel_L[:, idx_left] * weight_left + kernel_L[:, idx_right] * weight_right
    kernel_interp = kernel_at_cc.reshape(n_r, n_valid, n_nodes)
    # OPTIMIZED: Pure GPU interpolation for rho2
    rho2_cc = cp.interp(all_r_flat, r_gpu, rho2_uw).reshape(n_valid, n_nodes)


    # Inner integral: for each r1, sum over CC nodes with weights
    inner_integrand = kernel_interp * rho2_cc[None, :, :]
    
    # Sum inner: (n_r,)
    int_r2_cc = cp.sum(inner_integrand * weights_scaled[None, :, :], axis=(1, 2))
    
    # OPTIMIZED: Pure GPU interpolation for outer integral
    rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, n_nodes)
    int_r2_outer = cp.interp(all_r_flat, r_gpu, int_r2_cc).reshape(n_valid, n_nodes)
    
    outer_integrand = rho1_cc * int_r2_outer
    result = float(cp.sum(outer_integrand * weights_scaled))
    
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
    oscillatory_method: OscillatoryMethod = "advanced",
    # Additional parameters
    CC_nodes: int = 5,
    phase_increment: float = 1.5708,
    gpu_block_size: int = 2048
) -> RadialDWBAIntegrals:
    """
    GPU Accelerated Version of radial_ME_all_L using CuPy.
    """
    if not HAS_CUPY:
        raise RuntimeError("radial_ME_all_L_gpu called but cupy is not installed.")
    
    # Note: GPU now implements Filon/CC for oscillatory integrals

    r = grid.r
    N_grid = len(r)
    idx_limit = N_grid
    
    # Extract wave parameters for logic and tails
    k_i = cont_i.k_au
    k_f = cont_f.k_au
    l_i = cont_i.l
    l_f = cont_f.l
    k_total = k_i + k_f

    if hasattr(cont_i, 'idx_match') and cont_i.idx_match > 0:
        idx_limit = min(idx_limit, cont_i.idx_match + 1)
    if hasattr(cont_f, 'idx_match') and cont_f.idx_match > 0:
        idx_limit = min(idx_limit, cont_f.idx_match + 1)
    
    # Physics-based validation for match point
    k_eff = min(k_i, k_f) if min(k_i, k_f) > 0.1 else max(k_i, k_f)
    if k_eff > 1e-3:
        r_turn = (max(l_i, l_f) + 0.5) / k_eff
        idx_turn = np.searchsorted(r, r_turn)
        MIN_IDX = max(idx_turn + 20, N_grid // 10)
    else:
        MIN_IDX = N_grid // 10
    
    if idx_limit < MIN_IDX:
        idx_limit = MIN_IDX
    
    r_match = r[idx_limit - 1] if idx_limit > 0 else r[-1]
    
    # Extract phase shift parameters for analytical tail
    delta_i = cont_i.phase_shift if hasattr(cont_i, 'phase_shift') else 0.0
    delta_f = cont_f.phase_shift if hasattr(cont_f, 'phase_shift') else 0.0
    
    # Get Coulomb parameters for ionic targets
    eta_i = cont_i.eta if hasattr(cont_i, 'eta') else 0.0
    eta_f = cont_f.eta if hasattr(cont_f, 'eta') else 0.0
    sigma_i = cont_i.sigma_l if hasattr(cont_i, 'sigma_l') else 0.0
    sigma_f = cont_f.sigma_l if hasattr(cont_f, 'sigma_l') else 0.0
    
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
    # If using Filon, we can avoid the full N*N matrix for kernel_L
    # and compute int_r2_dir in blocks.
    
    use_filon = use_oscillatory_quadrature and k_total > 0.5
    
    if not use_filon:
        r_col = r_gpu[:, None]
        r_row = r_gpu[None, :]
        inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
        ratio = cp.minimum(r_col, r_row) * inv_gtr
        ratio = cp.minimum(ratio, 1.0 - 1e-12)
        log_ratio = cp.log(ratio + 1e-30)
        del r_col, r_row
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # Filon mode: pre-calculate components ONLY for CC nodes
        filon_params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, CC_nodes)
        if filon_params:
            idx_l, idx_r = filon_params['idx_left'], filon_params['idx_right']
            w_l, w_r = filon_params['w_left'], filon_params['w_right']
            
            r_col = r_gpu[:, None]
            r_cc_l = r_gpu[idx_l][None, :]
            r_cc_r = r_gpu[idx_r][None, :]
            
            # inv_gtr_at_cc_l: (n_r, n_cc_total)
            inv_gtr_cc_l = 1.0 / cp.maximum(r_col, r_cc_l + 1e-30)
            log_ratio_cc_l = cp.log(cp.minimum(r_col, r_cc_l) * inv_gtr_cc_l + 1e-30)
            
            inv_gtr_cc_r = 1.0 / cp.maximum(r_col, r_cc_r + 1e-30)
            log_ratio_cc_r = cp.log(cp.minimum(r_col, r_cc_r) * inv_gtr_cc_r + 1e-30)
            
            # Interpolate to CC nodes
            inv_gtr_at_cc = inv_gtr_cc_l * w_l + inv_gtr_cc_r * w_r
            log_ratio_at_cc = log_ratio_cc_l * w_l + log_ratio_cc_r * w_r
            
            kernel_at_cc_pre = {
                'inv_gtr_at_cc': inv_gtr_at_cc,
                'log_ratio_at_cc': log_ratio_at_cc
            }
            del r_col, r_cc_l, r_cc_r, inv_gtr_cc_l, inv_gtr_cc_r, log_ratio_cc_l, log_ratio_cc_r
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        else:
            use_filon = False # Fallback if params failed
            r_col = r_gpu[:, None]
            r_row = r_gpu[None, :]
            inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
            ratio = cp.minimum(r_col, r_row) * inv_gtr
            ratio = cp.minimum(ratio, 1.0 - 1e-12)
            log_ratio = cp.log(ratio + 1e-30)
            del r_col, r_row
            cp.get_default_memory_pool().free_all_blocks()

    I_L_dir: Dict[int, float] = {}
    I_L_exc: Dict[int, float] = {}

    for L in range(L_max + 1):
        # Determine kernel_L (only if not using Filon efficiently)
        kernel_L = None
        if not use_filon:
            if L == 0:
                kernel_L = inv_gtr
            else:
                kernel_L = inv_gtr * cp.exp(L * log_ratio)
        else:
            # In Filon mode, for direct integral legacy/full_split/advanced paths,
            # we need int_r2_dir = kernel_L @ rho2_dir.
            # We compute it in blocks to save memory if N_grid is large.
            rho2_eff = rho2_dir_uw * w_gpu
            int_r2_dir = cp.zeros(idx_limit, dtype=float)
            
            BLOCK_SIZE = gpu_block_size
            r_col = r_gpu[:, None]
            r_row = r_gpu[None, :]
            for start in range(0, idx_limit, BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, idx_limit)
                # Compute block of kernel_L
                r_row_block = r_row[:, start:end]
                inv_gtr_b = 1.0 / cp.maximum(r_col, r_row_block + 1e-30)
                if L == 0:
                    kb = inv_gtr_b
                else:
                    ratio_b = cp.minimum(r_col, r_row_block) * inv_gtr_b
                    kb = inv_gtr_b * cp.exp(L * cp.log(cp.minimum(ratio_b, 1.0 - 1e-12) + 1e-30))
                
                int_r2_dir += cp.dot(kb, rho2_eff[start:end])
                del kb, inv_gtr_b
                if L > 0: del ratio_b
            del r_col, r_row
            cp.get_default_memory_pool().free_all_blocks()

        
        # --- Direct Integral on GPU ---
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if use_filon:
            # oscillatory_method: legacy, full_split, or advanced
            # int_r2_dir was already computed block-wise above to save memory
            I_in = _gpu_filon_direct(rho1_dir_uw, int_r2_dir, r_gpu, w_gpu, k_total, phase_increment, CC_nodes, precomputed=filon_params)
            
            if oscillatory_method == "legacy":
                I_dir = I_in
                # Add analytical tail if applicable
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
                    if abs(moment_L) > 1e-12:
                        tail_contrib = _analytical_multipole_tail(
                            r_match, k_i, k_f, delta_i, delta_f, l_i, l_f, L, moment_L,
                            eta_i, eta_f, sigma_i, sigma_f
                        )
                        I_dir += tail_contrib
            else:
                # full_split or advanced: both use Levin/Filon for tail
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
                            r_match, float(grid.r[-1]),
                            delta_phi=np.pi / 4
                        )
                I_dir = I_in + I_out
        else:
            # Standard method (weighted densities)
            # kernel_L was computed if not use_filon
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
            # Pass pre-sliced components instead of full matrix to save memory
            k_spec = {**kernel_at_cc_pre, 'L': L}
            I_ex = _gpu_filon_exchange(k_spec, rho1_ex_uw, rho2_ex_uw, r_gpu, w_gpu, k_total, phase_increment, CC_nodes, precomputed=filon_params)
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

