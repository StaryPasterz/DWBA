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


# =============================================================================
# GPU CACHE FOR ENERGY-LEVEL REUSE
# =============================================================================
# Safe cache containing only base resources that don't depend on continuum waves.
# Created once per energy point, passed to radial_ME_all_L_gpu to skip rebuilds.
# =============================================================================

@dataclass
class GPUCache:
    """
    Energy-level GPU cache for base resources.
    
    Contains persistent GPU buffers built once per energy point.
    Avoids O(L_max_proj^2) matrix rebuilds for each (l_i, l_f) pair.
    
    IMPORTANT: Does NOT cache kernel_L (depends on L) or exchange potentials
    (depend on continuum waves) - those were sources of problems.
    
    Attributes
    ----------
    r_gpu : cupy.ndarray
        Radial grid on device (full grid).
    w_gpu : cupy.ndarray
        Simpson weights on device (full grid).
    inv_gtr : cupy.ndarray or None
        Base kernel matrix 1/r_> (idx_limit × idx_limit).
    log_ratio : cupy.ndarray or None
        Base ratio log(r_</r_>) matrix.
    filon_params : dict or None
        Precomputed Filon quadrature parameters.
    idx_limit : int
        Grid index limit used for cached matrices.
    chi_cache : dict
        Cache of continuum waves: (channel, l) -> cp.ndarray.
    chi_lru : list
        LRU tracking for chi_cache eviction.
    max_chi_cached : int
        Maximum number of continuum waves to cache (default 20).
    """
    r_gpu: 'cp.ndarray'
    w_gpu: 'cp.ndarray'
    inv_gtr: Optional['cp.ndarray'] = None
    log_ratio: Optional['cp.ndarray'] = None
    filon_params: Optional[dict] = None
    idx_limit: int = -1
    # Phase 4: Continuum wave cache with LRU
    chi_cache: Optional[Dict] = None
    chi_lru: Optional[List] = None
    max_chi_cached: int = 20
    
    def __post_init__(self):
        """Initialize mutable fields after dataclass creation."""
        if self.chi_cache is None:
            self.chi_cache = {}
        if self.chi_lru is None:
            self.chi_lru = []
    
    @classmethod
    def from_grid(cls, grid: RadialGrid, idx_limit: int = -1, max_chi_cached: int = 20) -> 'GPUCache':
        """Create cache from RadialGrid."""
        if not HAS_CUPY:
            raise RuntimeError("GPUCache requires CuPy")
        
        r_gpu = cp.asarray(grid.r)
        w_gpu = cp.asarray(grid.w_simpson)
        
        return cls(
            r_gpu=r_gpu, 
            w_gpu=w_gpu, 
            idx_limit=idx_limit,
            chi_cache={},
            chi_lru=[],
            max_chi_cached=max_chi_cached
        )
    
    def get_chi(self, chi_wave: 'ContinuumWave', channel: str = 'i') -> 'cp.ndarray':
        """
        Get continuum wave on GPU, using cache if available.
        
        Parameters
        ----------
        chi_wave : ContinuumWave
            The continuum wave object (CPU).
        channel : str
            Channel identifier ('i' for incident, 'f' for final).
            
        Returns
        -------
        cp.ndarray
            Continuum wave on GPU.
        """
        key = (channel, chi_wave.l)
        
        # Check cache
        if key in self.chi_cache:
            # Update LRU
            if key in self.chi_lru:
                self.chi_lru.remove(key)
            self.chi_lru.append(key)
            return self.chi_cache[key]
        
        # Not in cache - transfer to GPU
        chi_gpu = cp.asarray(chi_wave.chi_of_r)
        
        # LRU eviction if needed
        while len(self.chi_cache) >= self.max_chi_cached and self.chi_lru:
            old_key = self.chi_lru.pop(0)
            if old_key in self.chi_cache:
                del self.chi_cache[old_key]
        
        # Add to cache
        self.chi_cache[key] = chi_gpu
        self.chi_lru.append(key)
        
        return chi_gpu
    
    def build_kernel_matrix(self, idx_limit: int) -> None:
        """Build base kernel matrices for given idx_limit."""
        if self.inv_gtr is not None and self.idx_limit == idx_limit:
            return
        
        # Clear old matrices
        self.inv_gtr = None
        self.log_ratio = None
        
        r_sub = self.r_gpu[:idx_limit]
        r_col = r_sub[:, None]
        r_row = r_sub[None, :]
        
        self.inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
        ratio = cp.minimum(r_col, r_row) * self.inv_gtr
        ratio = cp.minimum(ratio, 1.0 - 1e-12)
        self.log_ratio = cp.log(ratio + 1e-30)
        
        del r_col, r_row, ratio
        self.idx_limit = idx_limit
    
    def clear(self) -> None:
        """Clear all cached data (call at end of energy point)."""
        self.inv_gtr = None
        self.log_ratio = None
        self.filon_params = None
        self.idx_limit = -1
        # Phase 4: Clear chi cache
        if self.chi_cache:
            self.chi_cache.clear()
        if self.chi_lru:
            self.chi_lru.clear()
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()


def _compute_optimal_block_size(n_grid: int, gpu_memory_threshold: float = 0.7) -> int:
    """
    Compute optimal block size based on available GPU memory.
    
    Strategy: Each block requires n_grid × block_size × 8 bytes per matrix.
    During block computation we need ~4 intermediate matrices.
    
    Parameters
    ----------
    n_grid : int
        Number of grid points (rows in block matrix).
    gpu_memory_threshold : float
        Fraction of free GPU memory to use (default 0.7).
        
    Returns
    -------
    int
        Optimal block size that fits in available memory, clamped to [512, 16384].
    """
    if not HAS_CUPY:
        return 2048
    
    try:
        free_mem, total_mem = cp.cuda.Device().mem_info
        usable_mem = free_mem * gpu_memory_threshold
        
        # Each block: (n_grid, block_size) matrix × 8 bytes × 4 matrices
        bytes_per_column = n_grid * 8 * 4
        max_block = int(usable_mem / bytes_per_column) if bytes_per_column > 0 else 8192
        
        # Clamp to reasonable range [512, 16384], align to 512
        block = max(512, min(max_block, 16384))
        block = (block // 512) * 512
        
        logger.debug("GPU auto-tune: free=%.1f GB, computed block_size=%d", free_mem / 1e9, block)
        return block
    except Exception as e:
        logger.debug("GPU memory query failed: %s. Using default block size.", e)
        return 2048  # Safe fallback

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
    phase_increment: float = 1.5708,
    min_grid_fraction: float = 0.10,
    k_threshold: float = 0.5,
    # Bug #2 fix: Also check U_f for asymptotic validation
    U_f_array: Optional[np.ndarray] = None
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
    CC_nodes : int
        Number of Clenshaw-Curtis nodes per phase interval (oscillatory modes).
    phase_increment : float
        Phase increment (radians) per oscillatory sub-interval.
    min_grid_fraction : float
        Minimum fraction of the grid used for r_m (match point) validation.
    k_threshold : float
        Threshold for enabling oscillatory quadrature (default: 0.5 a.u.).
        When k_i + k_f > k_threshold, specialized Filon/Levin methods are used.
        When k_i + k_f <= k_threshold (low energies), standard Simpson integration
        is used instead, as oscillations are weak and standard quadrature is 
        both faster and sufficiently accurate.
        
        Physics rationale: At low energies (small k), the continuum waves have
        long wavelengths, so fewer oscillations occur within the integration
        domain. Standard Simpson quadrature with O(h⁴) accuracy handles these
        cases efficiently without requiring phase-adapted methods.
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
        MIN_IDX = max(idx_turn + 20, int(N_grid * min_grid_fraction))  # Past turning + minimum grid fraction
    else:
        # Very low energy: use 10% of grid as fallback
        MIN_IDX = max(1, int(N_grid * min_grid_fraction))
    
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
    # Check that |V_eff(r_m)| / (k²/2) < threshold ensures match point is truly
    # in the asymptotic region where potential is negligible compared to kinetic.
    # CRITICAL: We MUST include the centrifugal term here, as it's the dominant
    # "potential" for high partial waves!
    # ==========================================================================
    ASYMPTOTIC_THRESHOLD = 0.03  # |V_eff|/(k²/2) should be < 3%
    r_m_idx = max(0, idx_limit - 1)
    
    def get_max_V_eff(idx):
        ri = r[idx]
        Ui = abs(U_i_array[idx])
        Uf = abs(U_f_array[idx]) if U_f_array is not None else 0.0
        V_cent = (l_max_wave * (l_max_wave + 1)) / (2.0 * ri**2)
        return max(Ui, Uf) + V_cent

    kinetic_energy = 0.5 * min(k_i, k_f)**2
    
    if kinetic_energy > 1e-10:
        V_eff_rm = get_max_V_eff(r_m_idx)
        ratio = V_eff_rm / kinetic_energy
        
        if ratio > ASYMPTOTIC_THRESHOLD:
            # Try to find a better match point further out
            for try_idx in range(idx_limit, min(idx_limit + 500, N_grid)):
                V_eff_try = get_max_V_eff(try_idx)
                if V_eff_try / kinetic_energy < ASYMPTOTIC_THRESHOLD:
                    idx_limit = try_idx + 1
                    break
            else:
                # If cannot reach threshold, just warn
                logger.debug(
                    "Match point r_m=%.2f a₀ has V_eff/E = %.2f > %.2f. Using best available point.",
                    r[idx_limit-1], ratio, ASYMPTOTIC_THRESHOLD
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
        
        if use_oscillatory_quadrature and k_total > k_threshold:
            if oscillatory_method == "legacy":
                # === LEGACY METHOD ===
                # Use Filon + Clenshaw-Curtis on phase-split segments
                I_dir = oscillatory_kernel_integral_2d(
                    rho1_dir_uw, rho2_dir_uw, kernel_L, r, k_i, k_f, idx_limit,
                    method="filon", w_grid=w,
                    n_nodes=CC_nodes, phase_increment=phase_increment
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
                    rho1_dir_uw, rho2_dir_uw,
                    kernel_L, r, k_i, k_f, idx_limit,
                    idx_limit_r2=N_grid,
                    method="filon", w_grid=w,
                    n_nodes=CC_nodes, phase_increment=phase_increment
                )
                
                # --- I_out: Pure oscillatory [r_m, r_max] via Levin/Filon ---
                # For r1 > r_m, use asymptotic sinA×sinB = ½[cos(A-B) - cos(A+B)]
                I_out = 0.0
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    # Multipole moment from bound states (use full weights w)
                    moment_L = np.sum(w * (r ** L) * u_f * u_i)
                    
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
                        
                        # Bug #3 fix: Add analytical tail from r_max to ∞
                        # For L≥1, the multipole kernel 1/r^(L+1) has non-zero contribution beyond r_max
                        I_tail_inf = _analytical_multipole_tail(
                            r[-1], k_i, k_f, delta_i, delta_f, l_i, l_f, L,
                            bound_overlap=moment_L,
                            eta_i=eta_i, eta_f=eta_f, sigma_i=sigma_i, sigma_f=sigma_f
                        )
                        I_out += I_tail_inf
                
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
                    # Multipole moment from bound states (localized, use full weights w)
                    moment_L = np.sum(w * (r ** L) * u_f * u_i)
                    
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
                        
                        # Bug #3 fix: Add analytical tail from r_max to ∞
                        # The multipole kernel 1/r^(L+1) contribution beyond grid
                        I_tail_inf = _analytical_multipole_tail(
                            r[-1], k_i, k_f, delta_i, delta_f, l_i, l_f, L,
                            bound_overlap=moment_L,
                            eta_i=eta_i, eta_f=eta_f, sigma_i=sigma_i, sigma_f=sigma_f
                        )
                        I_out += I_tail_inf
                
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
                moment_L = np.sum(w * (r ** L) * u_f * u_i)
                
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
        
        if use_oscillatory_quadrature and k_total > k_threshold:
            # Use Filon + Clenshaw-Curtis for BOTH inner and outer integrals
            # This handles oscillations in exchange densities properly
            I_ex = oscillatory_kernel_integral_2d(
                rho1_ex_uw, rho2_ex_uw, kernel_L, r, k_i, k_f, idx_limit,
                method="filon_exchange", w_grid=w,
                n_nodes=CC_nodes, phase_increment=phase_increment
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

def _gpu_filon_direct(rho1_uw, int_r2, r_gpu, w_gpu, k_total, phase_increment=1.5708, n_nodes=5, precomputed=None, return_gpu=False):
    """
    GPU Filon quadrature for direct integral outer loop.
    
    Parameters
    ----------
    return_gpu : bool
        If True, return GPU scalar (cp.ndarray with shape ()) instead of Python float.
        This avoids GPU-CPU synchronization when accumulating in GPU arrays.
    """
    if k_total < 1e-6:
        result = cp.dot(rho1_uw * w_gpu, int_r2)
        return result if return_gpu else float(result)
    
    if precomputed is None:
        params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, n_nodes)
        if params is None:
            result = cp.dot(rho1_uw * w_gpu, int_r2)
            return result if return_gpu else float(result)
    else:
        params = precomputed

    n_valid = params['n_valid']
    all_r_flat = params['all_r_flat']
    weights_scaled = params['weights_scaled']
    n_nodes = params['n_nodes']
    
    # OPTIMIZED: Pure GPU interpolation
    rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, n_nodes)
    int_r2_cc = cp.interp(all_r_flat, r_gpu, int_r2).reshape(n_valid, n_nodes)
    
    integrand = rho1_cc * int_r2_cc
    
    # Sum - stays on GPU if return_gpu=True
    result = cp.sum(integrand * weights_scaled)
    return result if return_gpu else float(result)



def _gpu_filon_exchange(kernel_L, rho1_uw, rho2_uw, r_gpu, w_gpu, k_total, phase_increment=1.5708, n_nodes=5, precomputed=None, return_gpu=False):
    """
    GPU Filon quadrature for exchange integral (CC on both inner and outer).
    
    Parameters
    ----------
    return_gpu : bool
        If True, return GPU scalar instead of Python float.
    """
    if k_total < 1e-6:
        int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
        result = cp.dot(rho1_uw * w_gpu, int_r2)
        return result if return_gpu else float(result)
    
    if precomputed is None:
        params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, n_nodes)
        if params is None:
            int_r2 = cp.dot(kernel_L, rho2_uw * w_gpu)
            result = cp.dot(rho1_uw * w_gpu, int_r2)
            return result if return_gpu else float(result)
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
    result = cp.sum(outer_integrand * weights_scaled)
    
    return result if return_gpu else float(result)


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
    gpu_block_size: int = 0,  # 0 = auto-tune based on VRAM
    min_grid_fraction: float = 0.10,
    k_threshold: float = 0.5,
    gpu_memory_mode: str = "auto",
    gpu_memory_threshold: float = 0.7,
    gpu_cache: Optional[GPUCache] = None,  # Phase 3: Energy-level cache
    # Bug #2 fix: Also check U_f for asymptotic validation
    U_f_array: Optional[np.ndarray] = None
) -> RadialDWBAIntegrals:
    """
    GPU Accelerated Version of radial_ME_all_L using CuPy.

    Architecture (v2.4+):
    -------------------
    1. **Hybrid Memory Strategy**: By default (gpu_memory_mode="auto"), checks available
       GPU memory before deciding between full matrix (fast) or block-wise (safe).
       - "full": Forces full N×N matrix construction (fastest, may cause OOM)
       - "block": Forces block-wise construction (slower, constant memory)
       - "auto": Uses memory check + exception fallback
    2. **Auto-tuning Block Size**: When gpu_block_size=0, computes optimal block size
       based on available GPU memory. Set explicit value to override.
    3. **Full-Grid Parity**: The inner integral (r2) for both direct and 
       exchange terms is computed over the full grid range [0, R_max], 
       ensuring mathematical parity with the CPU "Full-Split" implementation.
    3. **Multipole Moments**: Asymptotic coefficients M_L are computed using 
       full-grid weights (w_full_gpu), accurately capturing target state 
       tails beyond the projectile match point r_m.
    4. **Native Interpolation**: CC/Filon nodes are mapped to the radial grid 
       using pure GPU interpolation (cp.interp), avoiding CPU synchronization.
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
        MIN_IDX = max(idx_turn + 20, int(N_grid * min_grid_fraction))
    else:
        MIN_IDX = max(1, int(N_grid * min_grid_fraction))
    
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
    
    # 1. Transfer Data to GPU (or use cache)
    # Match point limit for r1 and continuum waves
    r_lim = r[:idx_limit]
    
    # Use cached grid data if available, otherwise transfer
    if gpu_cache is not None:
        r_gpu = gpu_cache.r_gpu[:idx_limit]
        w_gpu = gpu_cache.w_gpu[:idx_limit]
        r_full_gpu = gpu_cache.r_gpu
        w_full_gpu = gpu_cache.w_gpu
    else:
        r_gpu = cp.asarray(r_lim)
        w_lim = grid.w_simpson[:idx_limit]
        w_gpu = cp.asarray(w_lim)
        r_full_gpu = cp.asarray(r)
        w_full_gpu = cp.asarray(grid.w_simpson)
    
    # Bound state data (must be transferred - depends on target)
    u_i_full_gpu = cp.asarray(bound_i.u_of_r)
    if isinstance(bound_f, BoundOrbital):
        u_f_full_gpu = cp.asarray(bound_f.u_of_r)
    elif hasattr(bound_f, 'chi_of_r'):
        u_f_full_gpu = cp.asarray(bound_f.chi_of_r) # Ionization path
    else:
        u_f_full_gpu = u_i_full_gpu # Fallback
    
    # Match point sliced data for standard integrals
    u_i_gpu = cp.asarray(bound_i.u_of_r[:idx_limit])
    u_f_gpu = u_f_full_gpu[:idx_limit] # already array
    
    # Phase 4: Use cached continuum waves when gpu_cache is available
    if gpu_cache is not None:
        chi_i_full = gpu_cache.get_chi(cont_i, 'i')
        chi_f_full = gpu_cache.get_chi(cont_f, 'f')
        chi_i_gpu = chi_i_full[:idx_limit]
        chi_f_gpu = chi_f_full[:idx_limit]
    else:
        chi_i_gpu = cp.asarray(cont_i.chi_of_r[:idx_limit])
        chi_f_gpu = cp.asarray(cont_f.chi_of_r[:idx_limit])
    
    V_diff_gpu = cp.asarray((V_core_array - U_i_array)[:idx_limit])

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

    # 3. Kernel Construction on GPU
    # Strategy selection: "auto" checks memory, "full" forces matrix, "block" forces block-wise
    
    use_filon = use_oscillatory_quadrature and k_total > k_threshold
    
    # Memory-based decision for matrix construction
    use_block_wise = False
    if gpu_memory_mode == "block":
        use_block_wise = True
    elif gpu_memory_mode == "full":
        use_block_wise = False
    else:  # "auto" - check GPU memory
        try:
            free_mem, total_mem = cp.cuda.Device().mem_info
            # Memory estimation depends on mode:
            # - Standard: 3 matrices (inv_gtr, ratio, log_ratio) × idx_limit²
            # - Filon: Extended matrix (idx_limit × N_grid) + standard
            if use_filon:
                # Extended Filon matrix + standard matrix
                required_mem = (idx_limit * N_grid * 8 * 2) + (idx_limit * idx_limit * 8 * 3)
            else:
                required_mem = idx_limit * idx_limit * 8 * 3
            
            if required_mem > free_mem * gpu_memory_threshold:
                logger.info("GPU memory limited (%.1f GB free, need %.1f GB). Using block-wise.",
                           free_mem / 1e9, required_mem / 1e9)
                use_block_wise = True
        except Exception as e:
            logger.debug("Could not check GPU memory: %s. Using block-wise as fallback.", e)
            use_block_wise = True
    
    # -------------------------------------------------------------------------
    # Kernel Matrix Construction
    # -------------------------------------------------------------------------
    # Strategy:
    # - use_filon: controls quadrature method (Filon for oscillatory integrals)
    # - use_block_wise: controls memory strategy (independent of use_filon!)
    # When VRAM permits, build full matrix and use it for BOTH standard and Filon paths.
    
    full_matrix_built = False
    inv_gtr = None
    log_ratio = None
    
    if not use_block_wise:
        # Try to build full kernel matrix (works for BOTH Filon and standard)
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        try:
            r_col = r_gpu[:, None]
            r_row = r_gpu[None, :]
            inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
            ratio = cp.minimum(r_col, r_row) * inv_gtr
            ratio = cp.minimum(ratio, 1.0 - 1e-12)
            log_ratio = cp.log(ratio + 1e-30)
            del r_col, r_row, ratio
            cp.get_default_memory_pool().free_all_blocks()
            full_matrix_built = True
            logger.debug("GPU: Standard matrix built (%d×%d)", idx_limit, idx_limit)
        except Exception as e:
            # Catch OutOfMemoryError and Windows pagefile errors
            err_str = str(e).lower()
            if "memory" in err_str or "pagefile" in err_str or "out of memory" in err_str:
                logger.warning("GPU memory error: %s. Falling back to block-wise.", e)
                use_block_wise = True
                full_matrix_built = False
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            else:
                raise
    
    # For Filon, build extended matrix (idx_limit × N_grid) if memory permits
    # This allows full cp.dot() for int_r2_dir without block-wise loop
    filon_kernel_built = False
    filon_inv_gtr = None
    filon_log_ratio = None
    
    if use_filon and not use_block_wise:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        try:
            r_col = r_gpu[:, None]  # (idx_limit, 1)
            r_row_full = r_full_gpu[None, :]  # (1, N_grid)
            filon_inv_gtr = 1.0 / cp.maximum(r_col, r_row_full + 1e-30)
            filon_ratio = cp.minimum(r_col, r_row_full) * filon_inv_gtr
            filon_ratio = cp.minimum(filon_ratio, 1.0 - 1e-12)
            filon_log_ratio = cp.log(filon_ratio + 1e-30)
            del r_col, r_row_full, filon_ratio
            cp.get_default_memory_pool().free_all_blocks()
            filon_kernel_built = True
            logger.debug("GPU: Filon extended matrix built (%d×%d)", idx_limit, N_grid)
        except Exception as e:
            err_str = str(e).lower()
            if "memory" in err_str or "pagefile" in err_str or "out of memory" in err_str:
                logger.info("GPU: Filon extended matrix too large, using hybrid mode")
                filon_kernel_built = False
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            else:
                raise
    
    # Filon params for oscillatory quadrature (only if use_filon)
    filon_params = None
    if use_filon:
        filon_params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, CC_nodes)
        if filon_params:
            # Precompute CC node kernels for Filon
            idx_l, idx_r = filon_params['idx_left'], filon_params['idx_right']
            w_l, w_r = filon_params['w_left'], filon_params['w_right']
            
            r_col = r_gpu[:, None]
            r_cc_l = r_gpu[idx_l][None, :]
            r_cc_r = r_gpu[idx_r][None, :]
            
            inv_gtr_cc_l = 1.0 / cp.maximum(r_col, r_cc_l + 1e-30)
            log_ratio_cc_l = cp.log(cp.minimum(r_col, r_cc_l) * inv_gtr_cc_l + 1e-30)
            
            inv_gtr_cc_r = 1.0 / cp.maximum(r_col, r_cc_r + 1e-30)
            log_ratio_cc_r = cp.log(cp.minimum(r_col, r_cc_r) * inv_gtr_cc_r + 1e-30)
            
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
            use_filon = False  # Fallback if params generation failed
    
    # Log final computation mode after all matrix construction
    if use_filon:
        if filon_kernel_built:
            mode_str = "Filon/full-matrix"
        elif full_matrix_built:
            mode_str = "Filon/hybrid"
        else:
            mode_str = "Filon/block-wise"
    else:
        mode_str = "standard/full-matrix" if full_matrix_built else "standard/block-wise"
    
    # Deduplicate log: only log if mode changed from last call (ignore grid/L_max changes)
    if not hasattr(radial_ME_all_L_gpu, '_last_mode') or radial_ME_all_L_gpu._last_mode != mode_str:
        logger.info("GPU mode: %s", mode_str)
        radial_ME_all_L_gpu._last_mode = mode_str
    
    # === PHASE 1 OPTIMIZATION: GPU Array Accumulation ===
    # Instead of Dict[int, float] with per-L float() syncs, use GPU arrays
    # and do a single transfer at the end. This eliminates ~2×L_max syncs.
    I_L_dir_gpu = cp.zeros(L_max + 1, dtype=cp.float64)
    I_L_exc_gpu = cp.zeros(L_max + 1, dtype=cp.float64)

    # Block size for block-wise computation (only used if needed)
    # Handle string "auto" or int values
    try:
        block_size_int = int(gpu_block_size) if gpu_block_size != "auto" else 0
    except (ValueError, TypeError):
        block_size_int = 0
    BLOCK_SIZE = block_size_int if block_size_int > 0 else _compute_optimal_block_size(idx_limit, gpu_memory_threshold)
    
    # Precompute rho2_eff_full ONCE before L-loop (optimization: was computed per-L)
    rho2_eff_full = (u_f_full_gpu * u_i_full_gpu) * w_full_gpu if use_filon else None
    
    # Precompute L=0 correction terms on GPU (avoid per-L sync)
    sum_rho2_dir = cp.sum(rho2_dir_w)
    sum_rho2_ex = cp.sum(rho2_ex_w)
    V_diff_dot_rho1_dir = cp.dot(rho1_dir_w, V_diff_gpu)
    V_diff_dot_rho1_ex = cp.dot(rho1_ex_w, V_diff_gpu)

    for L in range(L_max + 1):
        kernel_L = None
        int_r2_dir = None  # For Filon mode
        
        if full_matrix_built:
            # FAST PATH: use prebuilt full matrix
            if L == 0:
                kernel_L = inv_gtr
            else:
                kernel_L = inv_gtr * cp.exp(L * log_ratio)
            
            # For Filon mode, compute int_r2_dir:
            # FAST: If filon_kernel_built, use full (idx_limit × N_grid) matrix
            # HYBRID: If only full_matrix_built, use (idx_limit × idx_limit) + block-wise tail
            if use_filon:
                # rho2_eff_full precomputed before L-loop
                
                if filon_kernel_built:
                    # FULL MATRIX PATH: Single cp.dot() for entire grid
                    if L == 0:
                        filon_kernel_L = filon_inv_gtr
                    else:
                        filon_kernel_L = filon_inv_gtr * cp.exp(L * filon_log_ratio)
                    int_r2_dir = cp.dot(filon_kernel_L, rho2_eff_full)
                    del filon_kernel_L
                else:
                    # HYBRID PATH: Use standard matrix for head + block-wise for tail
                    int_r2_dir = cp.dot(kernel_L, rho2_eff_full[:idx_limit])
                    
                    if N_grid > idx_limit:
                        r_col = r_gpu[:, None]
                        r_row_tail = r_full_gpu[idx_limit:][None, :]
                        tail_size = N_grid - idx_limit
                        
                        for start in range(0, tail_size, BLOCK_SIZE):
                            end = min(start + BLOCK_SIZE, tail_size)
                            r_block = r_row_tail[:, start:end]
                            inv_gtr_b = 1.0 / cp.maximum(r_col, r_block + 1e-30)
                            if L == 0:
                                kb = inv_gtr_b
                            else:
                                ratio_b = cp.minimum(r_col, r_block) * inv_gtr_b
                                kb = inv_gtr_b * cp.exp(L * cp.log(cp.minimum(ratio_b, 1.0 - 1e-12) + 1e-30))
                            int_r2_dir += cp.dot(kb, rho2_eff_full[idx_limit + start:idx_limit + end])
                            del kb, inv_gtr_b
                        del r_col, r_row_tail
        else:
            # SLOW PATH: block-wise computation (no prebuilt matrix)
            if use_filon:
                # Compute int_r2_dir block-wise (rho2_eff_full precomputed)
                int_r2_dir = cp.zeros(idx_limit, dtype=float)
                r_col = r_gpu[:, None]
                r_row_full = r_full_gpu[None, :]
                for start in range(0, N_grid, BLOCK_SIZE):
                    end = min(start + BLOCK_SIZE, N_grid)
                    inv_gtr_b = 1.0 / cp.maximum(r_col, r_row_full[:, start:end] + 1e-30)
                    if L == 0:
                        kb = inv_gtr_b
                    else:
                        ratio_b = cp.minimum(r_col, r_row_full[:, start:end]) * inv_gtr_b
                        kb = inv_gtr_b * cp.exp(L * cp.log(cp.minimum(ratio_b, 1.0 - 1e-12) + 1e-30))
                    int_r2_dir += cp.dot(kb, rho2_eff_full[start:end])
                    del kb, inv_gtr_b
                del r_col, r_row_full
            else:
                # Standard path without prebuilt matrix - compute kernel block-wise
                # This case should be rare (low memory + low k)
                r_col = r_gpu[:, None]
                r_row = r_gpu[None, :]
                if L == 0:
                    kernel_L = 1.0 / cp.maximum(r_col, r_row + 1e-30)
                else:
                    inv_gtr_temp = 1.0 / cp.maximum(r_col, r_row + 1e-30)
                    ratio_temp = cp.minimum(r_col, r_row) * inv_gtr_temp
                    ratio_temp = cp.minimum(ratio_temp, 1.0 - 1e-12)
                    kernel_L = inv_gtr_temp * cp.exp(L * cp.log(ratio_temp + 1e-30))
                    del inv_gtr_temp, ratio_temp
                del r_col, r_row

        
        # --- Direct Integral on GPU ---
        is_excitation = isinstance(bound_f, BoundOrbital)
        
        if use_filon:
            # oscillatory_method: legacy, full_split, or advanced
            # int_r2_dir was already computed block-wise above to save memory
            # NOTE: _gpu_filon_direct returns a GPU scalar, not Python float
            I_dir_L = _gpu_filon_direct(rho1_dir_uw, int_r2_dir, r_gpu, w_gpu, k_total, phase_increment, CC_nodes, precomputed=filon_params, return_gpu=True)
            
            if oscillatory_method == "legacy":
                # Add analytical tail if applicable (requires CPU computation)
                # For legacy mode, we still need float conversion for tail
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = float(cp.sum(w_gpu * (r_gpu ** L) * u_f_gpu * u_i_gpu))
                    if abs(moment_L) > 1e-12:
                        tail_contrib = _analytical_multipole_tail(
                            r_match, k_i, k_f, delta_i, delta_f, l_i, l_f, L, moment_L,
                            eta_i, eta_f, sigma_i, sigma_f
                        )
                        I_dir_L = I_dir_L + tail_contrib
            else:
                # full_split or advanced: both use Levin/Filon for tail
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    # Multipole moment from bound states (use full grid for accuracy)
                    moment_L = float(cp.sum(w_full_gpu * (r_full_gpu ** L) * u_f_full_gpu * u_i_full_gpu))
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
                        
                        # Bug #3 fix: Add analytical tail from r_max to ∞
                        I_tail_inf = _analytical_multipole_tail(
                            float(grid.r[-1]), k_i, k_f, delta_i, delta_f, l_i, l_f, L,
                            bound_overlap=moment_L,
                            eta_i=eta_i, eta_f=eta_f, sigma_i=sigma_i, sigma_f=sigma_f
                        )
                        I_out += I_tail_inf
                        
                        I_dir_L = I_dir_L + I_out
        else:
            # Standard method (weighted densities) - stays on GPU
            int_r2 = cp.dot(kernel_L, rho2_dir_w)
            I_dir_L = cp.dot(rho1_dir_w, int_r2)
        
        # L=0 correction using precomputed GPU values
        if L == 0:
            # Use precomputed sums (no sync here)
            I_dir_L = I_dir_L + V_diff_dot_rho1_dir * sum_rho2_dir
                
        I_L_dir_gpu[L] = I_dir_L
        
        # --- Exchange Integral on GPU ---
        if use_oscillatory_quadrature and k_total > k_threshold:
            # Use GPU Filon + CC for exchange (both inner and outer)
            # Pass pre-sliced components instead of full matrix to save memory
            k_spec = {**kernel_at_cc_pre, 'L': L} if 'kernel_at_cc_pre' in locals() else kernel_L
            I_ex_L = _gpu_filon_exchange(
                k_spec, rho1_ex_uw, rho2_ex_uw, r_gpu, w_gpu, k_total, 
                phase_increment, CC_nodes, precomputed=filon_params, return_gpu=True
            )
        else:
            int_r2_ex = cp.dot(kernel_L, rho2_ex_w)
            I_ex_L = cp.dot(rho1_ex_w, int_r2_ex)

        # L=0 exchange correction using precomputed GPU values
        if L == 0:
            I_ex_L = I_ex_L + V_diff_dot_rho1_ex * sum_rho2_ex
                
        I_L_exc_gpu[L] = I_ex_L
        
    # === SINGLE TRANSFER AT END ===
    # Convert GPU arrays to CPU and build result dicts
    I_L_dir_cpu = I_L_dir_gpu.get()
    I_L_exc_cpu = I_L_exc_gpu.get()
    
    I_L_dir = {L: float(I_L_dir_cpu[L]) for L in range(L_max + 1)}
    I_L_exc = {L: float(I_L_exc_cpu[L]) for L in range(L_max + 1)}
        
    # Cleanup big arrays (only if they exist and weren't already deleted)
    # Note: These may have been deleted earlier in block-wise paths
    if 'inv_gtr' in locals() and inv_gtr is not None:
        del inv_gtr
    if 'log_ratio' in locals() and log_ratio is not None:
        del log_ratio
    if 'filon_inv_gtr' in locals() and filon_inv_gtr is not None:
        del filon_inv_gtr
    if 'filon_log_ratio' in locals() and filon_log_ratio is not None:
        del filon_log_ratio
    # NOTE: Removed free_all_blocks() from here - moved to driver.py level (Phase 2)

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)
