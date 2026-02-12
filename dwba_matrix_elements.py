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
import logging
import os
import time
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
    dwba_outer_integral_1d_multipole_batch,
    get_outer_batch_config,
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


def _is_hotpath_verbose_debug_enabled() -> bool:
    """
    Return True when verbose per-call GPU hot-path debug is explicitly enabled.

    Set environment variable `DWBA_HOTPATH_DEBUG=1` to force detailed logging.
    """
    return os.environ.get("DWBA_HOTPATH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _should_sampled_hotpath_debug(
    key: str,
    every: int = 200,
    initial: int = 3
) -> bool:
    """
    Sample hot-path DEBUG logs to avoid I/O bottlenecks in tight GPU loops.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return False
    if _is_hotpath_verbose_debug_enabled():
        return True

    counters = getattr(_should_sampled_hotpath_debug, "_counters", None)
    if counters is None:
        counters = {}
        _should_sampled_hotpath_debug._counters = counters

    count = counters.get(key, 0) + 1
    counters[key] = count

    if count <= initial:
        return True
    return (count % max(1, int(every))) == 0


def _get_ratio_policy() -> str:
    """
    Read ratio-power policy for GPU kernel generation.

    Environment:
        DWBA_GPU_RATIO_POLICY = auto | on | off
    """
    raw = os.environ.get("DWBA_GPU_RATIO_POLICY", "auto").strip().lower()
    if raw in {"auto", "on", "off"}:
        return raw
    return "auto"


def _matrix_bytes(n_rows: int, n_cols: int, itemsize: int = 8) -> int:
    """Return approximate dense matrix size in bytes."""
    return int(n_rows) * int(n_cols) * int(itemsize)


def _should_enable_ratio_cache(
    policy: str,
    matrix_bytes: int,
    mem_budget_bytes: Optional[float],
    kind: str
) -> bool:
    """
    Decide whether ratio-cache/recursive power path should be enabled.

    Notes
    -----
    - `on`: always enabled
    - `off`: always disabled (legacy per-L exp path)
    - `auto`: enabled only for moderate matrix sizes and sufficient budget headroom
    """
    if policy == "on":
        return True
    if policy == "off":
        return False

    # auto: conservative thresholds to avoid VRAM pressure on consumer GPUs.
    if kind == "filon":
        abs_limit = 128 * 1024 * 1024
    elif kind == "standard":
        abs_limit = 96 * 1024 * 1024
    elif kind == "cc":
        abs_limit = 80 * 1024 * 1024
    else:
        abs_limit = 64 * 1024 * 1024

    if matrix_bytes > abs_limit:
        return False

    # Recursive mode needs ratio-cache + working matrix (~2x matrix_bytes extra).
    extra_need = 2.0 * float(matrix_bytes)
    if mem_budget_bytes is not None and extra_need > 0.35 * float(mem_budget_bytes):
        return False

    return True

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
    ratio : cupy.ndarray or None
        Cached exp(log_ratio) for fast recursive L-updates.
    filon_inv_gtr : cupy.ndarray or None
        Extended Filon kernel base 1/r_> (idx_limit × N_grid).
    filon_log_ratio : cupy.ndarray or None
        Extended Filon ratio log(r_</r_>) matrix (idx_limit × N_grid).
    filon_ratio : cupy.ndarray or None
        Cached exp(filon_log_ratio) for full Filon recursive L-updates.
    kernel_cc_pre : dict or None
        Last precomputed Filon exchange kernel-at-CC base tensors (backward compatibility).
    kernel_cc_cache : dict[tuple, dict] or None
        Small LRU cache for Filon exchange kernel-at-CC tensors keyed by Filon params.
    filon_params : dict or None
        Last precomputed Filon quadrature parameters (backward compatibility).
    filon_params_cache : dict[tuple, dict] or None
        Small LRU cache for Filon quadrature parameters keyed by Filon params.
    idx_limit : int
        Grid index limit used for cached matrices.
    chi_cache : dict
        Cache of continuum waves: (channel, l) -> cp.ndarray.
    array_cache : dict
        Cache of static 1D arrays transferred to GPU (potentials/bound states).
    chi_lru : list
        LRU tracking for chi_cache eviction.
    max_chi_cached : int
        Maximum number of continuum waves to cache (default 20).
    """
    r_gpu: 'cp.ndarray'
    w_gpu: 'cp.ndarray'
    inv_gtr: Optional['cp.ndarray'] = None
    log_ratio: Optional['cp.ndarray'] = None
    ratio: Optional['cp.ndarray'] = None
    filon_inv_gtr: Optional['cp.ndarray'] = None
    filon_log_ratio: Optional['cp.ndarray'] = None
    filon_ratio: Optional['cp.ndarray'] = None
    filon_idx_limit: int = -1
    kernel_cc_pre: Optional[dict] = None
    kernel_cc_key: Optional[tuple] = None
    filon_params: Optional[dict] = None
    kernel_cc_cache: Optional[Dict[tuple, dict]] = None
    kernel_cc_lru: Optional[List[tuple]] = None
    filon_params_cache: Optional[Dict[tuple, dict]] = None
    filon_params_lru: Optional[List[tuple]] = None
    max_small_cached: int = 6
    idx_limit: int = -1
    # Phase 4: Continuum wave cache with LRU
    chi_cache: Optional[Dict] = None
    array_cache: Optional[Dict] = None
    chi_lru: Optional[List] = None
    max_chi_cached: int = 20
    v_diff_cache: Optional['cp.ndarray'] = None
    v_diff_key: Optional[tuple] = None
    
    def __post_init__(self) -> None:
        """Initialize mutable fields after dataclass creation."""
        if self.chi_cache is None:
            self.chi_cache = {}
        if self.array_cache is None:
            self.array_cache = {}
        if self.chi_lru is None:
            self.chi_lru = []
        if self.kernel_cc_cache is None:
            self.kernel_cc_cache = {}
        if self.kernel_cc_lru is None:
            self.kernel_cc_lru = []
        if self.filon_params_cache is None:
            self.filon_params_cache = {}
        if self.filon_params_lru is None:
            self.filon_params_lru = []
    
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
        N_grid = len(self.r_gpu)  # Current grid size
        
        # Check cache
        if key in self.chi_cache:
            cached_chi = self.chi_cache[key]
            # Validate size matches current grid (LOCAL adaptive may change grid)
            if len(cached_chi) == N_grid:
                # Update LRU
                if key in self.chi_lru:
                    self.chi_lru.remove(key)
                self.chi_lru.append(key)
                return cached_chi
            else:
                # Size mismatch - invalidate and recompute
                del self.chi_cache[key]
                if key in self.chi_lru:
                    self.chi_lru.remove(key)
        
        # Not in cache or size mismatch - transfer to GPU (truncate if needed)
        chi_raw = chi_wave.chi_of_r
        if len(chi_raw) > N_grid:
            chi_raw = chi_raw[:N_grid]
        elif len(chi_raw) < N_grid:
            raise ValueError(f"Continuum wave chi has {len(chi_raw)} pts but grid has {N_grid}")
        chi_gpu = cp.asarray(chi_raw)
        
        # LRU eviction if needed
        while len(self.chi_cache) >= self.max_chi_cached and self.chi_lru:
            old_key = self.chi_lru.pop(0)
            if old_key in self.chi_cache:
                del self.chi_cache[old_key]
        
        # Add to cache
        self.chi_cache[key] = chi_gpu
        self.chi_lru.append(key)
        
        return chi_gpu

    def get_static_array(
        self,
        arr: np.ndarray,
        tag: str,
        n_grid: Optional[int] = None
    ) -> 'cp.ndarray':
        """
        Get a static 1D numpy array on GPU with pointer-based caching.

        This avoids repeated cp.asarray transfers for data that stays constant
        over many (l_i, l_f) matrix-element calls within one energy point.
        """
        arr_np = np.asarray(arr)
        if arr_np.ndim != 1:
            raise ValueError(f"Expected 1D array for tag={tag}, got shape={arr_np.shape}")

        target_n = arr_np.shape[0] if n_grid is None else int(n_grid)
        if arr_np.shape[0] < target_n:
            raise ValueError(
                f"Array '{tag}' has {arr_np.shape[0]} pts but requested {target_n}"
            )

        ptr = int(arr_np.__array_interface__["data"][0])
        key = (tag, ptr, target_n)
        cached = self.array_cache.get(key)
        if cached is not None and len(cached) == target_n:
            return cached

        arr_view = arr_np[:target_n] if arr_np.shape[0] != target_n else arr_np
        arr_gpu = cp.asarray(arr_view)

        # Keep cache bounded to avoid unbounded growth if many unique arrays appear.
        if len(self.array_cache) >= 64:
            self.array_cache.clear()
        self.array_cache[key] = arr_gpu
        return arr_gpu

    def get_v_diff(
        self,
        V_core_array: np.ndarray,
        U_i_array: np.ndarray,
        n_grid: int
    ) -> 'cp.ndarray':
        """
        Get cached V_core-U_i on GPU for the current grid size.
        """
        v_np = np.asarray(V_core_array)
        u_np = np.asarray(U_i_array)
        if v_np.shape[0] < n_grid or u_np.shape[0] < n_grid:
            raise ValueError(
                f"Cannot build V_diff: V_core={v_np.shape[0]}, U_i={u_np.shape[0]}, n_grid={n_grid}"
            )

        key = (
            int(v_np.__array_interface__["data"][0]),
            int(u_np.__array_interface__["data"][0]),
            int(n_grid),
        )
        if self.v_diff_cache is not None and self.v_diff_key == key and len(self.v_diff_cache) == n_grid:
            return self.v_diff_cache

        self.v_diff_cache = cp.asarray(v_np[:n_grid] - u_np[:n_grid])
        self.v_diff_key = key
        return self.v_diff_cache
    
    def build_kernel_matrix(self, idx_limit: int) -> None:
        """Build base kernel matrices for given idx_limit."""
        if self.inv_gtr is not None and self.idx_limit >= idx_limit:
            return
        
        # Clear old matrices
        self.inv_gtr = None
        self.log_ratio = None
        self.ratio = None
        
        r_sub = self.r_gpu[:idx_limit]
        r_col = r_sub[:, None]
        r_row = r_sub[None, :]
        
        self.inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
        ratio = cp.minimum(r_col, r_row) * self.inv_gtr
        ratio = cp.minimum(ratio, 1.0 - 1e-12)
        self.log_ratio = cp.log(ratio + 1e-30)
        
        del r_col, r_row, ratio
        self.idx_limit = idx_limit

    def get_kernel_ratio(self, idx_limit: int) -> Optional['cp.ndarray']:
        """
        Return cached exp(log_ratio) slice for requested idx_limit.
        """
        if self.log_ratio is None or self.idx_limit < idx_limit:
            return None
        if self.ratio is None or self.ratio.shape[0] < idx_limit:
            # Compute only requested prefix (avoid historical-size escalation).
            self.ratio = cp.exp(self.log_ratio[:idx_limit, :idx_limit])
        else:
            # If cached ratio is much larger than current need, shrink it.
            cached_n = int(self.ratio.shape[0])
            if cached_n > (idx_limit + 512) and cached_n > int(1.5 * idx_limit):
                self.ratio = self.ratio[:idx_limit, :idx_limit].copy()
        return self.ratio[:idx_limit, :idx_limit]

    def build_filon_kernel_matrix(self, idx_limit: int) -> None:
        """
        Build extended Filon base matrices for given idx_limit.

        Shape is (idx_limit × N_grid) and can be reused across many (l_i, l_f)
        calls at the same energy/grid.
        """
        if self.filon_inv_gtr is not None and self.filon_idx_limit >= idx_limit:
            return

        self.filon_inv_gtr = None
        self.filon_log_ratio = None
        self.filon_ratio = None

        r_col = self.r_gpu[:idx_limit, None]
        r_row_full = self.r_gpu[None, :]

        self.filon_inv_gtr = 1.0 / cp.maximum(r_col, r_row_full + 1e-30)
        ratio = cp.minimum(r_col, r_row_full) * self.filon_inv_gtr
        ratio = cp.minimum(ratio, 1.0 - 1e-12)
        self.filon_log_ratio = cp.log(ratio + 1e-30)

        del r_col, r_row_full, ratio
        self.filon_idx_limit = idx_limit

    def get_filon_ratio(self, idx_limit: int) -> Optional['cp.ndarray']:
        """
        Return cached exp(filon_log_ratio) slice for requested idx_limit rows.
        """
        if self.filon_log_ratio is None or self.filon_idx_limit < idx_limit:
            return None
        if self.filon_ratio is None or self.filon_ratio.shape[0] < idx_limit:
            # Compute only requested rows (avoid historical-size escalation).
            self.filon_ratio = cp.exp(self.filon_log_ratio[:idx_limit, :])
        else:
            cached_rows = int(self.filon_ratio.shape[0])
            if cached_rows > (idx_limit + 512) and cached_rows > int(1.5 * idx_limit):
                self.filon_ratio = self.filon_ratio[:idx_limit, :].copy()
        return self.filon_ratio[:idx_limit, :]

    @staticmethod
    def _lru_get(cache: Dict[tuple, dict], lru: List[tuple], key: tuple) -> Optional[dict]:
        """LRU helper: return value and refresh recency."""
        value = cache.get(key)
        if value is None:
            return None
        if key in lru:
            lru.remove(key)
        lru.append(key)
        return value

    @staticmethod
    def _lru_set(
        cache: Dict[tuple, dict],
        lru: List[tuple],
        key: tuple,
        value: dict,
        max_size: int
    ) -> None:
        """LRU helper: insert/replace and evict oldest if needed."""
        cache[key] = value
        if key in lru:
            lru.remove(key)
        lru.append(key)
        while len(lru) > max(1, int(max_size)):
            old = lru.pop(0)
            cache.pop(old, None)

    def get_filon_params_cached(self, key: tuple) -> Optional[dict]:
        """Return Filon params from small LRU cache."""
        if self.filon_params_cache is None or self.filon_params_lru is None:
            return None
        return self._lru_get(self.filon_params_cache, self.filon_params_lru, key)

    def set_filon_params_cached(self, key: tuple, params: dict) -> None:
        """Store Filon params in small LRU cache."""
        if self.filon_params_cache is None:
            self.filon_params_cache = {}
        if self.filon_params_lru is None:
            self.filon_params_lru = []
        self._lru_set(
            self.filon_params_cache,
            self.filon_params_lru,
            key,
            params,
            self.max_small_cached
        )
        # Backward compatibility (single-entry fields).
        self.filon_params = {"key": key, "params": params}

    def get_kernel_cc_cached(self, key: tuple) -> Optional[dict]:
        """Return precomputed Filon exchange kernel-at-CC tensors from LRU cache."""
        if self.kernel_cc_cache is None or self.kernel_cc_lru is None:
            return None
        return self._lru_get(self.kernel_cc_cache, self.kernel_cc_lru, key)

    def set_kernel_cc_cached(self, key: tuple, kernel_cc_pre: dict) -> None:
        """Store precomputed Filon exchange kernel-at-CC tensors in LRU cache."""
        if self.kernel_cc_cache is None:
            self.kernel_cc_cache = {}
        if self.kernel_cc_lru is None:
            self.kernel_cc_lru = []
        self._lru_set(
            self.kernel_cc_cache,
            self.kernel_cc_lru,
            key,
            kernel_cc_pre,
            self.max_small_cached
        )
        # Backward compatibility (single-entry fields).
        self.kernel_cc_pre = kernel_cc_pre
        self.kernel_cc_key = key
    
    def clear(self) -> None:
        """Clear all cached data (call at end of energy point)."""
        self.inv_gtr = None
        self.log_ratio = None
        self.ratio = None
        self.filon_inv_gtr = None
        self.filon_log_ratio = None
        self.filon_ratio = None
        self.filon_idx_limit = -1
        self.kernel_cc_pre = None
        self.kernel_cc_key = None
        self.filon_params = None
        if self.kernel_cc_cache:
            self.kernel_cc_cache.clear()
        if self.kernel_cc_lru:
            self.kernel_cc_lru.clear()
        if self.filon_params_cache:
            self.filon_params_cache.clear()
        if self.filon_params_lru:
            self.filon_params_lru.clear()
        self.idx_limit = -1
        self.v_diff_cache = None
        self.v_diff_key = None
        # Phase 4: Clear chi cache
        if self.chi_cache:
            self.chi_cache.clear()
        if self.array_cache:
            self.array_cache.clear()
        if self.chi_lru:
            self.chi_lru.clear()
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()


def _compute_optimal_block_size(
    n_grid: int,
    gpu_memory_threshold: float = 0.8,
    effective_free_mem: Optional[int] = None
) -> int:
    """
    Compute optimal block size based on available GPU memory.
    
    Strategy: Each block requires n_grid × block_size × 8 bytes per matrix.
    During block computation we need ~4 intermediate matrices.
    
    Parameters
    ----------
    n_grid : int
        Number of grid points (rows in block matrix).
    gpu_memory_threshold : float
        Fraction of free GPU memory to use (default 0.8).
    effective_free_mem : int, optional
        Effective free bytes (device-free + reusable pool) when available.
        
    Returns
    -------
    int
        Optimal block size that fits in available memory, clamped to [512, 16384].
    """
    if not HAS_CUPY:
        return 2048
    
    try:
        free_mem, _total_mem = cp.cuda.Device().mem_info
        avail_mem = int(effective_free_mem) if effective_free_mem is not None else int(free_mem)
        usable_mem = avail_mem * gpu_memory_threshold
        
        # Each block: (n_grid, block_size) matrix × 8 bytes × 4 matrices
        bytes_per_column = n_grid * 8 * 4
        max_block = int(usable_mem / bytes_per_column) if bytes_per_column > 0 else 8192
        
        # Clamp to reasonable range [512, 16384], align to 512
        block = max(512, min(max_block, 16384))
        block = (block // 512) * 512
        
        if _should_sampled_hotpath_debug("gpu_auto_tune", every=250, initial=2):
            logger.debug(
                "GPU auto-tune: free=%.1f GB%s, computed block_size=%d",
                free_mem / 1e9,
                "" if effective_free_mem is None else f", effective={avail_mem / 1e9:.1f} GB",
                block
            )
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


def _refine_idx_limit_physics(
    idx_limit: int,
    r: np.ndarray,
    w_trapz: np.ndarray,
    U_i_array: np.ndarray,
    U_f_array: Optional[np.ndarray],
    k_i: float,
    k_f: float,
    l_i: int,
    l_f: int,
    bound_states: Optional[List[np.ndarray]] = None,
    context: str = "CPU",
) -> int:
    """
    Apply physical guardrails to match-point based split index.

    Mirrors the conservative checks used in the CPU path:
    - bound-state extent coverage (99% cumulative probability),
    - asymptotic validity of effective potential at r_m.
    """
    N_grid = len(r)
    idx_limit = int(max(1, min(int(idx_limit), N_grid)))
    l_max_wave = max(int(l_i), int(l_f))

    # Bound-state extent check: required for split-tail excitation formulas.
    if bound_states:
        BOUND_STATE_THRESHOLD = 0.99
        idx_99_max = 0
        for u_state in bound_states:
            if u_state is None:
                continue
            u_arr = np.asarray(u_state)
            if u_arr.ndim != 1 or len(u_arr) < N_grid:
                continue
            u_arr = u_arr[:N_grid]
            u_sq = u_arr * u_arr
            if float(np.sum(u_sq)) <= 1e-30:
                continue
            prob_cum = np.cumsum(u_sq * w_trapz[:N_grid])
            total = float(prob_cum[-1]) if len(prob_cum) else 0.0
            if total <= 1e-300 or not np.isfinite(total):
                continue
            prob_cum /= (total + 1e-300)
            idx_99 = int(np.searchsorted(prob_cum, BOUND_STATE_THRESHOLD))
            idx_99_max = max(idx_99_max, idx_99)

        if idx_99_max > idx_limit:
            old_idx = idx_limit
            idx_limit = min(N_grid, idx_99_max + 1)
            logger.debug(
                "%s bound extent: extending idx_limit %d -> %d (r_m %.2f -> %.2f a0)",
                context,
                old_idx,
                idx_limit,
                r[max(old_idx - 1, 0)],
                r[min(idx_limit - 1, N_grid - 1)],
            )

    # Asymptotic validation: V_eff / (k^2/2) should be small at split point.
    ASYMPTOTIC_THRESHOLD = 0.03
    kinetic_energy = 0.5 * min(float(k_i), float(k_f)) ** 2
    if kinetic_energy > 1e-10 and idx_limit > 0:
        r_m_idx = max(0, min(idx_limit - 1, N_grid - 1))

        def get_max_V_eff(idx: int) -> float:
            ri = max(float(r[idx]), 1e-12)
            Ui = abs(float(U_i_array[idx]))
            Uf = abs(float(U_f_array[idx])) if U_f_array is not None else 0.0
            V_cent = (l_max_wave * (l_max_wave + 1)) / (2.0 * ri * ri)
            return max(Ui, Uf) + V_cent

        V_eff_rm = get_max_V_eff(r_m_idx)
        ratio = V_eff_rm / kinetic_energy
        if ratio > ASYMPTOTIC_THRESHOLD:
            found = False
            for try_idx in range(idx_limit, min(idx_limit + 500, N_grid)):
                if get_max_V_eff(try_idx) / kinetic_energy < ASYMPTOTIC_THRESHOLD:
                    idx_limit = try_idx + 1
                    found = True
                    break
            if not found:
                logger.debug(
                    "%s asymptotic: V_eff/E=%.2f > %.2f at r_m=%.2f a0; using best available split.",
                    context,
                    ratio,
                    ASYMPTOTIC_THRESHOLD,
                    r[min(idx_limit - 1, N_grid - 1)],
                )

    return int(max(1, min(idx_limit, N_grid)))


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
    # NOTE: Also clamp to N_grid to handle grid size changes in LOCAL adaptive mode
    if hasattr(cont_i, 'idx_match') and 0 < cont_i.idx_match < N_grid:
        idx_limit = min(idx_limit, cont_i.idx_match + 1)
    if hasattr(cont_f, 'idx_match') and 0 < cont_f.idx_match < N_grid:
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
        # CRITICAL: Clamp idx_turn to prevent MIN_IDX exceeding N_grid (LOCAL adaptive fix)
        idx_turn = min(idx_turn, N_grid - 20)  # Ensure room for +20 safety margin
        MIN_IDX = min(max(idx_turn + 20, int(N_grid * min_grid_fraction)), N_grid)
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
    # BOUND STATE EXTENT CHECK (CRITICAL for "advanced" method)
    # ==========================================================================
    # The "advanced" oscillatory method factorizes the 2D integral as:
    #   I_out = moment_L * ∫[r_m,∞] χ_i χ_f / r^(L+1) dr
    # 
    # This is ONLY valid when the bound states u_i, u_f are negligible beyond r_m!
    # If significant bound state density exists beyond r_m, the factorization
    # produces WRONG results (factor of ~10x error for H 2s).
    #
    # Solution: Use the MAX of individual state extents (conservative estimate).
    # For H 1s→2s: 1s extends to ~4 a₀, 2s to ~13 a₀, so use 13 a₀.
    # ==========================================================================
    
    BOUND_STATE_THRESHOLD = 0.99
    w_trapz = grid.w_trapz if hasattr(grid, 'w_trapz') else np.gradient(r)
    
    # Find 99% extent for EACH bound state individually, then take max
    idx_99_max = 0
    for u_state in [u_i, u_f]:
        u_sq = u_state ** 2
        if np.sum(u_sq) > 1e-30:
            prob_cum = np.cumsum(u_sq * w_trapz)
            prob_cum /= (prob_cum[-1] + 1e-300)
            idx_99 = np.searchsorted(prob_cum, BOUND_STATE_THRESHOLD)
            idx_99_max = max(idx_99_max, idx_99)
    
    if idx_99_max > idx_limit:
        r_old = r[idx_limit - 1] if idx_limit > 0 else r[0]
        r_new = r[idx_99_max] if idx_99_max < len(r) else r[-1]
        logger.debug(
            "Bound state extent check: extending r_m from %.1f to %.1f a₀ "
            "(idx %d -> %d) to cover 99%% of bound state density.",
            r_old, r_new, idx_limit, idx_99_max + 1
        )
        idx_limit = idx_99_max + 1
    
    
    
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
    
    # Get phase shifts for analytical tail (if available)
    delta_i = cont_i.phase_shift if hasattr(cont_i, 'phase_shift') else 0.0
    delta_f = cont_f.phase_shift if hasattr(cont_f, 'phase_shift') else 0.0
    
    # Get Coulomb parameters for ionic targets
    eta_i = cont_i.eta if hasattr(cont_i, 'eta') else 0.0
    eta_f = cont_f.eta if hasattr(cont_f, 'eta') else 0.0
    eta_total = eta_i + eta_f
    sigma_i = cont_i.sigma_l if hasattr(cont_i, 'sigma_l') else 0.0
    sigma_f = cont_f.sigma_l if hasattr(cont_f, 'sigma_l') else 0.0
    
    k_total = k_i + k_f
    if use_oscillatory_quadrature:
        max_phase, is_ok, prob_idx = check_phase_sampling(
            r[:idx_limit], k_total, eta_total=eta_total
        )
        if not is_ok and idx_limit > 10:
            log_phase_diagnostic(
                r[:idx_limit], k_i, k_f, l_i, l_f, eta_total=eta_total
            )
    
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
                    n_nodes=CC_nodes, phase_increment=phase_increment,
                    eta_total=eta_total
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
                    n_nodes=CC_nodes, phase_increment=phase_increment,
                    eta_total=eta_total
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
                    n_nodes=CC_nodes, phase_increment=phase_increment, eta_total=eta_total
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
                n_nodes=CC_nodes, phase_increment=phase_increment,
                eta_total=eta_total
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

def _init_gpu_cc_weights(n_nodes: int = 5) -> None:
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


def _generate_gpu_filon_params(r_gpu, k_total, phase_increment=1.5708, n_nodes=5) -> Optional[dict]:
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

def _gpu_filon_direct(
    rho1_uw,
    int_r2,
    r_gpu,
    w_gpu,
    k_total,
    phase_increment=1.5708,
    n_nodes=5,
    precomputed=None,
    rho1_cc_pre=None,
    return_gpu=False
) -> float | cp.ndarray:
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
    if rho1_cc_pre is not None:
        rho1_cc = rho1_cc_pre
    else:
        rho1_cc = cp.interp(all_r_flat, r_gpu, rho1_uw).reshape(n_valid, n_nodes)
    int_r2_cc = cp.interp(all_r_flat, r_gpu, int_r2).reshape(n_valid, n_nodes)
    
    integrand = rho1_cc * int_r2_cc
    
    # Sum - stays on GPU if return_gpu=True
    result = cp.sum(integrand * weights_scaled)
    return result if return_gpu else float(result)



def _gpu_filon_exchange(
    kernel_L,
    rho1_uw,
    rho2_uw,
    r_gpu,
    w_gpu,
    k_total,
    phase_increment=1.5708,
    n_nodes=5,
    precomputed=None,
    rho1_cc_pre=None,
    rho2_cc_pre=None,
    return_gpu=False
) -> float | cp.ndarray:
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
        # Optimized path: use precomputed kernel at CC nodes if provided.
        if 'kernel_at_cc' in kernel_L:
            kernel_at_cc = kernel_L['kernel_at_cc']
        else:
            inv_gtr_at_cc = kernel_L['inv_gtr_at_cc']
            L = kernel_L['L']
            if L == 0:
                kernel_at_cc = inv_gtr_at_cc
            else:
                if 'ratio_at_cc' in kernel_L:
                    ratio_at_cc = kernel_L['ratio_at_cc']
                    kernel_at_cc = inv_gtr_at_cc * cp.power(ratio_at_cc, L)
                else:
                    log_ratio_at_cc = kernel_L['log_ratio_at_cc']
                    kernel_at_cc = inv_gtr_at_cc * cp.exp(L * log_ratio_at_cc)
    else:
        # Standard path: slice from full matrix
        kernel_at_cc = kernel_L[:, idx_left] * weight_left + kernel_L[:, idx_right] * weight_right
    kernel_interp = kernel_at_cc.reshape(n_r, n_valid, n_nodes)
    
    # OPTIMIZED: Pure GPU interpolation for rho2
    if rho2_cc_pre is not None:
        rho2_cc = rho2_cc_pre
    else:
        rho2_cc = cp.interp(all_r_flat, r_gpu, rho2_uw).reshape(n_valid, n_nodes)

    # Inner integral: for each r1, sum over CC nodes with weights
    inner_integrand = kernel_interp * rho2_cc[None, :, :]
    
    # Sum inner: (n_r,)
    int_r2_cc = cp.sum(inner_integrand * weights_scaled[None, :, :], axis=(1, 2))
    
    # OPTIMIZED: Pure GPU interpolation for outer integral
    if rho1_cc_pre is not None:
        rho1_cc = rho1_cc_pre
    else:
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
    gpu_memory_threshold: float = 0.8,
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
    
    # Very chatty hot-path debug is sampled by default; set DWBA_HOTPATH_DEBUG=1
    # for full per-call diagnostics.
    if _should_sampled_hotpath_debug("radial_me_entry", every=250, initial=2):
        logger.debug(
            "radial_ME_all_L_gpu ENTRY | N_grid=%d | V_core=%d, U_i=%d | "
            "bound_i.u=%d, bound_f.u=%d | cont_i.chi=%d (idx_m=%d), cont_f.chi=%d (idx_m=%d)",
            N_grid, len(V_core_array), len(U_i_array),
            len(bound_i.u_of_r), len(bound_f.u_of_r),
            len(cont_i.chi_of_r), cont_i.idx_match,
            len(cont_f.chi_of_r), cont_f.idx_match
        )
        if U_f_array is not None:
            logger.debug("radial_ME_all_L_gpu ENTRY | U_f=%d", len(U_f_array))
    
    # ==========================================================================
    # ARRAY SIZE VALIDATION (v2.9+: LOCAL adaptive mode safety)
    # ==========================================================================
    # In LOCAL adaptive mode, prep may have orbitals/potentials from a different
    # grid size. We validate all input arrays and truncate/error as needed.
    # ==========================================================================
    
    # Check V_core and U_i arrays
    if len(V_core_array) != N_grid:
        if len(V_core_array) > N_grid:
            logger.debug("Truncating V_core from %d to %d pts", len(V_core_array), N_grid)
            V_core_array = V_core_array[:N_grid]
        else:
            raise ValueError(f"V_core has {len(V_core_array)} pts but grid has {N_grid}")
    
    if len(U_i_array) != N_grid:
        if len(U_i_array) > N_grid:
            logger.debug("Truncating U_i from %d to %d pts", len(U_i_array), N_grid)
            U_i_array = U_i_array[:N_grid]
        else:
            raise ValueError(f"U_i has {len(U_i_array)} pts but grid has {N_grid}")
    
    # Check continuum waves
    if len(cont_i.chi_of_r) != N_grid:
        if len(cont_i.chi_of_r) > N_grid:
            logger.debug("Truncating cont_i.chi from %d to %d pts", len(cont_i.chi_of_r), N_grid)
            # Note: can't modify dataclass, slicing will be done at use site
        else:
            raise ValueError(f"cont_i.chi has {len(cont_i.chi_of_r)} pts but grid has {N_grid}")
    
    if len(cont_f.chi_of_r) != N_grid:
        if len(cont_f.chi_of_r) > N_grid:
            logger.debug("Truncating cont_f.chi from %d to %d pts", len(cont_f.chi_of_r), N_grid)
        else:
            raise ValueError(f"cont_f.chi has {len(cont_f.chi_of_r)} pts but grid has {N_grid}")
    
    # Check U_f if provided
    if U_f_array is not None and len(U_f_array) != N_grid:
        if len(U_f_array) > N_grid:
            logger.debug("Truncating U_f from %d to %d pts", len(U_f_array), N_grid)
            U_f_array = U_f_array[:N_grid]
        else:
            raise ValueError(f"U_f has {len(U_f_array)} pts but grid has {N_grid}")
    
    # Extract wave parameters for logic and tails
    k_i = cont_i.k_au
    k_f = cont_f.k_au
    l_i = cont_i.l
    l_f = cont_f.l
    k_total = k_i + k_f

    # NOTE: Also clamp to N_grid to handle grid size changes in LOCAL adaptive mode
    if hasattr(cont_i, 'idx_match') and 0 < cont_i.idx_match < N_grid:
        idx_limit = min(idx_limit, cont_i.idx_match + 1)
    if hasattr(cont_f, 'idx_match') and 0 < cont_f.idx_match < N_grid:
        idx_limit = min(idx_limit, cont_f.idx_match + 1)
    
    # Physics-based validation for match point
    k_eff = min(k_i, k_f) if min(k_i, k_f) > 0.1 else max(k_i, k_f)
    if k_eff > 1e-3:
        r_turn = (max(l_i, l_f) + 0.5) / k_eff
        idx_turn = np.searchsorted(r, r_turn)
        # CRITICAL: Clamp idx_turn to prevent MIN_IDX exceeding N_grid (LOCAL adaptive fix)
        idx_turn = min(idx_turn, N_grid - 20)  # Ensure room for +20 safety margin
        MIN_IDX = min(max(idx_turn + 20, int(N_grid * min_grid_fraction)), N_grid)
    else:
        MIN_IDX = max(1, int(N_grid * min_grid_fraction))
    
    if idx_limit < MIN_IDX:
        idx_limit = MIN_IDX

    # Mirror CPU split-point guardrails on GPU:
    # - bound-state extent coverage (for excitation split-tail formulas),
    # - asymptotic V_eff/kinetic validation at r_m.
    w_trapz = grid.w_trapz if hasattr(grid, "w_trapz") else np.gradient(r)
    bound_states_for_extent: List[np.ndarray] = [bound_i.u_of_r]
    if isinstance(bound_f, BoundOrbital):
        bound_states_for_extent.append(bound_f.u_of_r)
    idx_limit = _refine_idx_limit_physics(
        idx_limit=idx_limit,
        r=r,
        w_trapz=w_trapz,
        U_i_array=U_i_array,
        U_f_array=U_f_array,
        k_i=k_i,
        k_f=k_f,
        l_i=l_i,
        l_f=l_f,
        bound_states=bound_states_for_extent,
        context="GPU",
    )
    
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
    # NOTE: In LOCAL adaptive mode, bound orbitals may come from a PreparedTarget
    # with a DIFFERENT grid size. We must validate and truncate if necessary.
    u_i_raw = bound_i.u_of_r
    if len(u_i_raw) > N_grid:
        logger.debug("Truncating u_i from %d to %d pts (LOCAL adaptive grid change)", len(u_i_raw), N_grid)
        u_i_raw = u_i_raw[:N_grid]
    elif len(u_i_raw) < N_grid:
        raise ValueError(f"Bound state u_i has {len(u_i_raw)} pts but grid has {N_grid} - cannot interpolate")
    if gpu_cache is not None:
        u_i_full_gpu = gpu_cache.get_static_array(u_i_raw, "u_i", n_grid=N_grid)
    else:
        u_i_full_gpu = cp.asarray(u_i_raw)
    
    if isinstance(bound_f, BoundOrbital):
        u_f_raw = bound_f.u_of_r
        if len(u_f_raw) > N_grid:
            logger.debug("Truncating u_f from %d to %d pts (LOCAL adaptive grid change)", len(u_f_raw), N_grid)
            u_f_raw = u_f_raw[:N_grid]
        elif len(u_f_raw) < N_grid:
            raise ValueError(f"Bound state u_f has {len(u_f_raw)} pts but grid has {N_grid} - cannot interpolate")
        if gpu_cache is not None:
            u_f_full_gpu = gpu_cache.get_static_array(u_f_raw, "u_f_bound", n_grid=N_grid)
        else:
            u_f_full_gpu = cp.asarray(u_f_raw)
    elif hasattr(bound_f, 'chi_of_r'):
        u_f_raw = bound_f.chi_of_r
        if len(u_f_raw) > N_grid:
            u_f_raw = u_f_raw[:N_grid]
        elif len(u_f_raw) < N_grid:
            raise ValueError(f"Continuum u_f has {len(u_f_raw)} pts but grid has {N_grid}")
        # Ionization continuum can vary per call; avoid cache churn here.
        u_f_full_gpu = cp.asarray(u_f_raw)  # Ionization path
    else:
        u_f_full_gpu = u_i_full_gpu  # Fallback
    
    # Match point sliced data for standard integrals
    u_i_gpu = u_i_full_gpu[:idx_limit]
    u_f_gpu = u_f_full_gpu[:idx_limit]
    
    # Phase 4: Use cached continuum waves when gpu_cache is available
    if gpu_cache is not None:
        chi_i_full = gpu_cache.get_chi(cont_i, 'i')
        chi_f_full = gpu_cache.get_chi(cont_f, 'f')
        # Verify sizes match grid after get_chi truncation
        if len(chi_i_full) != N_grid:
            logger.warning("chi_i_full size mismatch: %d vs N_grid=%d (idx_limit=%d)", 
                          len(chi_i_full), N_grid, idx_limit)
            chi_i_full = chi_i_full[:N_grid] if len(chi_i_full) > N_grid else chi_i_full
        if len(chi_f_full) != N_grid:
            logger.warning("chi_f_full size mismatch: %d vs N_grid=%d (idx_limit=%d)", 
                          len(chi_f_full), N_grid, idx_limit)
            chi_f_full = chi_f_full[:N_grid] if len(chi_f_full) > N_grid else chi_f_full
        chi_i_gpu = chi_i_full[:idx_limit]
        chi_f_gpu = chi_f_full[:idx_limit]
    else:
        chi_i_gpu = cp.asarray(cont_i.chi_of_r[:idx_limit])
        chi_f_gpu = cp.asarray(cont_f.chi_of_r[:idx_limit])
    
    if gpu_cache is not None:
        V_diff_gpu = gpu_cache.get_v_diff(V_core_array, U_i_array, idx_limit)
    else:
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
    ratio_policy = _get_ratio_policy()
    matrix_bytes_standard = _matrix_bytes(idx_limit, idx_limit, itemsize=8)
    matrix_bytes_filon = _matrix_bytes(idx_limit, N_grid, itemsize=8)
    
    # Memory-based decision for matrix construction
    use_block_wise = False
    prefer_filon_full = False
    effective_free_mem = None
    mem_budget_bytes = None
    required_mem_estimate = 0
    if gpu_memory_mode == "block":
        use_block_wise = True
    elif gpu_memory_mode == "full":
        use_block_wise = False
        prefer_filon_full = use_filon
    else:  # "auto" - check GPU memory
        try:
            free_mem, _total_mem = cp.cuda.Device().mem_info

            # Include reusable bytes already reserved in CuPy pool.
            # Device free memory alone can be pessimistic after many iterations.
            mem_pool = cp.get_default_memory_pool()
            pool_total = mem_pool.total_bytes()
            pool_used = mem_pool.used_bytes()
            pool_reusable = max(0, pool_total - pool_used)
            effective_free = free_mem + pool_reusable
            effective_free_mem = int(effective_free)
            mem_budget = float(effective_free) * float(gpu_memory_threshold)
            mem_budget_bytes = mem_budget

            # If base kernel is already cached for this idx_limit, don't count it again.
            has_cached_base_kernel = (
                gpu_cache is not None and
                gpu_cache.inv_gtr is not None and
                gpu_cache.log_ratio is not None and
                gpu_cache.idx_limit >= idx_limit
            )
            standard_build_mem = 0 if has_cached_base_kernel else (matrix_bytes_standard * 3)
            standard_recursive_mem = 0
            if L_max > 0 and _should_enable_ratio_cache(
                ratio_policy,
                matrix_bytes_standard,
                mem_budget_bytes,
                "standard"
            ):
                # ratio cache + recursive working kernel
                standard_recursive_mem = matrix_bytes_standard * 2
            standard_required_mem = standard_build_mem + standard_recursive_mem

            # Memory estimation depends on mode:
            # - Standard: kernel build + optional recursive-ratio buffers
            # - Filon: extended kernel build + optional recursive-ratio buffers
            if use_filon:
                has_cached_filon_kernel = (
                    gpu_cache is not None and
                    gpu_cache.filon_inv_gtr is not None and
                    gpu_cache.filon_log_ratio is not None and
                    gpu_cache.filon_idx_limit >= idx_limit
                )
                filon_build_mem = 0 if has_cached_filon_kernel else (matrix_bytes_filon * 2)
                filon_recursive_mem = 0
                if L_max > 0 and _should_enable_ratio_cache(
                    ratio_policy,
                    matrix_bytes_filon,
                    mem_budget_bytes,
                    "filon"
                ):
                    # ratio cache + recursive working kernel
                    filon_recursive_mem = matrix_bytes_filon * 2

                # Conservative CC/workspace headroom for full Filon mode.
                filon_workspace_mem = matrix_bytes_standard
                filon_required_mem = filon_build_mem + filon_recursive_mem + filon_workspace_mem

                if filon_required_mem <= mem_budget:
                    prefer_filon_full = True
                    required_mem_estimate = filon_required_mem
                elif standard_required_mem <= mem_budget:
                    # Not enough for full Filon matrix, but enough for hybrid.
                    prefer_filon_full = False
                    required_mem_estimate = standard_required_mem
                else:
                    use_block_wise = True
                    required_mem_estimate = max(filon_required_mem, standard_required_mem)
            else:
                required_mem_estimate = standard_required_mem
                if required_mem_estimate > mem_budget:
                    use_block_wise = True

            if use_block_wise:
                logger.info(
                    "GPU memory limited (free %.1f GB + pool %.1f GB, need %.1f GB). Using block-wise.",
                    free_mem / 1e9,
                    pool_reusable / 1e9,
                    required_mem_estimate / 1e9
                )
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

    # Defensive: release reusable pool blocks before large matrix allocations
    # to avoid cumulative growth from previous calls exhausting device VRAM.
    mem_pool = cp.get_default_memory_pool()
    pool_reusable_pre = max(0, mem_pool.total_bytes() - mem_pool.used_bytes())
    if pool_reusable_pre > 256 * 1024 * 1024:  # >256 MB sitting unused in pool
        mem_pool.free_all_blocks()
    
    if not use_block_wise and (not use_filon or not prefer_filon_full):
        # Try to build/reuse full kernel matrix (works for BOTH Filon and standard).
        # IMPORTANT: avoid unconditional memory-pool flush in hot path; it introduces
        # frequent host-device synchronization and defeats allocator reuse.
        try:
            if gpu_cache is not None:
                gpu_cache.build_kernel_matrix(idx_limit)
                inv_gtr = gpu_cache.inv_gtr[:idx_limit, :idx_limit]
                log_ratio = gpu_cache.log_ratio[:idx_limit, :idx_limit]
            else:
                r_col = r_gpu[:, None]
                r_row = r_gpu[None, :]
                inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
                ratio = cp.minimum(r_col, r_row) * inv_gtr
                ratio = cp.minimum(ratio, 1.0 - 1e-12)
                log_ratio = cp.log(ratio + 1e-30)
                del r_col, r_row, ratio
            full_matrix_built = True
            if _should_sampled_hotpath_debug("standard_matrix_ready", every=200, initial=2):
                logger.debug("GPU: Standard matrix ready (%d×%d)", idx_limit, idx_limit)
        except Exception as e:
            # Catch OutOfMemoryError and Windows pagefile errors
            err_str = str(e).lower()
            if "memory" in err_str or "pagefile" in err_str or "out of memory" in err_str:
                logger.warning("GPU memory error: %s. Falling back to block-wise.", e)
                use_block_wise = True
                full_matrix_built = False
                cp.get_default_memory_pool().free_all_blocks()
            else:
                raise
    
    # For Filon, build extended matrix (idx_limit × N_grid) if memory permits
    # This allows full cp.dot() for int_r2_dir without block-wise loop
    filon_kernel_built = False
    filon_inv_gtr = None
    filon_log_ratio = None
    
    if use_filon and not use_block_wise and prefer_filon_full:
        try:
            if gpu_cache is not None:
                gpu_cache.build_filon_kernel_matrix(idx_limit)
                filon_inv_gtr = gpu_cache.filon_inv_gtr[:idx_limit, :]
                filon_log_ratio = gpu_cache.filon_log_ratio[:idx_limit, :]
            else:
                r_col = r_gpu[:, None]  # (idx_limit, 1)
                r_row_full = r_full_gpu[None, :]  # (1, N_grid)
                filon_inv_gtr = 1.0 / cp.maximum(r_col, r_row_full + 1e-30)
                filon_ratio = cp.minimum(r_col, r_row_full) * filon_inv_gtr
                filon_ratio = cp.minimum(filon_ratio, 1.0 - 1e-12)
                filon_log_ratio = cp.log(filon_ratio + 1e-30)
                del r_col, r_row_full, filon_ratio
            filon_kernel_built = True
            if _should_sampled_hotpath_debug("filon_matrix_build", every=200, initial=2):
                logger.debug("GPU: Filon extended matrix ready (%d×%d)", idx_limit, N_grid)
        except Exception as e:
            err_str = str(e).lower()
            if "memory" in err_str or "pagefile" in err_str or "out of memory" in err_str:
                logger.info("GPU: Filon extended matrix too large, using hybrid mode")
                filon_kernel_built = False
                cp.get_default_memory_pool().free_all_blocks()
            else:
                raise

    # Hybrid fallback: if full Filon matrix was preferred but not built, we still
    # need the standard kernel matrix for head-dot and exchange fallbacks.
    if use_filon and not use_block_wise and not filon_kernel_built and not full_matrix_built:
        try:
            if gpu_cache is not None:
                gpu_cache.build_kernel_matrix(idx_limit)
                inv_gtr = gpu_cache.inv_gtr[:idx_limit, :idx_limit]
                log_ratio = gpu_cache.log_ratio[:idx_limit, :idx_limit]
            else:
                r_col = r_gpu[:, None]
                r_row = r_gpu[None, :]
                inv_gtr = 1.0 / cp.maximum(r_col, r_row + 1e-30)
                ratio = cp.minimum(r_col, r_row) * inv_gtr
                ratio = cp.minimum(ratio, 1.0 - 1e-12)
                log_ratio = cp.log(ratio + 1e-30)
                del r_col, r_row, ratio
            full_matrix_built = True
            if _should_sampled_hotpath_debug("standard_matrix_hybrid_ready", every=200, initial=2):
                logger.debug("GPU: Standard matrix ready for hybrid fallback (%d×%d)", idx_limit, idx_limit)
        except Exception as e:
            err_str = str(e).lower()
            if "memory" in err_str or "pagefile" in err_str or "out of memory" in err_str:
                logger.warning("GPU hybrid fallback allocation failed: %s. Switching to block-wise.", e)
                use_block_wise = True
                full_matrix_built = False
                cp.get_default_memory_pool().free_all_blocks()
            else:
                raise
    
    # Filon params for oscillatory quadrature (only if use_filon)
    filon_params = None
    kernel_at_cc_pre = None
    if use_filon:
        filon_key = (
            int(idx_limit),
            float(np.round(k_total, 12)),
            float(np.round(phase_increment, 8)),
            int(CC_nodes),
        )
        if gpu_cache is not None:
            filon_params = gpu_cache.get_filon_params_cached(filon_key)

        if filon_params is None:
            filon_params = _generate_gpu_filon_params(r_gpu, k_total, phase_increment, CC_nodes)
            if gpu_cache is not None and filon_params is not None:
                gpu_cache.set_filon_params_cached(filon_key, filon_params)

        if filon_params:
            kernel_cc_key = filon_key
            if gpu_cache is not None:
                kernel_at_cc_pre = gpu_cache.get_kernel_cc_cached(kernel_cc_key)
            else:
                kernel_at_cc_pre = None

            if kernel_at_cc_pre is None:
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
                # Keep log-ratio by default; ratio cache is policy-driven and optional.
                kernel_at_cc_pre = {
                    'inv_gtr_at_cc': inv_gtr_at_cc,
                    'log_ratio_at_cc': log_ratio_at_cc
                }
                del r_col, r_cc_l, r_cc_r, inv_gtr_cc_l, inv_gtr_cc_r, log_ratio_cc_l, log_ratio_cc_r
                if gpu_cache is not None:
                    gpu_cache.set_kernel_cc_cached(kernel_cc_key, kernel_at_cc_pre)
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
    # Handle string "auto" or int values.
    # Avoid auto-tuning when no block-wise path is used (saves overhead and log noise).
    try:
        block_size_int = int(gpu_block_size) if gpu_block_size != "auto" else 0
    except (ValueError, TypeError):
        block_size_int = 0
    need_block_loops = use_block_wise or (use_filon and not filon_kernel_built and N_grid > idx_limit)
    BLOCK_SIZE = None
    if need_block_loops:
        BLOCK_SIZE = block_size_int if block_size_int > 0 else _compute_optimal_block_size(
            idx_limit,
            gpu_memory_threshold,
            effective_free_mem=effective_free_mem
        )
        if BLOCK_SIZE is None or BLOCK_SIZE <= 0:
            BLOCK_SIZE = 2048
    
    # Precompute rho2_eff_full ONCE before L-loop (optimization: was computed per-L)
    rho2_eff_full = (u_f_full_gpu * u_i_full_gpu) * w_full_gpu if use_filon else None
    
    # Precompute L=0 correction terms on GPU (avoid per-L sync)
    sum_rho2_dir = cp.sum(rho2_dir_w)
    sum_rho2_ex = cp.sum(rho2_ex_w)
    V_diff_dot_rho1_dir = cp.dot(rho1_dir_w, V_diff_gpu)
    V_diff_dot_rho1_ex = cp.dot(rho1_ex_w, V_diff_gpu)

    is_excitation = isinstance(bound_f, BoundOrbital)

    # Precompute bound multipole moments once on CPU to avoid per-L GPU syncs.
    moments_head = None
    moments_full = None
    if use_filon and is_excitation and idx_limit < N_grid - 10 and L_max >= 1:
        r_head = r[:idx_limit]
        r_full = r[:N_grid]
        w_head = grid.w_simpson[:idx_limit]
        w_full = grid.w_simpson[:N_grid]

        base_head = w_head * u_f_raw[:idx_limit] * u_i_raw[:idx_limit]
        base_full = w_full * u_f_raw[:N_grid] * u_i_raw[:N_grid]

        moments_head = np.zeros(L_max + 1, dtype=float)
        moments_full = np.zeros(L_max + 1, dtype=float)

        r_pow_head = np.ones_like(r_head)
        r_pow_full = np.ones_like(r_full)
        for Lm in range(1, L_max + 1):
            r_pow_head *= r_head
            r_pow_full *= r_full
            moments_head[Lm] = float(np.dot(base_head, r_pow_head))
            moments_full[Lm] = float(np.dot(base_full, r_pow_full))

    # Batch CPU outer-tail integrals for advanced/full_split mode.
    # This reduces per-L Python/closure overhead in the hot loop.
    outer_tail_batch = None
    outer_batch_moment_tol = 1e-12
    if use_filon and oscillatory_method != "legacy" and is_excitation and idx_limit < N_grid - 10 and moments_full is not None and L_max >= 1:
        min_active_batch, outer_batch_moment_tol = get_outer_batch_config()
        L_arr = np.arange(1, L_max + 1, dtype=int)
        moments_arr = moments_full[1:]
        n_active = int(np.count_nonzero(np.abs(moments_arr) > outer_batch_moment_tol))
        if n_active >= min_active_batch:
            try:
                t_outer_batch = time.perf_counter()
                outer_tail_batch = dwba_outer_integral_1d_multipole_batch(
                    moments_arr,
                    L_arr,
                    k_i, l_i, delta_i, eta_i, sigma_i,
                    k_f, l_f, delta_f, eta_f, sigma_f,
                    r_match, float(grid.r[-1]),
                    delta_phi=np.pi / 4
                )
                if _should_sampled_hotpath_debug("outer_tail_batch", every=100, initial=2):
                    logger.debug(
                        "Outer-tail batch ready: active=%d/%d, elapsed=%.3fs",
                        n_active, L_max, time.perf_counter() - t_outer_batch
                    )
            except Exception as e:
                logger.debug("Outer-tail batch failed, fallback to per-L path: %s", e)
                outer_tail_batch = None

    # Ratio policy: keep recursive power path only when memory/size are favorable.
    need_standard_kernel_for_direct = (not use_filon) or (use_filon and not filon_kernel_built)
    need_standard_kernel_for_exchange = not (
        use_oscillatory_quadrature and k_total > k_threshold and kernel_at_cc_pre is not None
    )
    need_standard_kernel = need_standard_kernel_for_direct or need_standard_kernel_for_exchange

    kernel_ratio = None
    kernel_L_work = None
    enable_standard_recursive = (
        full_matrix_built and need_standard_kernel and L_max > 0 and
        _should_enable_ratio_cache(
            ratio_policy,
            matrix_bytes_standard,
            mem_budget_bytes,
            "standard"
        )
    )
    if enable_standard_recursive:
        try:
            if gpu_cache is not None:
                kernel_ratio = gpu_cache.get_kernel_ratio(idx_limit)
            if kernel_ratio is None:
                kernel_ratio = cp.exp(log_ratio)
            kernel_L_work = inv_gtr.copy()
        except Exception as e:
            logger.debug("GPU standard recursive-ratio disabled: %s", e)
            enable_standard_recursive = False
            kernel_ratio = None
            kernel_L_work = None

    filon_ratio = None
    filon_kernel_work = None
    enable_filon_recursive = (
        filon_kernel_built and L_max > 0 and
        _should_enable_ratio_cache(
            ratio_policy,
            matrix_bytes_filon,
            mem_budget_bytes,
            "filon"
        )
    )
    if enable_filon_recursive:
        try:
            if gpu_cache is not None:
                filon_ratio = gpu_cache.get_filon_ratio(idx_limit)
            if filon_ratio is None:
                filon_ratio = cp.exp(filon_log_ratio)
            filon_kernel_work = filon_inv_gtr.copy()
        except Exception as e:
            logger.debug("GPU Filon recursive-ratio disabled: %s", e)
            enable_filon_recursive = False
            filon_ratio = None
            filon_kernel_work = None

    cc_ratio = None
    cc_kernel_work = None
    enable_cc_recursive = False
    if kernel_at_cc_pre is not None and L_max > 0:
        cc_shape = kernel_at_cc_pre['inv_gtr_at_cc'].shape
        cc_matrix_bytes = _matrix_bytes(cc_shape[0], cc_shape[1], itemsize=8)
        enable_cc_recursive = _should_enable_ratio_cache(
            ratio_policy,
            cc_matrix_bytes,
            mem_budget_bytes,
            "cc"
        )
        if enable_cc_recursive:
            try:
                cc_ratio = kernel_at_cc_pre.get('ratio_at_cc')
                if cc_ratio is None:
                    log_ratio_at_cc = kernel_at_cc_pre.get('log_ratio_at_cc')
                    if log_ratio_at_cc is not None:
                        cc_ratio = cp.exp(log_ratio_at_cc)
                if cc_ratio is not None:
                    cc_kernel_work = kernel_at_cc_pre['inv_gtr_at_cc'].copy()
                else:
                    enable_cc_recursive = False
            except Exception as e:
                logger.debug("GPU CC recursive-ratio disabled: %s", e)
                enable_cc_recursive = False
                cc_ratio = None
                cc_kernel_work = None

    if _should_sampled_hotpath_debug("ratio_policy_mode", every=200, initial=2):
        logger.debug(
            "GPU ratio policy=%s | recursive standard=%s filon=%s cc=%s | matrix MB std=%.1f filon=%.1f",
            ratio_policy,
            enable_standard_recursive,
            enable_filon_recursive,
            enable_cc_recursive,
            matrix_bytes_standard / (1024.0 * 1024.0),
            matrix_bytes_filon / (1024.0 * 1024.0),
        )

    # Drop stale ratio caches when recursive mode is not used in this call.
    if gpu_cache is not None:
        if not enable_standard_recursive:
            gpu_cache.ratio = None
        if not enable_filon_recursive:
            gpu_cache.filon_ratio = None
    if kernel_at_cc_pre is not None and not enable_cc_recursive:
        kernel_at_cc_pre.pop('ratio_at_cc', None)

    # Precompute CC interpolations reused across all L for Filon outer loops.
    rho1_dir_cc_pre = None
    rho1_ex_cc_pre = None
    rho2_ex_cc_pre = None
    if use_filon and filon_params:
        all_r_flat = filon_params['all_r_flat']
        n_valid = filon_params['n_valid']
        n_nodes = filon_params['n_nodes']
        rho1_dir_cc_pre = cp.interp(all_r_flat, r_gpu, rho1_dir_uw).reshape(n_valid, n_nodes)
        if use_oscillatory_quadrature and k_total > k_threshold:
            rho1_ex_cc_pre = cp.interp(all_r_flat, r_gpu, rho1_ex_uw).reshape(n_valid, n_nodes)
            rho2_ex_cc_pre = cp.interp(all_r_flat, r_gpu, rho2_ex_uw).reshape(n_valid, n_nodes)

    for L in range(L_max + 1):
        kernel_L = None
        int_r2_dir = None  # For Filon mode
        
        if full_matrix_built or filon_kernel_built:
            # FAST PATH: use prebuilt matrices.
            if full_matrix_built and need_standard_kernel:
                if L == 0:
                    kernel_L = inv_gtr
                else:
                    if enable_standard_recursive and kernel_ratio is not None and kernel_L_work is not None:
                        kernel_L_work *= kernel_ratio
                        kernel_L = kernel_L_work
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
                        if enable_filon_recursive and filon_ratio is not None and filon_kernel_work is not None:
                            filon_kernel_work *= filon_ratio
                            filon_kernel_L = filon_kernel_work
                        else:
                            filon_kernel_L = filon_inv_gtr * cp.exp(L * filon_log_ratio)
                    int_r2_dir = cp.dot(filon_kernel_L, rho2_eff_full)
                    if filon_kernel_L is not filon_kernel_work:
                        del filon_kernel_L
                else:
                    # HYBRID PATH: Use standard matrix for head + block-wise for tail
                    if kernel_L is None:
                        raise RuntimeError("Hybrid Filon path requires standard kernel_L, but it is not available.")
                    int_r2_dir = cp.dot(kernel_L, rho2_eff_full[:idx_limit])
                    
                    if N_grid > idx_limit:
                        r_col = r_gpu[:, None]
                        r_row_tail = r_full_gpu[idx_limit:][None, :]
                        tail_size = N_grid - idx_limit
                        block_step = BLOCK_SIZE if BLOCK_SIZE is not None else 2048
                        
                        for start in range(0, tail_size, block_step):
                            end = min(start + block_step, tail_size)
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
                block_step = BLOCK_SIZE if BLOCK_SIZE is not None else 2048
                for start in range(0, N_grid, block_step):
                    end = min(start + block_step, N_grid)
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
        
        if use_filon:
            # oscillatory_method: legacy, full_split, or advanced
            # int_r2_dir was already computed block-wise above to save memory
            # NOTE: _gpu_filon_direct returns a GPU scalar, not Python float
            I_dir_L = _gpu_filon_direct(
                rho1_dir_uw,
                int_r2_dir,
                r_gpu,
                w_gpu,
                k_total,
                phase_increment,
                CC_nodes,
                precomputed=filon_params,
                rho1_cc_pre=rho1_dir_cc_pre,
                return_gpu=True
            )
            
            if oscillatory_method == "legacy":
                # Add analytical tail if applicable (requires CPU computation)
                # For legacy mode, we still need float conversion for tail
                if L >= 1 and idx_limit < N_grid - 10 and is_excitation:
                    moment_L = moments_head[L] if moments_head is not None else 0.0
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
                    moment_L = moments_full[L] if moments_full is not None else 0.0
                    if abs(moment_L) > outer_batch_moment_tol:
                        if outer_tail_batch is not None:
                            I_out = outer_tail_batch.get(L, 0.0)
                        else:
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
            if kernel_at_cc_pre is not None:
                if L == 0:
                    k_spec = {'kernel_at_cc': kernel_at_cc_pre['inv_gtr_at_cc']}
                else:
                    if enable_cc_recursive and cc_ratio is not None and cc_kernel_work is not None:
                        cc_kernel_work *= cc_ratio
                        k_spec = {'kernel_at_cc': cc_kernel_work}
                    else:
                        k_spec = {**kernel_at_cc_pre, 'L': L}
            else:
                k_spec = kernel_L
            I_ex_L = _gpu_filon_exchange(
                k_spec, rho1_ex_uw, rho2_ex_uw, r_gpu, w_gpu, k_total, 
                phase_increment, CC_nodes, precomputed=filon_params,
                rho1_cc_pre=rho1_ex_cc_pre, rho2_cc_pre=rho2_ex_cc_pre,
                return_gpu=True
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
    # Conditional pool cleanup: only release when pool pressure is high.
    # Unconditional free_all_blocks() after every call kills GPU utilization
    # (forces cudaMalloc/cudaFree instead of fast pool reuse).
    # 70% of device VRAM threshold protects consumer GPUs from OOM while
    # preserving pool reuse across the ~60 (l_i, l_f) calls per energy.
    _pool = cp.get_default_memory_pool()
    _pool_total = _pool.total_bytes()
    if _pool_total > 0:
        # Cache device total VRAM to avoid repeated sync-inducing mem_info calls
        if not hasattr(radial_ME_all_L_gpu, '_dev_total_vram'):
            _, radial_ME_all_L_gpu._dev_total_vram = cp.cuda.Device().mem_info
        if _pool_total > 0.70 * radial_ME_all_L_gpu._dev_total_vram:
            _pool.free_all_blocks()

    return RadialDWBAIntegrals(I_L_direct=I_L_dir, I_L_exchange=I_L_exc)
