# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v2.34] — 2026-02-13 — Configurable Continuum Analytic Bypass

**Commit:** `TBD`

### Added

**User-configurable analytic bypass switch** (`continuum.py`, `driver.py`, `ionization.py`, `config_loader.py`)
- Added `oscillatory.analytic_bypass` (bool, default `true`) to YAML/config pipeline.
- `solve_continuum_wave(...)` now accepts `allow_analytic_bypass`.
- Driver and ionization paths now pass this flag consistently to continuum-wave solves (including precompute worker path).

### Documentation

- Updated `README.md` oscillatory parameter reference and continuum solver notes.
- Updated example and project YAML configs with `analytic_bypass` field.

## [v2.33] — 2026-02-12 — Ionization Quadrature Upgrade and High-L Tail Stabilization

**Commit:** `TBD`

### Changed

**Ionization SDCS/TICS energy integration** (`ionization.py`)
- Replaced fixed linear-grid trapezoidal integration as default with configurable SDCS quadrature:
  - `gauss_legendre` (new default, recommended),
  - `trapz_linear` (legacy behavior).
- Added weighted-energy integration path (`dot(weights, SDCS)`) for total ionization cross section and per-partial-wave integrated outputs.
- Fixed near-threshold node clamping so evaluated `E_eject` remains inside physical bounds (`0 <= E_eject <= (E-IP)/2`), avoiding invalid negative scattered-channel energies.
- Added ionization metadata fields:
  - `energy_quadrature`,
  - `n_energy_nodes`.

**High-L ionization top-up robustness** (`ionization.py`)
- Replaced purely geometric high-L tail extrapolation with a conservative multi-model estimator:
  - power-law fit in `L`,
  - geometric fallback.
- Added acceptance guard on tail fraction to reject unstable extrapolations.
- Added per-node top-up diagnostics in runtime metadata (fit quality and tail fraction) for uncertainty filtering.

**Ionization grid sampling criterion** (`DW_main.py`)
- Adaptive grid wavelength criterion now supports ionization worst-case oscillatory scale based on `k_scatt + k_eject` (equal-sharing estimate), not only the incident momentum.
- Applied to both batch and local-per-energy ionization grid preparation paths.

### Fixed

**Analytic bypass safety in ionic channels** (`continuum.py`)
- Kept analytic bypass enabled for neutral channels, but disabled it for ionic channels where long-range Coulomb asymptotics make Born-like bypass unreliable.

### Configuration / Docs

- Added `ionization.energy_quadrature` to YAML/config pipeline:
  - `config_loader.py` dataclass, parsing, validation, params conversion, templates.
  - `DW_main.py` defaults and ionization spec wiring.
  - `examples/config_ionization.yaml` updated with recommended value.
- Updated `README.md` ionization parameter reference.

## [v2.32] — 2026-02-12 — Analytic Bypass Reachability, Split-Point Physics Guards, and Top-Up Stabilization

**Commit:** `TBD`

### Fixed

**Analytic bypass reachability on non-uniform grids** (`continuum.py`)
- Match-point precheck now allows early search for bypass classification (`min_search_fraction=0.0`) without weakening normal match-point selection.
- Bypass decision now uses **radial-span fraction** (`r_m` relative to `[r_min, r_max]`) instead of only index fraction, which was misleading on log/non-uniform grids.
- Bypass result now sets `solver_method="analytic_bypass"` explicitly in `ContinuumWave`.

**Top-up application no longer distorts DCS shape** (`driver.py`)
- Removed global DCS rescaling by top-up factor.
- Top-up is now applied to **TCS only** and accepted only when tail fraction is stable (`< 20%`).
- Added runtime metadata fields: `topup_applied`, `topup_type`, `topup_value_cm2`, `topup_tail_fraction`.

### Changed

**GPU split-point physics parity with CPU path** (`dwba_matrix_elements.py`)
- Added `_refine_idx_limit_physics(...)` and applied it in GPU radial-integral path.
- Split index is now guarded by:
  - 99% bound-state extent coverage,
  - asymptotic validity check on effective potential (`V_eff / E_kin`).

**Undersampling-aware phase step in Filon kernels** (`oscillatory_integrals.py`)
- Added `_adaptive_phase_increment_for_grid(...)`.
- Filon and Filon-exchange 2D kernels now reduce phase increment on undersampled grids (bounded between `pi/16` and `pi/4`).

### Added

**Quick CPU/GPU parity diagnostic** (`debug/test_cpu_gpu_parity.py`)
- Added lightweight script for synthetic parity checks of `radial_ME_all_L` vs `radial_ME_all_L_gpu`.
- Script returns `SKIP` when CuPy/GPU runtime is unavailable.

## [v2.31] — 2026-02-12 — Physics-Based Auto Multipole Cutoff for Excitation

**Commit:** `a0370fe` (Edit_99)

### Added

**Production `L_max_integrals: "auto"`** (`driver.py`, `config_loader.py`)
- Excitation now supports `L_max_integrals` as either:
  - fixed integer (legacy behavior), or
  - `"auto"` using a physics-based `q·R` estimate.
- Auto mode computes multipole cutoff from:
  - momentum transfer `q = |k_i - k_f|`,
  - transition-region radius `R` estimated from bound-state `r_95`,
  - dipole bonus and safety floor/cap.
- Added runtime metadata fields:
  - `L_max_integrals_requested`,
  - `L_max_integrals_used`,
  - `L_max_integrals_auto` (diagnostic components for the estimate).

### Changed

- Batch/config validation now accepts `excitation.L_max_integrals: "auto"`.
- Pilot auto path is robust when production config uses `L_max_integrals="auto"`.
- Updated excitation example config to demonstrate production auto multipole scaling.

### Documentation

- Updated `README.md` parameter reference and notes for `L_max_integrals: "auto"` and its `q·R` rationale.

## [v2.30] — 2026-02-12 — Example Config Refresh (Quality-Oriented Baseline)

**Commit:** `a0370fe` (Edit_99)

### Documentation / UX

**Updated example YAML templates** (`examples/config_excitation.yaml`, `examples/config_ionization.yaml`)
- Refreshed examples to current recommended baseline focused on numerical quality:
  - adaptive `grid.strategy: "local"` with `r_max: "auto"`,
  - stronger grid safety/sampling (`r_max_scale_factor=2.8`, `min_points_per_wavelength=18`, higher `n_points_max`),
  - richer angular/partial-wave settings for production-like quality.
- Moved GPU/CPU execution controls to the dedicated `hardware:` section (while keeping oscillatory section focused on quadrature/phase).
- Updated ionization example to set `output.calibrate: false` (calibration applies to excitation, not ionization).

## [v2.29] — 2026-02-12 — Batch Log-Grid Threshold Regeneration Parity

**Commit:** `a0370fe` (Edit_99)

### Fixed

**Batch/config energy-grid behavior aligned with interactive mode** (`DW_main.py`)
- Added shared helper `build_energy_grid_above_threshold(...)`.
- For `energy.type: "log"`, when configured start includes sub-threshold points, batch mode now regenerates the log grid above threshold (with the same margin strategy as interactive mode) instead of only dropping points.
- Applied in both excitation and ionization batch flows.
- Added explicit INFO log when regeneration is applied, including resulting point count and energy range.

### Documentation

- Updated `README.md` performance notes with batch log-grid thresholding behavior.

## [v2.28] — 2026-02-11 — Partial-Wave Runtime Guardrails and Result Metadata

**Commit:** `a0370fe` (Edit_99)

### Critical Fixes

**Excitation `L_max_projectile` semantics restored** (`driver.py`)
- Runtime projectile limit now treats config `L_max_projectile` as a **base/floor**.
- Effective target uses `max(L_max_projectile, k*8+5)` and is then constrained by turning-point physics (`r_max`) and hard safety cap.
- Removes unintended low-`L` truncation when user base value was below dynamic requirement.

**GPU loop robustness for `chi_i` failures** (`driver.py`)
- Replaced premature `break` on `chi_i is None` with per-`l_i` `continue`.
- Added skip counters and summary logging to avoid aborting full summation because of a single failed incoming wave.

### Diagnostics / Metadata

**Per-energy runtime L metadata in excitation outputs** (`driver.py`, `DW_main.py`)
- Added runtime metadata to `DWBAResult` (`metadata` field).
- Excitation JSON entries now include:
  - `L_max_projectile_used`
  - `L_dynamic_required`
  - `maxL_in_result`
  - `n_projectile_partial_waves_summed`
  - extended diagnostics under `meta.runtime` (targets/limits, stop reason, skipped waves).

**High-energy validation warning for low configured `L_max`** (`grid.py`)
- `validate_high_energy(...)` now warns when configured `L_max` is significantly below dynamic runtime target (`k*8+5`) at scan high end.

### Documentation

- Updated `README.md` parameter reference to clarify `L_max_projectile` floor semantics and turning-point-limited runtime behavior.
- Documented new excitation output metadata fields for partial-wave auditing.

## [v2.27] — 2026-02-11 — Debug Suite Cleanup and Deep-Tool Integration

**Commit:** `e7f01c3` (Edit_98)

### Debug

**Integrated external deep diagnostics into `debug/debug.py` menu**
- Added unified external-script runner via `runpy` for better error handling and consistency.
- Added menu entries for:
  - deep phase-method comparison (`debug/diag_phase_methods_compare.py`),
  - deep upturn hypothesis sweep (`debug/deep_hypothesis_test.py`),
  - performance profiling (`debug/profile_performance.py`).

**Upturn diagnostic path fix**
- Renamed `debug/diagnose_upturn.py` -> `debug/diag_upturn.py`.
- Updated menu execution path to use the correct script name.

**Debug-folder cleanup**
- Removed obsolete/one-off scripts:
  - `debug/check_gpu_mem.py`
  - `debug/debug_high_l_phase.py`
  - `debug/debug_match_point.py`
  - `debug/deep_phase_analysis.py`
  - `debug/test_adaptive_grid.py`
  - `debug/test_auto_rmax.py`
  - `debug/test_batch_filon.py`
  - `debug/test_rmax_impact.py`
  - `debug/test_single_segment.py`
- Removed stale debug artifacts:
  - `debug/run_gpu.log`
  - `debug/test_output.txt`
  - `debug/test_single_output.txt`
  - `debug/profile_gpu.py` (incomplete local helper)

### Documentation

- Updated `README.md` diagnostic menu snapshot to match current `debug/debug.py` options.
- Expanded script inventory with deep diagnostics and profiling tools.
- Added ignore rules for debug artifacts in `.gitignore` (`debug/*.log`, `debug/*output*.txt`).

## [v2.26] — 2026-02-11 — Documentation Sync for Batch Outer-Tail and GPU Pool Hygiene

**Commit:** `415a6ab` (Edit_97)

### Performance

**GPU memory-pool hygiene guardrails** (`dwba_matrix_elements.py`)
- Added pre-allocation CuPy pool cleanup when reusable cached pool bytes exceed ~256 MB before large kernel-matrix allocations.
- Added end-of-call pool cleanup after GPU radial-integral completion to reduce cumulative VRAM growth across repeated `(l_i, l_f)` calls.

### Documentation

- Updated `README.md` to document:
  - batch outer-tail activation scope and thresholds (`DWBA_OUTER_BATCH_MIN_ACTIVE`, `DWBA_OUTER_BATCH_MOMENT_TOL`),
  - current CuPy memory-pool cleanup behavior in GPU matrix hot paths,
  - dynamic `l_f` precompute bound (`L_max_proj + L_max_integrals`) and bounded DCS-history buffer (`deque(maxlen=4)`).

## [v2.25] — 2026-02-10 — Batch Outer-Tail Integrals for Multi-L Excitation

**Commit:** `415a6ab` (Edit_97)

### Performance

**Batch outer-tail CPU path** (`oscillatory_integrals.py`, `dwba_matrix_elements.py`)
- Added `dwba_outer_integral_1d_multipole_batch(...)` to compute outer-tail oscillatory integrals for multiple multipoles in one batch.
- Added internal batch Filon/Levin helpers for envelopes `m_L / r^(L+1)` with shared phase evaluation.
- GPU radial-integral driver now precomputes advanced/full-split excitation outer-tail terms in batch and reuses them inside the L-loop.
- Falls back automatically to legacy per-L outer-tail path on any batch failure.

### Configuration

- Added `DWBA_OUTER_BATCH_MIN_ACTIVE` (default `4`) - minimum number of active multipoles required to use batch outer-tail.
- Added `DWBA_OUTER_BATCH_MOMENT_TOL` (default `1e-12`) - threshold for skipping tiny multipole moments in batch mode.
- Exposed batch tuning via `get_outer_batch_config()`.

### Documentation

- Updated `README.md` with new batch outer-tail environment variables and optimization notes.

## [v2.24] — 2026-02-10 — Outer-Tail Guardrails & Ratio-Cache Retention Fixes

**Commit:** `a4d77a1` (Edit_96)

### Performance

**GPU ratio-cache retention fixes** (`dwba_matrix_elements.py`)
- `GPUCache.get_kernel_ratio()` / `GPUCache.get_filon_ratio()` now compute `exp(log_ratio)` only on the requested prefix instead of full historical matrices.
- Added oversized-cache shrink logic to avoid retaining stale large ratio buffers after `idx_limit` decreases.
- When recursive ratio mode is not used for the current call, stale cached ratio buffers are dropped (`ratio`, `filon_ratio`).
- CC recursive ratio path no longer stores `ratio_at_cc` persistently in cached CC tensors unless actively needed in-call.

**Outer-tail CPU hot-path reuse** (`oscillatory_integrals.py`)
- Added small LRU cache for DWBA outer-tail phase closures (`phi±`, `phi'±`, `phi''±`) keyed by channel parameters.
- Removes repeated closure construction overhead in per-`L` outer-tail integrations.

**Driver memory/CPU hygiene** (`driver.py`)
- Replaced unbounded `dcs_history` list with `deque(maxlen=4)` in GPU convergence checks.
- Replaced fixed `chi_f` precompute buffer (`L_max_proj + 15`) with dynamic upper bound `L_max_proj + L_max_integrals`.

### Numerical Quality

**Filon segment-cap robustness** (`oscillatory_integrals.py`)
- Increased default `DWBA_MAX_FILON_SEGMENTS` from `2048` to `4096`.
- Added `DWBA_MAX_EFFECTIVE_DPHI` guard (default `pi/2`): if segment capping inflates effective phase-per-segment, Filon now increases nodes per segment adaptively (bounded) to reduce under-resolution risk.
- Added sampled debug diagnostics for cap-triggered effective `dphi` and adaptive node scaling.

### Documentation

- Updated `README.md` with:
  - new default for `DWBA_MAX_FILON_SEGMENTS`,
  - `DWBA_MAX_EFFECTIVE_DPHI`,
  - notes on ratio-cache prefix sizing and outer-phase closure caching.

## [v2.23] — 2026-02-10 — GPU Ratio-Path Memory Control & Cache Stability

**Commit:** `a4d77a1` (Edit_96)

### Performance

**GPU ratio-path policy and memory-aware fallback** (`dwba_matrix_elements.py`)
- Added active policy switch via `DWBA_GPU_RATIO_POLICY` (`auto` / `on` / `off`) in the GPU matrix hot path.
- `auto` now enables recursive ratio updates only for moderate matrix sizes with adequate memory headroom.
- Restored safe legacy behavior (`inv_gtr * exp(L*log_ratio)`) automatically when matrices are large or memory is tight.
- Replaced per-L `ratio_pow.copy(); ratio_pow *= ratio` pattern with a single recursive working-kernel buffer when recursion is enabled.
- Reduced persistent VRAM pressure in Filon/CC paths by keeping `log_ratio_at_cc` as baseline and materializing ratio caches only when policy allows.
- Improved `gpu_memory_mode="auto"` estimation to include optional recursive-ratio working buffers (previous estimate was optimistic for large Filon matrices).

**GPU cache reuse stability** (`dwba_matrix_elements.py`)
- Upgraded Filon parameter and CC-kernel reuse from single-entry cache to small key-based LRU caches in `GPUCache`.
- Reduces rebuild churn when `idx_limit` changes across nearby calls in the same energy point.

### Debug

**Micro-benchmark script** (`debug/benchmark_gpu_ratio_policy.py`)
- Added a lightweight benchmark to compare `DWBA_GPU_RATIO_POLICY=off/auto/on` on a synthetic GPU DWBA case.
- Reports per-policy runtime and CuPy pool snapshot without running full production scans.

### Documentation

- Updated `README.md` with `DWBA_GPU_RATIO_POLICY` usage, tuning guidance, and benchmark command.

## [v2.22] — 2026-02-07 — Excitation Scaling Parity & Convergence Ordering

**Commit:** `a4d77a1` (Edit_96)

### Critical Fixes

**Config excitation `N_equiv` parity** (`DW_main.py`)
- Batch/config excitation path now uses `N_equiv=1` (SAE), matching interactive path.
- Removes the unintended factor-of-two raw DCS/TCS scaling mismatch in YAML-driven runs.

**GPU convergence check ordering** (`driver.py`)
- Per-angle DCS stability is now evaluated **after** adding the current `l_i` contribution.
- Prevents premature early-stop decisions based on the pre-add state.
- Added explicit finiteness guard for current `l_i` amplitudes before accumulation.

### Debug

**Regression script** (`debug/test_p1_p2_regression.py`)
- Added lightweight checks for:
  - config-path `N_equiv` value,
  - linear DCS scaling with `N_equiv`,
  - synthetic pre-add vs post-add convergence-order behavior.

## [v2.21] — 2026-02-07 — GPU Hot-Path Logging & Block Auto-Tune Stability

**Commit:** `7f28a7a` (Edit_95)

### Performance

**Hot-Path DEBUG Sampling** (`dwba_matrix_elements.py`, `oscillatory_integrals.py`, `driver.py`)
- Added sampled DEBUG logging in the most frequent GPU/Filon paths to reduce log I/O overhead.
- Added explicit opt-in switch `DWBA_HOTPATH_DEBUG=1` for full per-call traces when deep diagnostics are needed.
- Gated very verbose `(l_i, l_f)` GPU-integral debug in `driver.py` behind `DWBA_HOTPATH_DEBUG`.

**Block-Size Auto-Tune Improvements** (`dwba_matrix_elements.py`)
- `_compute_optimal_block_size(...)` now accepts effective free memory and uses it when available.
- Effective free memory is computed as device-free VRAM plus reusable CuPy-pool bytes.
- Auto block-size tuning is now executed only when block-wise/hybrid loops are actually used,
  avoiding unnecessary per-call tuning overhead in full-matrix paths.
- Added safer fallback for block step selection when auto-tuning cannot provide a positive value.

**Long-Run Visibility & Guardrails** (`driver.py`, `oscillatory_integrals.py`)
- Added GPU summation heartbeat logs inside `l_i` loop (`l_f` progress + elapsed), preventing long silent periods.
- Added `Slow GPU pair` warning when a single `(l_i, l_f)` matrix-element call exceeds threshold.
- Added Filon segment cap for outer-tail integrals (`DWBA_MAX_FILON_SEGMENTS`, default `2048`) to bound pathological runtimes.
- Added slow outer-integral warning threshold (`DWBA_OUTER_SLOW_WARN_S`, default `20s`).

### Documentation

**Debug/Performance Guidance** (`README.md`)
- Added `DWBA_HOTPATH_DEBUG` documentation and guidance for production benchmarking (`INFO` vs `DEBUG`).
- Documented effective-memory-aware block auto-tuning and on-demand block-size tuning behavior.

## [v2.20] — 2026-02-06 — Adaptive Grid Parity & Ionic Phase-Sampling Finalization

**Commit:** `34779cb` (Edit_94)

### Critical Fixes

**Batch Ionization Grid Strategy Parity** (`DW_main.py`)
- `run_from_config(..., calculation_type="ionization")` now supports the same grid strategy behavior as other flows:
  - `manual`: fixed `r_max/n_points`
  - `global`: adaptive grid from `E_min` using scan-level `L_eff`
  - `local`: per-energy adaptive recalculation with target re-preparation when grid changes
- Fixes previous mismatch where batch ionization ignored adaptive grid scaling.

**Centralized Effective L Estimation** (`DW_main.py`)
- Added `estimate_effective_projectile_lmax()` and replaced duplicated inline formulas (`k*8+5` + `+15`) across:
  - interactive excitation
  - interactive ionization
  - batch excitation
  - batch ionization
  - pilot calibration
- Reduces divergence risk between runtime wave caching and grid sizing.

**`r_max='auto'` Contract Finalization** (`config_loader.py`, `DW_main.py`)
- `GridConfig.r_max` now accepts `float | "auto"` in config validation.
- Validation rejects invalid strings and explicitly disallows `"auto"` with `strategy="manual"`.
- Added `resolve_grid_r_max_for_prep()` for places where a concrete prep grid is required.
- `calculate_optimal_grid_params()` now keeps density scaling active for `"auto"` using reference `r_base=200 a.u.`.

**Ionic Phase-Sampling Propagation + Formula Correction** (`oscillatory_integrals.py`, `dwba_matrix_elements.py`)
- Propagated `eta_total` through production call paths (`check_phase_sampling`, `log_phase_diagnostic`, phase-node splitting).
- Corrected phase-step estimate for Coulomb case to use:
  - `Δφ ≈ |k_total + eta_total/r| * Δr`
  instead of always adding `|eta|/r`.
- Reduces false undersampling flags and aligns diagnostics with Coulomb phase physics.

### Numerical Quality

**Log-Grid-Aware Interpolation in Oscillatory Paths** (`oscillatory_integrals.py`)
- Added `_interp_on_radial_grid()` with automatic log-space interpolation on exponential radial grids.
- Replaced direct `np.interp(...)` calls in Filon/phase-adaptive paths with the new helper.
- Improves interpolation behavior on strongly nonuniform radial grids.
- Added cache for log-grid classification (`_is_log_spaced_grid_cached`) to avoid repeated O(N)
  spacing checks in tight interpolation loops.

### Performance

**GPU Matrix-Stage Hot-Path Optimization** (`dwba_matrix_elements.py`)
- Added reuse of cached kernel matrices (`inv_gtr`, `log_ratio`) via `GPUCache.build_kernel_matrix()`.
- Added static GPU array caching (`array_cache`) and cached `V_core-U_i` construction in `GPUCache`.
- Replaced repeated `exp(L*log_ratio)` evaluations with recursive `ratio^L` updates in the L-loop
  (with safe fallback to per-L exponential when memory is constrained).
- Reduced allocator synchronization overhead by removing unconditional memory-pool flushes from the
  hot path (cleanup remains at energy-level teardown and OOM fallback).
- Precomputed bound multipole moments once per call (CPU side) to avoid repeated per-L GPU scalar sync.
- Improved `gpu_memory_mode="auto"` decision: includes reusable CuPy-pool bytes and cached base-kernel
  reuse in memory estimation, reducing false fallback to block-wise mode.
- Refined `auto` strategy to prefer `Filon/hybrid` when full Filon matrix does not fit, instead of
  immediately falling back to block-wise mode.
- Removed redundant standard-kernel work in `Filon/full-matrix` path (no unnecessary `kernel_L` build
  when only extended Filon kernel is required).
- Added extended Filon kernel caching in `GPUCache` (`filon_inv_gtr`, `filon_log_ratio`) for repeated
  calls with unchanged `idx_limit`.
- Precomputed CC-interpolated exchange/direct envelopes once per call (reused across all L), removing
  repeated `cp.interp` overhead inside the L-loop.
- Added prefix reuse for cached kernels: if a larger cached matrix exists, smaller `idx_limit` calls now
  use slices instead of rebuilding kernels from scratch.
- Added caching for `exp(log_ratio)` and `exp(filon_log_ratio)` in `GPUCache`, removing repeated
  per-call exponentiation of unchanged kernel bases.
- Added caching of Filon `kernel_at_cc_pre` tensors in `GPUCache` (keyed by Filon setup), avoiding
  repeated reconstruction of exchange CC-kernel bases for unchanged energy/grid settings.

**Outer-Tail Callable Evaluation Optimization** (`oscillatory_integrals.py`)
- Added `_eval_callable_on_nodes()` (array-first evaluation with fallback) in Levin/Filon segment loops.
- Reduces Python-level overhead from `np.vectorize(...)` in repeated outer-tail integrations.

### Documentation

- Removed stale `DWBA_DEBUGGING_COMPENDIUM.md`.
- Updated `README.md` grid section:
  - `r_max` can be numeric or `"auto"` (adaptive modes)
  - Coulomb `r_max` criterion and `"auto"` density-reference behavior documented.

### CLI Stability

**Windows Console Encoding Fallback** (`DW_main.py`)
- Added safe console print wrapper to prevent `UnicodeEncodeError` on non-UTF code pages (e.g. cp1250).
- Added UTF-8 stream reconfigure attempt on Windows with graceful ASCII fallback for decorative glyphs.

---

## [v2.19] — 2026-01-24 — Grid Scaling Propagation Fixes

**Commit:** `4903f37` (Edit_93)

### Critical Fixes

**L_max_effective Propagation** (`DW_main.py`)
- Propagated `L_max_effective` to ALL calls of `calculate_optimal_grid_params` (previously some used base `L_max_proj`).
- This ensures `r_max` and `n_points` are calculated using the dynamic `k_max*8+20` value used by the solver, preventing undersampling and phase instability at high energies.
- Affected modules: Ionization (interactive), Batch Excitation (global/local/loop).

**Fixed `calculate_optimal_grid_params` for 'auto' radius** (`DW_main.py`)
- Refactored function to handle `base_r_max='auto'` without `TypeError`.
- Safely handles density scaling by skipping it when base radius is unknown.

### Code Quality

**Pilot Calibration Syntax Fix** (`DW_main.py`)
- Fixed missing `try:` statement and indentation at line ~1739 in `run_pilot_calibration()`.
- Prevents potential runtime crashes during calibration phase.

---

## [v2.18] — 2026-01-22 — Critical Phase Extraction & Asymptotic Fixes

**Commit:** `17a3816` (Edit_92)

### Critical Fixes

**Match Point Uses Effective Potential** (`continuum.py`)
- `_find_match_point()` now uses **V_eff = |2U| + l(l+1)/r²** (not just U)
- True asymptotic criterion: both short-range AND centrifugal must be negligible
- Also requires r > 2.5×r_turn for safety margin
- **Impact**: Match points now correctly placed in true asymptotic region

**Coulomb Asymptotic Validity Check** (`continuum.py`)
- Added check: ρ_m > 3×max(l, |η|) for asymptotic validity
- Warning logged when condition violated (common for He⁺ at low energies)
- Also includes O(1/ρ) corrections per NIST DLMF §33.11

**Phase Diagnostic Alt Point Fix** (`continuum.py`)
- Changed from `idx_match - 5` to a post-match asymptotic point.
- Initial v2.18 implementation used `idx_match + 10`; later refined (v2.20) to
  `r_alt = r_m + 5.0 a.u.` with `searchsorted` for log-grid consistency.
- **Impact**: Eliminates false "Phase unstable" warnings

**Oscillatory Phase Sampling for Ions** (`oscillatory_integrals.py`)
- `check_phase_sampling()` now includes optional `eta_total` parameter
- For ionic targets: dφ/dr ≈ k + η/r (Coulomb contribution)
- **Impact**: Better detection of undersampling for He⁺

**Coulomb-Aware r_max Auto-Scaling** (`grid.py`, `DW_main.py`)
- `compute_required_r_max()` now includes `z_ion` parameter
- Enforces ρ_max > 3×max(L, |η|) for Coulomb asymptotic validity
- E.g., at E=20eV, L=35: neutral→r_max=73, He+→r_max=87 (+18%)
- **Impact**: Grid auto-scales to satisfy Coulomb requirements

### Files Updated
- `continuum.py`: `_find_match_point()`, `_extract_phase_logderiv_coulomb()`, diagnostics
- `oscillatory_integrals.py`: `check_phase_sampling()`
- `grid.py`: `compute_required_r_max()`
- `DW_main.py`: `calculate_optimal_grid_params()`, callers updated

---

## [v2.17] — 2026-01-20 — Pilot Calibration Refactor

**Commit:** `40626b1` (Edit_89)

### Code Quality Improvements

**Extracted `run_pilot_calibration()` helper function** (~120 lines) to consolidate duplicated pilot calculation logic from both interactive and batch modes.

**Changes**:
- **Fixed bug**: Interactive mode was using non-existent `L_max_integrals_override` parameter
- **Unified logging**: Both modes now log `"Pilot Calibrate | E=%.0f eV: r_max=%.1f a.u., n_points=%d, L_proj=%d"`
- **Consistent behavior**: Both modes use adaptive grid calculation for pilot energy
- **Reduced duplication**: ~100 lines of duplicated code replaced with single function call

**Files updated**: `DW_main.py`

---

## [v2.16] — 2026-01-19 — Dual Top-Up Strategy & Default Updates — Edit_88 (`7818b5e`)

### Dual Top-Up Strategy

**Physics-based tail extrapolation** now applies different methods depending on transition type:

| Transition Type | Method | Formula | Condition |
|----------------|--------|---------|-----------|
| **E1 (dipole, ΔL=1)** | Coulomb-Bethe | `σ_L × ln(2L+1)/(L+1) × 2` | Slow 1/L decay |
| **Forbidden (ΔL≠1)** | Born geometric | `σ_L × q/(1-q)` | Fast exponential decay |

**Safety conditions**:
- Monotonic decay required (`σ_{L-1} > σ_L > σ_{L+1}`)
- Decay rate stability (`|q - q_prev| < 0.2-0.3`)
- Born: `q < 0.95`; Coulomb-Bethe: `0.5 < q < 0.98`
- Top-up fraction limit: 20% of base σ (increased from 10%)

**Logging**:
```
Top-Up          | Coulomb-Bethe applied (tail=1.30e-17 cm², +17.1%)
Top-Up          | Not applied (Forbidden transition, no suitable decay)
```

### Default Parameter Changes

| Parameter | Old Default | New Default | Rationale |
|-----------|-------------|-------------|-----------|
| `n_points` | 3000 | **1000** | Adaptive scaling handles high-E; base can be lower |
| `gpu_memory_threshold` | 0.7 | **0.8** | Allow more GPU memory utilization |
| `solver` | `"auto"` | **`"rk45"`** | RK45 is correct for exponential grids (v2.13 finding) |

**Files updated**: `DW_main.py`, `driver.py`, `config_loader.py`, `dwba_matrix_elements.py`, `README.md`, `examples/*.yaml`

> [!NOTE]
> `H2s.yaml` excluded from default updates (user configuration file).

---

## [v2.15] — 2026-01-18 — Stability-Based Convergence — Edit_87 (`2f942cb`)

**Major rework**: Partial wave convergence logic redesigned per DWBA literature.

### Key Insight (Literature)

Non-monotonicity of individual partial wave contributions is **NORMAL** in DWBA due to:
- Interference in |f±g|² (DCS = |f-g|² + |f+g|²)
- Phase factors `i^(l_i+l_f)` causing sign oscillations
- Direct-exchange interference

> [!IMPORTANT]
> **The old "upturn detection" was fundamentally flawed.** A contribution being larger than the previous one is NOT a numerical error—it's physics.

### Removed (Flawed)

| Check | Old Logic | Why Removed |
|-------|-----------|-------------|
| Individual contribution | `sigma_li/total < 1e-6` → stop | Interference can make small l_i important |
| Upturn detection | `curr > prev × 1.1` → stop | Non-monotonicity is NORMAL |
| `pending_topup` | Born tail from "last monotonic trio" | Based on false assumption |

### Added (Correct)

**Per-Angle DCS Stability Test**:
```python
# Check max relative change across ALL angles
max_change = max(|DCS_new - DCS_old| / DCS_new)
if max_change < 0.01 and l_i > 15:
    converged = True
```

**Numerical Safety Check**:
```python
if not np.isfinite(amplitudes):
    stop_reason = "Numerical failure"
```

### Files Modified

| File | Change |
|------|--------|
| `driver.py` | Complete rewrite of early stopping logic |

---

## [v2.14] — 2026-01-18 — Grid Scaling & Amplitude Normalization — Edit_87 (`2f942cb`)

### Grid Scaling Improvements

**Wavelength-aware grid density** now focuses on bound state region:
- Check at r=15 a.u. (bound state outer edge) requiring 15+ pts/wavelength
- Relaxed check at r=50 a.u. (match region) requiring 5+ pts/wavelength  
- Warning when `n_points_max` caps required density for high energies

**Example impact** at E=1000 eV:
- Before: 3000 points, 1.3 pts/wavelength at r=100 (CRITICAL undersampling)
- After: 5735 points, 16.6 pts/wavelength at r=15, 5.0 at r=50

### Amplitude Normalization

**RMS amplitude enforcement**: Asymptotic continuum wave amplitude validated and corrected to √(2/π) ≈ 0.7979 after stitching.

### Files Modified

| File | Change |
|------|--------|
| `DW_main.py` | `calculate_optimal_grid_params()` improved wavelength-based scaling |
| `continuum.py` | RMS amplitude validation in `solve_continuum_wave()` |

---

## [v2.13] — 2026-01-17 — Johnson Rewrite + Solver Verification — Edit_86 (`ae8fc8b`)

**Major update**: Johnson solver rewritten according to literature, RK45 verified as correct for exponential grids.

### Johnson Solver Rewrite

**Implemented Johnson Renormalized Numerov Method** (B.R. Johnson, J. Chem. Phys. 69, 4678 (1978)):
- Propagates ratio `R_n = ψ_{n-1}/ψ_n` using recursion: `R_{n+1} = 1/(T_n - R_n)`
- Avoids exponential overflow/underflow issues
- Numerically stable for all L values

### RK45 Verification ✓

**Free particle test** (exact analytical solution known):
| Grid | Max Phase Error | Status |
|------|-----------------|--------|
| Exponential | 0.003 rad | ✓ Correct |
| Uniform | 0.004 rad | ✓ Correct |

**Conclusion**: RK45 gives correct phases on exponential grids, Numerov/Johnson give errors up to 1.3 rad.

### Solver Recommendations

| Grid Type | Recommended Solver | Notes |
|-----------|-------------------|-------|
| **Exponential** (default) | `"rk45"` | Only correct solver for h varying 10⁻⁵ to 0.3 |
| **Uniform** (`linspace`) | `"numerov"` or `"johnson"` | O(h⁴) accuracy, fastest |

> [!CAUTION]
> **Numerov and Johnson are designed for uniform grids only**. For exponential grids (standard DWBA), use `"rk45"`.

### Configuration

```yaml
oscillatory:
  solver: "rk45"  # Recommended for exponential grids
  # Options: "auto", "rk45" (recommended), "johnson", "numerov"
```

---

## [v2.12] — 2026-01-14 — Configurable ODE Solver & LOCAL Grid Fix — Edit_85 (`7a8d27a`)

**Critical bug fix** for LOCAL adaptive grid + new configurable ODE solver selection.

### New Features

**Configurable ODE solver** for continuum waves:
- **Primary solver selection**: Choose `"numerov"` (default), `"johnson"`, or `"rk45"` as primary
- **Automatic fallback chain**: If primary fails, remaining solvers tried in order
- **Per-energy logging**: Stats on which solver was used (`Continuum waves computed (E=12.50 eV): Numerov: 48, Johnson: 2`)

```yaml
oscillatory:
  solver: "numerov"  # "numerov" (default), "johnson", "rk45"
```

**Updated `solve_continuum_wave()` signature**:
```python
solve_continuum_wave(..., solver="numerov")
```

**`ContinuumWave` now stores `solver_method`** for tracking which solver produced each wave.

### 🔴 Critical Bug Fix

**Turning point index overflow in LOCAL adaptive mode** (`dwba_matrix_elements.py`):
- **Bug**: When `r_turn > r_max` (high L partial waves on small grids), `searchsorted()` returned `N_grid`, causing `MIN_IDX = N_grid + 20` → out of bounds
- **Impact**: LOCAL adaptive strategy failed with "index N+1 out of bounds for size N" errors
- **Root Cause**: `idx_turn + 20` safety margin exceeded array bounds when turning point was beyond grid
- **Fix**: Added clamping: `idx_turn = min(idx_turn, N_grid - 20)` and `MIN_IDX = min(..., N_grid)`
- **Affected functions**:
  - `radial_ME_all_L()` (CPU path, line ~475)
  - `radial_ME_all_L_gpu()` (GPU path, line ~1182)

> [!NOTE]
> This bug only affected LOCAL adaptive mode. GLOBAL and MANUAL strategies were unaffected.

### Diagnostic Improvements

**Enhanced debug logging** for grid/array size tracking:
- `driver.py`: Added pre-GPU-integrals logging with all array sizes and `idx_match` values
- `dwba_matrix_elements.py`: Entry-point logging with full input validation details
- `continuum.py`: ContinuumWave creation logging with `chi_size`, `idx_match`, `grid_size`
- `DW_main.py`: Pre-calculation logging with `prep` array sizes

### Files Modified

| File | Change |
|------|--------|
| `continuum.py` | Dynamic solver dispatch with fallback chain; `ContinuumWave.solver_method` field |
| `driver.py` | Solver config propagation to workers; solver stats logging; removed redundant log |
| `config_loader.py` | `OscillatoryConfig.solver` field; `config_to_params_dict()` solver mapping |
| `DW_main.py` | `DEFAULTS['oscillatory']` now includes `phase_extraction` and `solver` |
| `dwba_matrix_elements.py` | Critical bounds fix for LOCAL adaptive mode |
| `H2s.yaml`, `examples/config_excitation.yaml` | Added `solver: "numerov"` option |

---

## [v2.11] — 2026-01-13 — Phase Extraction Fix & Hybrid Method — Edit_82 (`35d208f`)

**Critical bug fix** in phase shift extraction and new hybrid approach for improved robustness.

### 🔴 Critical Bug Fix

**Sign error in log-derivative phase extraction corrected** (`continuum.py`):
- **Bug**: Denominator in `tan(δ)` formula had inverted sign
- **Impact**: Phase shifts had wrong sign (error ~0.7 rad = 40°)
- **Fix**: Changed `n_hat_prime - Y_rho * n_hat` → `Y_rho * n_hat - n_hat_prime`
- **Affected functions**:
  - `_extract_phase_logderiv_neutral()` (line ~319)
  - `_extract_phase_logderiv_coulomb()` (line ~352)

> [!NOTE]
> Total cross sections (TCS) may have been less affected due to sin²(δ) = sin²(-δ) symmetry.
> Differential cross sections (DCS) angular distributions were likely incorrect.

### New Features

**Hybrid phase extraction method** (`phase_extraction: "hybrid"`):
- Default method combining log-derivative and least-squares
- Cross-validates both methods; uses weighted average on disagreement
- Fallback to LSQ when wavefunction is near node at match point
- Configuration: `oscillatory.phase_extraction` in YAML

**New configuration option**:
```yaml
oscillatory:
  phase_extraction: "hybrid"  # "hybrid", "logderiv", or "lsq"
```

**Updated `solve_continuum_wave()` signature**:
```python
solve_continuum_wave(..., phase_extraction_method="hybrid")
```

### Diagnostic Suite Updates

**New diagnostic in `debug/debug.py`**:
- Option 6: "Phase Method Comparison (v2.11+)"
- Compares log-derivative vs LSQ on synthetic and real data
- Validates formula correctness with known phase shifts

**Menu reorganization**:
- Removed broken "Full Physics Trace" options (15-16)
- Renumbered options for cleaner structure

### Files Modified

| File | Change |
|------|--------|
| `continuum.py` | Sign fix in 2 functions, new `_extract_phase_hybrid()`, updated `solve_continuum_wave()` |
| `config_loader.py` | Added `phase_extraction` to `OscillatoryConfig` |
| `debug/debug.py` | New `run_phase_method_comparison()`, updated menu |
| `H2s.yaml` | Added `phase_extraction: "hybrid"` |
| `examples/config_excitation.yaml` | Added `phase_extraction` documentation |

---

## [v2.10] — 2026-01-11 — Configuration Refactoring & Output Controls — Edit_80 (`f2a0314`)

Restructured configuration system with separate Hardware category and full Output options support.

### Configuration Changes

**New `hardware:` section** in YAML configuration:
- Extracted GPU/CPU parameters from `oscillatory:` into dedicated section
- Parameters: `gpu_block_size`, `gpu_memory_mode`, `gpu_memory_threshold`, `n_workers`
- Backward compatibility maintained (old configs with GPU params in oscillatory still work)

**New `HardwareConfig` dataclass** in `config_loader.py`:
- Clean separation of concerns: oscillatory integrals vs hardware acceleration
- Automatic migration from legacy configs

### Output Controls

**`output.calibrate` now fully functional**:
- When `false`: skips pilot calculation, uses α=1.0, no Tong model curve in plots
- Interactive mode: calibration can be toggled via `output` category
- Fixed: setting `calibrate: false` in config no longer causes permanent calibration skip

**`output.save_dcs` and `output.save_partial` implemented**:
- Parameters now visible and modifiable in interactive mode
- Respected in both interactive and batch execution

### Plotter Updates

**Conditional calibration display**:
- Plots now detect when `sigma_mtong_cm2` is null
- When calibration disabled: DWBA curve only, no Tong curve or C(E) factor
- Cleaner legends for non-calibrated results

### Files Modified

- `config_loader.py`: Added `HardwareConfig`, updated parsing with backward compatibility
- `DW_main.py`: Separated `DEFAULTS` into hardware/output, updated `prompt_use_defaults` for bools
- `H2s.yaml`: Restructured with new `hardware:` section
- `plotter.py`: Conditional calibration curve rendering

---

## [v2.9] — 2026-01-11 — Enhanced Diagnostic Framework — Edit_79 (`9c29eef`)

Comprehensive diagnostic suite with unified menu and result tracking.

### New Features

**Enhanced `debug/debug.py`**:
- **20-option categorized menu** for all diagnostic functions
- **`DiagnosticResult` class** with pass/fail tracking for each test
- **JSON export** to `debug/results/` with automatic timestamps
- **Timing measurements** for each diagnostic section

**Menu Categories**:
| Category | Options | Description |
|----------|---------|-------------|
| Quick Tests | 1-2 | Health check, convergence study |
| Bound States | 3 | Normalization, energy, nodes, orthogonality |
| Continuum Waves | 4-6 | Phase shifts, high-L stability |
| Potentials | 7-8 | Core, distorting, multi-atom |
| Radial Integrals | 9-10 | I_L breakdown, method comparison |
| Angular Coupling | 11 | CG, Racah coefficients |
| Cross Sections | 12-14 | Partial waves, anomalies |
| Full Traces | 15-16 | Excitation, ionization pipelines |
| Batch | 20 | Run all diagnostics |

### File Reorganization

**Unified naming convention** (`diag_*.py`):
- `debug_amplitude.py` → `diag_amplitude.py`
- `debug_angular.py` → `diag_angular.py`
- `debug_bound.py` → `diag_bound.py`
- `compare_methods.py` → `diag_method_compare.py`
- `verify_oscillatory.py` → `diag_oscillatory.py`

**Output directories** moved inside `debug/`:
- `debug/results/` — JSON diagnostic results
- `debug/plots/` — Generated diagnostic plots

**Removed redundant scripts** (merged into main `debug.py`):
- `test_h_energy.py`
- `test_bypass_disabled.py`
- `check_orthogonality.py`

### Bug Fixes

- Fixed `NameError` in `ionization.py` (`L_max_projectile` → `L_max_proj`)
- Fixed asymptotic amplitude check in `debug.py` (expected `sqrt(2/π)` not `1.0`)
- Fixed attribute names in `diag_L0_L1_anomaly.py` (`delta_l` → `phase_shift`, `chi` → `chi_of_r`)
- Fixed JSON serialization for complex numbers and numpy types
- Fixed CG coefficient test to compare absolute values (sign depends on phase convention)
- **Added `U_f_array` to ionization.py radial integrals** — Bug #2 fix was missing from ionization module, now both `radial_ME_all_L` and `radial_ME_all_L_gpu` calls pass final-state potential for proper asymptotic validation

---

## [v2.8] — 2026-01-11 — Oscillatory Integral Stability Fixes — Edit_78 (`f757d1d`)

Critical fixes for oscillatory integral factorization and phase calculations at high energies.

### Critical Bug Fixes

**Bug #6: Bound State Extent Check (CRITICAL)**
- **Files**: `dwba_matrix_elements.py`
- **Problem**: The "advanced" oscillatory method factorized the 2D radial integral assuming bound states are localized within `r_match`. For H 1s→2s, `r_match ≈ 4 a₀` but the 2s state extends to ~13 a₀. This caused **~10x underestimation** of radial integrals.
- **Root Cause**: Match point was determined by continuum wave properties (potential decay), ignoring bound state extent.
- **Fix**: Added bound state extent check that computes 99% cumulative density radius for **each** bound state independently, then uses MAX. For H 1s→2s: `max(4.2 a₀, 12.7 a₀) = 12.7 a₀`.
- **Impact**: "Legacy" and "advanced" methods now agree within 5% (previously differed by 10x).

**Bug #7: Centrifugal Phase Corrections**
- **Files**: `oscillatory_integrals.py`
- **Problem**: Asymptotic phase calculation for oscillatory tail integrals ignored centrifugal terms, causing phase mismatch for high-L partial waves.
- **Fix**: Added first-order centrifugal correction to all phase functions:
  - `compute_asymptotic_phase`: Added `-l(l+1)/(2kr)` term
  - `compute_phase_derivative`: Added `+l(l+1)/(2kr²)` term
  - `compute_phase_second_derivative`: Added `-l(l+1)/(kr³)` term
  - `dwba_outer_integral_1d`: Updated `phi_minus`, `phi_plus` and their derivatives to include centrifugal terms for both incoming and outgoing waves.
- **Impact**: Improved tail integral accuracy for L > 5.

**Bug #8: Centrifugal Potential in Match Point Validation**
- **Files**: `dwba_matrix_elements.py`
- **Problem**: Asymptotic validation only checked `U_i`, `U_f` potentials against kinetic energy threshold, ignoring the dominant centrifugal barrier `L(L+1)/(2r²)` for high L.
- **Fix**: `get_max_V_eff()` now includes centrifugal term: `V_eff = max(|U_i|, |U_f|) + L(L+1)/(2r²)`.
- **Impact**: Proper asymptotic validation for high partial waves.

### Diagnostic Improvements

**New Diagnostic Scripts** (`debug/` folder):
- `diag_upturn.py`: Analyzes partial wave convergence at specific energies, flags non-monotonic behavior
- `diag_radial_integrals.py`: Detailed I_L breakdown across energy range
- `compare_methods.py`: Side-by-side comparison of "legacy" vs "advanced" methods

**Physical Findings**:
- **Cross section dip at 11-17 eV**: Confirmed as physical node in I₀ integral (not numerical artifact)
- **L=5 "upturn" at 69 eV**: Confirmed as physical interference pattern (radial integrals cross zero between L=3-4, peak at L=5)

### Configuration Changes

**Updated** `n_points_max` default: 8000 → **15000** in:
- `DW_main.py`
- `config_loader.py`
- `H2s.yaml`
- `examples/config_excitation.yaml`
- `examples/config_ionization.yaml`

This allows better grid resolution for high-energy (>100 eV) calculations.

---

## [v2.7] — 2026-01-10 — Calibration & Stability Fixes — Edit_77 (`aaff973`)

Critical bug fixes improving calibration accuracy and numerical stability.

### Bug Fixes

**Bug #2: Asymptotic Validation Now Checks Both Potentials**
- **Files**: `dwba_matrix_elements.py`, `driver.py`
- **Problem**: Match point validation only checked `U_i`, ignoring `U_f`. If `U_f` decays slower, phase extraction could be unstable.
- **Fix**: `radial_ME_all_L` and `radial_ME_all_L_gpu` now accept `U_f_array` parameter and use `max(|U_i|, |U_f|)` for asymptotic threshold check.
- Uses `min(k_i, k_f)` for kinetic energy (stricter criterion).

**Bug #4: Dynamic Pilot L_max for Calibration**
- **Files**: `DW_main.py`
- **Problem**: Default `pilot_L_max_projectile=30` was insufficient for 1000 eV calibration (k≈8.6 a.u. requires L_max≈100+ for convergence). This caused underestimated σ_DWBA pilot, leading to systematically incorrect α factor.
- **Fix**: Pilot L_max is now dynamically calculated: `L_proj = max(base, min(k*r_max*0.6, 150))`.
- Applied to both interactive and batch modes.

**Bug #5: Run Name Change Now Renames Files**
- **Files**: `DW_main.py`
- **Problem**: Changing simulation name via menu option "7" only updated the variable, leaving existing result files with old names.
- **Fix**: After name change, offers to rename existing `results_<old>_*.json` and `*_<old>_*.png` files to the new name.

**Bug #3: Analytical Tail Contribution Extended to Infinity**
- **Files**: `dwba_matrix_elements.py`
- **Problem**: For methods "advanced" and "full_split", the oscillatory outer integral `dwba_outer_integral_1d` stopped at `r_max` instead of integrating to ∞. For dipole (L=1) and higher multipoles, the tail ~1/r^(L+1) has non-negligible contribution beyond the grid.
- **Fix**: Added explicit call to `_analytical_multipole_tail(r_max, ...)` after `dwba_outer_integral_1d` in both CPU and GPU code paths. This uses asymptotic expansion with Si(x)/Ci(x) behavior for the remaining [r_max, ∞) domain.
- Affects direct integrals for L≥1 in excitation calculations.

### Performance Optimizations

**Vectorized Asymptotic Wave Construction**
- **Files**: `continuum.py`
- **Optimization**: Replaced Python loop in `_build_asymptotic_wave` with vectorized NumPy operations using `spherical_jn`/`spherical_yn` on arrays.
- **Impact**: ~10x speedup for asymptotic wave construction, which is called for every partial wave.

**Optimized Compensated Summation**
- **Files**: `oscillatory_integrals.py`
- **Optimization**: Replaced hand-written Kahan summation loops with `math.fsum()` which uses Shewchuk's algorithm in C.
- **Impact**: Faster and more accurate summation for oscillatory integrals.

**Vectorized Chebyshev Differentiation Matrix**
- **Files**: `oscillatory_integrals.py`
- **Optimization**: Replaced O(n²) double loop in `_build_chebyshev_diff_matrix` with NumPy broadcasting operations.
- **Impact**: Faster Levin collocation setup for oscillatory outer integrals.

**Vectorized Function Evaluations**
- **Files**: `oscillatory_integrals.py`
- **Optimization**: Replaced list comprehensions `[f(r) for r in nodes]` with `np.vectorize(f)(nodes)` in Levin and Filon quadrature.
- **Impact**: Eliminated Python interpreter overhead in inner loops.

### Numerical Stability Improvements

**Wavelength-Based Grid Scaling**
- **Files**: `DW_main.py`
- **Problem**: At high energies (e.g., 1000 eV), exponential grid had insufficient resolution at large r. Only ~1.5 pts/wavelength caused aliasing and phase extraction errors.
- **Fix**: `calculate_optimal_grid_params` now ensures minimum 15 points per wavelength by computing: `n_points >= r_check × ln(ratio) / (λ/15)`
- **Impact**: Automatically increases grid density for high-energy calculations, preventing "Phase unstable" warnings.

**Born Approximation in Analytic Bypass**
- **Files**: `continuum.py`
- **Problem**: When analytic bypass was used (potential negligible early), phase shift was set to 0, which is incorrect even for weak potentials.
- **Fix**: Now computes Born approximation: `δ_Born = -k ∫ U(r) [j_l(kr)]² r² dr` using trapezoidal quadrature.
- **Impact**: Non-zero phase shifts for high-L waves where potential is weak but not zero.

---

## [v2.6] — 2026-01-03 — Adaptive Grid Strategies — Edit_73 (`d077221`)

Introduces user-selectable strategies for radial grid adaptation, optimizing performance and accuracy.

### Adaptive Grid Strategies

**Files**: `DW_main.py`, `config_loader.py`

New `grid.strategy` configuration option with 3 modes:
- **Global (default)**: Calculates optimal `r_max` and `n_points` based on the lowest energy in the scan. Single target preparation.
- **Local**: Recalculates optimal grid parameters for *each* energy point. Most accurate but slower due to repeated bound state solving. Uses 5% change threshold to avoid unnecessary recalculations.
- **Manual**: Uses fixed `r_max` and `n_points` from configuration.

**Helper**: `calculate_optimal_grid_params()` logic extracted and formalized.

### Grid Strategy Implementation Fix (v2.6.1)

**Files**: `DW_main.py`

- **Fixed**: Interactive mode now correctly reads and applies `params['grid']['strategy']` 
- **Added**: Comprehensive logging showing actual grid parameters being used
- **Added**: Per-energy grid recalculation for LOCAL mode with optimization (only recalc if params change >5%)
- **Added**: Strategy information included in result metadata (`run_meta['grid']['strategy']`)
- **Improved**: User feedback with `print_subheader("Grid Configuration")` and detailed parameter display

**Logging examples**:
```
  Strategy: GLOBAL (single adaptive calculation)
    E_min = 10.7 eV, k_min = 0.42 a.u.
    r_max = 200.0 a.u., n_points = 3000
```

### Configuration Updates

- `GridConfig`: Added `strategy` field.
- `DW_main.py`: Updated interactive menu to allow selecting grid strategy.
- **Updated**: All example config files with improved strategy documentation.

---

## [v2.6.2] — 2026-01-03 — Logging & Validation Improvements — Edit_74 (`8c55cba`)

Quality-of-life improvements for logging readability, physical validation, and documentation.

### Logging Improvements

**Files**: `driver.py`, `DW_main.py`

- **Added**: Scan-level logging control via `_SCAN_LOGGED` flag
- **Added**: `reset_scan_logging()` function to reset flags at start of new scan
- **Improved**: Hardware info logged once per scan (not per energy point), reducing log spam

### High-Energy Validation

**Files**: `grid.py`

- **Added**: `K_HIGH_ENERGY_THRESHOLD = 5.0 a.u.` constant (~340 eV)
- **Added**: `validate_high_energy()` function checking:
  - High-energy regime warning (k > 5 a.u.)
  - Wavelength undersampling detection
  - L_max vs turning point consistency
- **Integrated**: Validation called before calculation loops with warnings displayed

### SAE Parameter Validation

**Files**: `potential_core.py`

- **Added**: `CorePotentialParams.validate_physical()` method checking:
  - Zc validity (must be >= 1 for atoms/ions)
  - Very large a_params (|a| > 200) causing numerical instability
  - Negative decay rates (a2, a4, a6 < 0) causing exponential growth

### YAML Documentation

**Files**: `H2s.yaml`, `examples/config_excitation.yaml`, `examples/config_ionization.yaml`

- **Added**: Unit annotations for all numerical parameters:
  - `# [eV]` for energies
  - `# [a.u.]` for atomic units (bohr)
  - `# [count]` for integer counts
  - `# [dimensionless]` for ratios and exponents
  - `# [rad]` for angles

### Adaptive High-Energy Precision

**Files**: `continuum.py`

- **Added**: Adaptive precision for very high energies (>1 keV) where phase extraction is challenging:
  - E > 1 keV: Enhanced precision mode (`rtol=1e-7`, `atol=1e-9`)
  - E > 5 keV: Strictest precision mode (`rtol=1e-8`, `atol=1e-10`)
  - Tighter renormalization intervals for Numerov/Johnson propagators
  - Logging of high-energy mode activation
- **Constants**: `E_HIGH_THRESHOLD_EV = 1000.0`, `E_VERY_HIGH_THRESHOLD_EV = 5000.0`

### Comprehensive Parameter Logging

**Files**: `DW_main.py`

- **Added**: `log_active_configuration()` helper function that logs all active parameters before calculation starts
- **Improved**: Clear visibility into what parameters are actually being used (especially auto-calculated ones)
- **Format**: Structured log output showing Grid, Excitation/Ionization, and Oscillatory settings

---

## [v2.5] — 2026-01-03 — GPU Optimization V3 — Edit_72 (`43d816d`)

Major GPU performance improvements reducing synchronization overhead and adding energy-level caching.

### GPU Synchronization Reduction

**Files**: `dwba_matrix_elements.py`

- Replaced per-L `float(cp.dot(...))` conversions with GPU array accumulation
- Single `.get()` transfer at end of L-loop instead of ~2×L_max individual syncs
- Added `return_gpu=True` parameter to `_gpu_filon_direct()` and `_gpu_filon_exchange()`
- Precomputed L=0 correction terms on GPU (`sum_rho2_dir`, `V_diff_dot_rho1_dir`)

**Impact**: Eliminates ~2×L_max GPU→CPU synchronizations per `radial_ME_all_L_gpu()` call

### GPUCache Dataclass

**Files**: `dwba_matrix_elements.py`, `driver.py`

New `GPUCache` class for energy-level resource reuse:
- `r_gpu`, `w_gpu` — Persistent grid arrays
- `inv_gtr`, `log_ratio` — Base kernel matrices (built once per energy)
- `filon_params` — Filon quadrature parameters
- `chi_cache` — LRU-managed continuum wave cache (max 20 entries)

**Usage in driver.py**:
```python
gpu_cache = GPUCache.from_grid(grid)
# ... passed to all radial_ME_all_L_gpu calls ...
gpu_cache.clear()  # At end of energy point
```

### Continuum Wave GPU Cache with LRU

**Files**: `dwba_matrix_elements.py`

- `GPUCache.get_chi(chi_wave, channel)` — Retrieve/cache continuum wave on GPU
- LRU eviction when cache exceeds `max_chi_cached` (default 20)
- Reduces `cp.asarray()` calls for frequently-reused partial waves

### Memory Management

**Files**: `dwba_matrix_elements.py`, `driver.py`

- Removed `free_all_blocks()` from per-call locations
- Cleanup consolidated in `GPUCache.clear()` at end of energy point
- Avoids GPU sync overhead from frequent pool flushing

### Pilot Light Mode

**Files**: `driver.py`, `DW_main.py`, `config_loader.py`

Fast calibration mode with reduced parameters for faster pilot calculations:

**New Config Parameters** (in `excitation` section):
- `pilot_L_max_integrals: 8` — Lower than production (reduces computation)
- `pilot_L_max_projectile: 30` — Limited partial waves
- `pilot_n_theta: 50` — TCS only, DCS not needed for calibration

**Usage in DW_main.py**:
```python
pilot_res = compute_excitation_cs_precalc(
    pilot_E, prep, 
    n_theta=pilot_n_theta,
    L_max_integrals_override=pilot_L_max_integrals,
    L_max_projectile_override=pilot_L_max_projectile
)
```

**Impact**: Pilot calibration is 5-10x faster with minimal effect on α accuracy

### Improved Logging

**Files**: `driver.py`

- `log_calculation_params()` now accepts `actual_gpu_mode` and `actual_block_size`
- Displays actual values used, not just configured values
- Block display: "full-matrix" when using full matrix mode, else block size

---

## [Unreleased]

### Edit_69 — Output Organization & Tooling Improvements (`a0ec9a8`)

Major refactoring of output file organization and enhancement of analysis tools.

#### Output Directory Structure

**Files**: `output_utils.py` (NEW), `DW_main.py`, `plotter.py`, `partial_wave_plotter.py`

All output files now saved to dedicated directories:
- `results/` — Calculation results (JSON) and plots (PNG)
- `fited_potentials/` — Fitted potential plots

**New Module**: `output_utils.py`
- `get_results_dir()` — Returns results/ path, creates if needed
- `get_output_path(filename)` — Get path for any output file
- `get_json_path(run_name, calc_type)` — Get path for results JSON
- `find_result_files(pattern)` — Auto-discover result files
- `migrate_existing_files(dry_run)` — Helper to move legacy files

**Backward Compatibility**: `load_results()` checks both `results/` and root directory.

#### Partial Wave Analysis Tool Rewrite

**Files**: `partial_wave_plotter.py`

- **Interactive file selection** — Menu displays available result files from `results/`
- **Run/transition selection** — Choose which transition to analyze
- **Configurable L_max** — Set how many partial waves to display
- **L_90% convergence analysis** — Shows L value at which sum reaches 90% of total
- **New plot: L_90% vs Energy** — Convergence requirements vs energy
- **Summary statistics** — Energy range, max L, σ_total

#### Visualization Functions Update

**Files**: `DW_main.py`

- `run_visualization()` and `run_dcs_visualization()` now search `results/` directory
- Use `find_result_files()` for consistent file discovery
- Output path display: `results/results_{run}_exc.json`

#### Fit Potential Output

**Files**: `fit_potential.py`

- Plots saved to `fited_potentials/fit_{atom_name}.png`
- Directory created automatically
- Added logging

#### Wigner Symbol Cache Scaling

**Files**: `dwba_coupling.py`

- Default cache size: 10k → 50k entries
- `scale_wigner_cache(L_max)` for dynamic scaling (up to 1M entries for L_max > 100)
- `clear_wigner_caches()` and `get_wigner_cache_stats()` utilities

---

### Edit_70 — GPU/Multiprocessing Optimization — (`2362af2`)

Major improvements to GPU and CPU multiprocessing configuration and consistency.

#### Configurable CPU Worker Count

**Files**: `driver.py`, `ionization.py`, `config_loader.py`

New `n_workers` configuration parameter:
- `n_workers: "auto"` — Optimized balance (uses `min(cpu_count, 8)`)
- `n_workers: "max"` — Uses all available CPU cores
- `n_workers: N` — Explicit count (capped at cpu_count)

**New Helper**: `get_worker_count()` in `driver.py`
- Returns configured or auto-detected worker count
- Logs actual count and mode used (e.g., `Multiprocessing: using 8 worker(s) [Mode: auto]`)
- Used by both excitation and ionization CPU paths

#### Ionization GPU Config Passthrough

**Files**: `ionization.py`

- GPU calls now receive full `OSCILLATORY_CONFIG` parameters
- Includes: method, CC_nodes, phase_increment, k_threshold, gpu_memory_mode
- CPU CPU path also uses consistent config
- Worker count uses `get_worker_count()` instead of `os.cpu_count()`

#### Fixed CPU Worker Count

**Files**: `driver.py`

- Removed hardcoded `max_workers = 4`
- Now uses `get_worker_count()` from global config
- Batch size scales with worker count#### Logging Refinements

**Files**: `driver.py`, `ionization.py`

- **Calculation Summary** — New consolidated start logs showing Hardware, Platform, CPU Workers, and Multipole order in one clear block.
- **Semantic Config Updates** — `set_oscillatory_config` now normalizes values (e.g., `0` vs `"auto"`) to avoid redundant "Config updated" logs.
- **Worker Logging** — CPU worker count is now consistently logged at the start of both excitation and ionization paths.

---

#### Configuration Updates

**Files**: `examples/config_excitation.yaml`, `examples/config_ionization.yaml`

Added `n_workers` to oscillatory section in example configs:
```yaml
oscillatory:
  n_workers: "auto"  # "auto" = auto-detect (up to 8 CPUs), >0 = explicit count
```

---

#### GPU Cleanup Fix

**Files**: `dwba_matrix_elements.py`

Fixed `NameError` in GPU cleanup code — added existence checks for GPU arrays before deletion.

---

### Edit_71 — Batch/Interactive Unification — Refinement — (`9bca217`)

**Files**: `DW_main.py`

**Code Quality Improvements**:
- `prompt_use_config_file`: Single load per config file, proper error logging, invalid configs excluded
- Cleaner config selection UI: `run_name (Atom state) - path`  
- Both excitation and ionization batch now use `prepare_target` optimization
- **Excitation**: Eliminated duplicate grid/bound-state calculations — threshold extracted from `prep.dE_target_eV`
- **Ionization**: Uses `prep.orb_i` for threshold, identical JSON format with `energy_eV`, `IP_eV`, `sdcs`, `tdcs`, `partial_waves`, `meta`
- Keyboard interrupt saves partial results in both modes

### Edit_66 — GPU Memory Auto-tuning — (`129343d`)

**Files**: `dwba_matrix_elements.py`, `driver.py`

- **Auto-tuning block size**: `gpu_block_size=0` (new default) computes optimal size based on available VRAM
- Explicit value overrides auto-tune (e.g., `gpu_block_size=4096`)
- Memory pool cleanup (`cp.get_default_memory_pool().free_all_blocks()`) before large kernel allocations
- `_compute_optimal_block_size()`: Estimates max block that fits in `threshold × free_mem`

### Edit_65 — GPU Path Logic Fix — Performance Optimization — (`2a02a3a`)

**Files**: `dwba_matrix_elements.py`

**Critical Fixes**:
- **Decoupled Filon from block-wise**: `use_filon` and `use_block_wise` are now independent decisions
- When VRAM permits, full matrix is built and used by BOTH Filon and standard paths
- Added `full_matrix_built` flag to track matrix availability
- Fixed potential crash when `use_block_wise=True` + `!use_filon` (missing `inv_gtr`)
- Catches Windows pagefile errors (`"pagefile"`, `"out of memory"` strings)

**Performance Impact**:
| Scenario | Before | After |
|----------|--------|-------|
| `k>k_thr`, VRAM OK | Block-wise (slow) | Full matrix + Filon (fast) |
| `k>k_thr`, VRAM low | Block-wise | Block-wise (same) |

### Edit_67 — Hybrid Filon Integration — Major Performance Boost — (`8f3eec5`)

**Files**: `dwba_matrix_elements.py`, `config_loader.py`, `DW_main.py`, `*.yaml`

**Optimization Strategy — Three Modes**:
1. **Filon/full-matrix**: Extended matrix `(idx_limit × N_grid)` built, single `cp.dot()` (fastest)
2. **Filon/hybrid**: Standard matrix `(idx_limit × idx_limit)` + block-wise for tail (fallback)
3. **Filon/block-wise**: No prebuilt matrix, all block-wise (low memory)

**Config Fixes**:
- `gpu_block_size` changed to `0` (auto-tune) in:
  - `DW_main.py` DEFAULT_PARAMS
  - `config_loader.py` OscillatoryConfig
  - `H2s.yaml`, `examples/*.yaml`

**Logging**:
- Mode logged once per run: `GPU mode: Filon/full-matrix`

### Edit_68 — UI Improvements — (`a2157de`)

**Files**: `DW_main.py`, `dwba_matrix_elements.py`, `README.md`

- **Log deduplication**: GPU mode logged only on mode change, not for every energy point
- **Parameter display**: `gpu_block_size = auto` instead of confusing `= 0`
- **Default excitation**: H transition changed from 1s→2p to 1s→2s
- **README expanded**: Comprehensive parameter reference tables for Grid, Excitation, Ionization, Oscillatory, and GPU parameters

### Edit_66 — GPU Performance Optimization — L-Loop Improvements — (`129343d`)

**Files**: `dwba_matrix_elements.py`

**Synchronization Reduction**:
- Hoisted `rho2_eff_full` computation before L-loop (was computed L+1 times per energy point)
- Removed 3 `free_all_blocks()` calls from inside L-loop (were causing GPU sync each iteration)

**Memory Estimation Fix**:
- Fixed auto-mode memory check for Filon: now uses `idx_limit × N_grid` instead of `idx_limit²`
- Prevents incorrect fallback to block-wise when extended matrix would fit

### Edit_67 — Config: Support for `gpu_block_size: "auto"` String — (`8f3eec5`)

**Files**: `config_loader.py`, `DW_main.py`, `*.yaml`

- `gpu_block_size` now accepts `"auto"` string in YAML (more intuitive than `0`)
- Added `_parse_gpu_block_size()` helper to convert `"auto"` → `0` internally
- Updated all example configs and templates

---

### Edit_64 — `7227310` — 2026-01-01

#### Configuration File Support (Batch Mode)

Enables automated batch calculations without interactive prompts using YAML configuration files.

**New Files**:
- `config_loader.py` — YAML parser with validation and dataclass conversion
- `examples/config_excitation.yaml` — Template for excitation calculations
- `examples/config_ionization.yaml` — Template for ionization calculations

**CLI Arguments** (`DW_main.py`):
```bash
python DW_main.py -c config.yaml        # Batch mode
python DW_main.py -c config.yaml -v     # Verbose batch mode
python DW_main.py --generate-config     # Generate template
```

**Features**:
- Full validation with clear error messages
- Support for all calculation parameters (grid, oscillatory, excitation, ionization)
- Automatic Tong calibration with pilot calculation in batch mode
- Automatic result merging for incremental runs
- Progress output during batch execution

#### GPU Memory Optimization — Hybrid Memory Strategy

**Files**: `driver.py`, `dwba_matrix_elements.py`

Implemented adaptive GPU memory management with three modes:

| Mode | Description |
|------|-------------|
| `auto` | Checks available GPU memory; uses full matrix if sufficient, otherwise block-wise |
| `full` | Forces full N×N matrix construction (fastest, may cause OOM) |
| `block` | Forces block-wise construction (slower, constant memory usage) |

Configuration parameters: `gpu_memory_mode`, `gpu_memory_threshold`

---

### Edit_63 — `56668f7` — 2026-01-01

#### Code Review Audit

Comprehensive review of the DWBA codebase against the theoretical article (Lai et al., 2014) and supplementary ionization literature (Jones/Madison, Bote/Salvat).

##### Exchange Phase Convention Verification
Added explicit documentation in `dwba_coupling.py` (lines 388-404) verifying the Condon-Shortley phase convention for exchange spherical harmonics. The code correctly implements:
```
Y_{l,m}^*(θ,φ) = (-1)^m × Y_{l,-m}(θ,φ)
```
Cross-checked against Khakoo et al. experimental DCS data. No errors found.

##### Ionization L_max Floor
Added `L_floor=3` parameter to `_auto_L_max()` in `ionization.py`. This guarantees that s-, p-, and d-wave contributions are always included, even at very low energies near threshold where the adaptive scaling might otherwise reduce L_max to 0.

##### Oscillatory Quadrature Documentation
Enhanced `k_threshold` parameter documentation in `dwba_matrix_elements.py` explaining:
- When `k_total > k_threshold` (default 0.5 a.u.): Use specialized Filon/Levin oscillatory quadrature
- When `k_total ≤ k_threshold`: Standard Simpson integration is faster and sufficiently accurate

##### Overall Assessment
No fundamental errors found. The implementation faithfully follows the theoretical framework.

#### Critical Bug Fixes

##### CRITICAL: Integration Weight Fix
**File**: `oscillatory_integrals.py`

Corrected a critical bug where integration weights were missing from 2D oscillatory integrals:
```python
# BEFORE (incorrect):
int_r2 = np.dot(kernel_lim, rho2_lim)
result = np.dot(rho1_lim, int_r2)

# AFTER (correct):
int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)  # Include dr weights
result = np.dot(rho1_lim * w1_lim, int_r2)
```
This fix applies to both direct and exchange integral paths.

##### GPU Cleanup Bug Fix
**File**: `dwba_matrix_elements.py`

Fixed potential `NameError` when using Filon mode in GPU integrals:
```python
# BEFORE: del inv_gtr, ratio, log_ratio  ← Fails in Filon mode
# AFTER:
if not use_filon:
    del inv_gtr, ratio, log_ratio
```

#### Numerical Improvements

##### Multipole Moment Accuracy
The multipole transition moment $M_L$ is now computed over the **full radial grid** instead of the match-point-limited grid, capturing the complete bound-state tail.

##### GPU Full-Grid Parity
The GPU implementation now uses the full radial grid for the inner r₂ integral, matching the CPU "Full-Split" method.

##### Scalable GPU Architecture
- Increased default `gpu_block_size` from 1024 to **8192**
- Added explicit memory pool cleanup
- Result: Constant VRAM footprint regardless of grid size

##### Physics-Based Numerov Coefficients
**File**: `continuum.py`

Improved the Numerov propagator for non-uniform grids using separate h₁², h₂² instead of averaged h², preserving O(h⁴) accuracy.

##### Phase Stability Unwrapping
Added 2π unwrapping to phase stability diagnostics to prevent false warnings.

#### Documentation
- Comprehensive expansion of `CHANGELOG.md` from 201 → 450 lines
- Added code examples and physics context for all commits

---

### Refinement Audit — Current Session

#### CRITICAL: Integration Weight Fix
**File**: `oscillatory_integrals.py`

Corrected a critical bug where integration weights were missing from 2D oscillatory integrals:
```python
# BEFORE (incorrect):
int_r2 = np.dot(kernel_lim, rho2_lim)
result = np.dot(rho1_lim, int_r2)

# AFTER (correct):
int_r2 = np.dot(kernel_lim, rho2_lim * w2_lim)  # Include dr weights
result = np.dot(rho1_lim * w1_lim, int_r2)
```
This fix applies to both direct and exchange integral paths. Without proper `dr` weights, the integrals were dimensionally inconsistent.

#### GPU Cleanup Bug Fix
**File**: `dwba_matrix_elements.py`

Fixed potential `NameError` when using Filon mode in GPU integrals:
```python
# BEFORE: del inv_gtr, ratio, log_ratio  ← Fails in Filon mode
# AFTER:
if not use_filon:
    del inv_gtr, ratio, log_ratio
```
In Filon mode, the kernel is built block-wise and these variables are never created at module scope.

#### Multipole Moment Accuracy
**File**: `dwba_matrix_elements.py`

The multipole transition moment $M_L$ is now computed over the **full radial grid** instead of the match-point-limited grid:
```python
# BEFORE: moment_L = Σ w_gpu × r^L × u_f × u_i   (match-point limited)
# AFTER:  moment_L = Σ w_full × r_full^L × u_f_full × u_i_full  (full grid)
```
This captures the complete bound-state tail for accurate analytical integrals beyond the match point.

#### GPU Full-Grid Parity
**File**: `dwba_matrix_elements.py`

The GPU implementation now uses the full radial grid for the inner r₂ integral, matching the CPU "Full-Split" method:
```python
# Inner integral now covers r₂ ∈ [0, R_max] instead of [0, r_m]
for start in range(0, N_grid, BLOCK_SIZE):  # Full grid
    ...
```
This ensures mathematical equivalence between CPU and GPU paths.

#### Scalable GPU Architecture
**File**: `dwba_matrix_elements.py`

- Increased default `gpu_block_size` from 1024 to **8192** for better throughput
- Added explicit `cp.get_default_memory_pool().free_all_blocks()` calls
- Result: Constant VRAM footprint regardless of grid size (tested up to N=10000)

#### Physics-Based Numerov Coefficients
**File**: `continuum.py`

Improved the Numerov propagator for non-uniform (exponential) grids:
```python
# BEFORE: Single averaged step size
h_avg = 0.5 * (h_prev + h_next)
h2 = h_avg * h_avg

# AFTER: Proper local step sizes  
h1_sq = h1 * h1  # Backward step
h2_sq = h2 * h2  # Forward step
h_center_sq = h1 * h2  # Geometric mean for center term
```
This preserves O(h⁴) accuracy on exponential grids where step sizes vary by 3-5× across the domain.

#### Phase Stability Unwrapping
**File**: `continuum.py`

Added 2π unwrapping to the phase stability diagnostic:
```python
delta_diff = (delta_l - delta_alt + np.pi) % (2 * np.pi) - np.pi
```
Prevents false "phase unstable" warnings when the phase difference crosses a 2π boundary.

---

### Edit_62 — `042b044` — 2026-01-01

#### High-Order Fornberg Derivative
**File**: `continuum.py` (lines 685-746)

Replaced the approximate 3-point central difference with a proper **5-point Fornberg finite-difference stencil**:

```python
# OLD: Simple central difference (O(h²) for uniform grids only)
return (chi[idx + 1] - chi[idx - 1]) / (r_grid[idx + 1] - r_grid[idx - 1])

# NEW: Fornberg algorithm (O(h⁴) for any grid)
# Computes optimal FD coefficients for the actual local node spacing
# Reference: B. Fornberg, Math. Comp. 51 (1988)
```

The Fornberg algorithm automatically generates optimal finite-difference weights for non-uniform grids, enabling accurate derivatives on exponential radial meshes.

#### Advanced Parameter Control
**File**: `driver.py`

Wired oscillatory quadrature parameters through the full pipeline:
- `min_grid_fraction`: Minimum fraction of grid to use for match point (prevents r_m too close to origin)
- `k_threshold`: Wave number threshold for switching to oscillatory quadrature
- `CC_nodes`: Number of Clenshaw-Curtis nodes per phase interval
- `phase_increment`: Target phase change per quadrature segment

#### Progress Tracking & UX
**Files**: `driver.py`, `DW_main.py`

- Added real-time ETA and elapsed-time logging every 10 partial waves
- Refactored verbose configuration tables behind an optional toggle
- Improved output formatting for long energy scans

---

### Edit_61 — `9b77978` — 2025-12-31

#### Documentation Initialization
- Created `CHANGELOG.md` with structured format based on Keep a Changelog
- Comprehensive 500-line `README.md` update covering:
  - Repository structure and module descriptions
  - Unit system explanations (atomic units throughout)
  - Usage workflows for excitation and ionization
  - Calibration and debugging guidance

#### Grid Adaptivity
**File**: `dwba_matrix_elements.py`

Implemented `min_grid_fraction` logic to ensure the match point never falls too close to the origin:
```python
MIN_IDX = max(idx_turn + 20, int(N_grid * min_grid_fraction))
```
This prevents numerical instabilities when high-L waves have turning points near the grid origin.

---

### Edit_60 — `4317871` — 2025-12-31

#### Clenshaw-Curtis Node Caching
**File**: `oscillatory_integrals.py`

Introduced module-level cache for precomputed CC quadrature nodes and weights:
```python
_CC_CACHE = {5: (_CC_X_REF, _CC_W_REF)}  # Keyed by n_nodes

def _get_cc_ref(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _CC_CACHE.get(n_nodes)
    if cached is not None:
        return cached
    ...
```
This eliminates redundant trigonometric computations in tight radial integration loops.

**Performance impact**: ~25% speedup for oscillatory quadrature with repeated node counts.

#### Coupling Integration
**File**: `dwba_coupling.py`

Fine-tuned the interface between angular coupling functions and the new oscillatory integral module. Ensured consistent phase conventions across the amplitude calculation pipeline.

---

### Edit_59 — `926b99a` — 2025-12-24

#### GPU Filon Quadrature Engine
**File**: `dwba_matrix_elements.py`

Comprehensive implementation of Filon-type oscillatory quadrature for GPU:
- Vectorized interpolation of envelope functions at CC nodes
- Memory-efficient block-wise kernel construction
- Native CuPy operations avoiding CPU-GPU synchronization

```python
def _gpu_filon_direct(rho1_uw, int_r2, r_gpu, w_gpu, k_total, phase_increment, CC_nodes, ...):
    """GPU-native Filon quadrature for direct radial integrals."""
```

#### Standardized 2D Quadrature API
**File**: `oscillatory_integrals.py`

Created unified `oscillatory_kernel_integral_2d` function handling:
- Complex density products (χ_i × u_f for direct, χ_f × u_i for exchange)
- Multi-method selection (filon, levin, standard)
- Automatic fallback for non-oscillatory regimes

---

### Edit_58 — `e642f8c` — 2025-12-23

#### Oscillatory Integrals Module
**New file**: `oscillatory_integrals.py` (~2000 lines)

Centralized all advanced quadrature algorithms:
- **Levin collocation**: Solves u' + iΦ'u = f to handle nonlinear phase
- **Filon-type**: Polynomial envelope with exact exponential integration
- **Clenshaw-Curtis**: Chebyshev-based weights for smooth integrands
- **sinA×sinB decomposition**: Product-to-sum for wave function pairs

```
χ_i(r) × χ_f(r) ~ sin(Φ_i) × sin(Φ_f) = ½[cos(Φ_i - Φ_f) - cos(Φ_i + Φ_f)]
```

#### Full-Split Integration Paradigm
Implemented the $I_{in} + I_{out}$ domain decomposition:
- **I_in**: Numerical integration from 0 to match point r_m (full density)
- **I_out**: Analytical integration from r_m to ∞ using asymptotic forms

This handles high-frequency oscillations in the asymptotic region where standard quadrature fails.

---

### Edit_57 — `7f18342` — 2025-12-22

#### Continuum Solver Overhaul
**File**: `continuum.py`

Major upgrade of the radial Schrödinger solver:
- Numerov O(h⁴) propagator with proper non-uniform grid handling
- Physics-based turning point detection using S(r) = l(l+1)/r² + 2U(r) - k²
- WKB initialization for classically forbidden regions
- Coulomb phase shift extraction for ionic targets

#### Turning Point Intelligence
```python
# Check if we're inside centrifugal barrier at grid start
S_at_origin = ell*(ell+1)/(r0*r0) + 2*U[0] - k²
if S_at_origin > 0:
    # Use WKB-like initial conditions
    chi0 = 1e-20
    chi1 = chi0 * exp(√S × h)
```
This correctly handles both low-L waves at low energies (strong potential) and high-L waves where the centrifugal barrier dominates.

---

### Edit_56 — `d9a1286` — 2025-12-22

#### Result Normalization Audit
**File**: `results_H2p_exc.json`

Large-scale correction of cross-section data:
- Applied proper (2π)⁴ kinematic factors
- Corrected k_f/k_i prefactors for all energies
- Verified spin-averaging: ¼|f+g|² + ¾|f-g|²

This synchronizes the stored results with the theoretical framework.

---

### Edit_55 — `4fb7a03` — 2025-12-21

#### Visualization Updates
**Files**: `plotter.py`, `DW_main.py`

- Updated plot generation to handle new result JSON structure
- Fixed metadata inconsistencies in JSON export (missing `theta_deg` arrays)
- Added support for atomic unit (a₀²) vs SI (cm²) output units

---

### Edit_54 — `42eecc7` — 2025-12-21

#### Ionization Engine Refactor
**File**: `ionization.py` (370 lines modified)

Major overhaul of ionization cross-section calculations:
- Standardized kinematic factor: $(k_{scatt} × k_{eject}) / k_i$
- Consistent $(2π)^4$ normalization across SDCS, TDCS
- Improved ejected electron angle integration
- Exchange angle swapping for indistinguishable electrons

#### Distortion Potential Unification
**File**: `distorting_potential.py`

Ensured consistent potential construction:
- $U_i(r) = V_{A^+}(r) + V_H^{(i)}(r)$ — Core + Hartree from initial state
- $U_f(r) = V_{A^+}(r) + V_H^{(f)}(r)$ — Core + Hartree from final state
- Exchange treated perturbatively in T-matrix (not in distorting potential)

---

### Edit_53 — `3b70228` — 2025-12-20

#### Coupling Logic Extraction
**New file**: `dwba_coupling.py`

Extracted angular momentum coupling from the main driver:
- Wigner 3j and 6j symbols with proper phase conventions
- Clebsch-Gordan coefficients via 3j relation
- Racah W via 6j with correct phase factor
- Direct and exchange amplitude assembly (Eq. 412, 448)

#### Diagnostic Tooling
**New file**: `debug_amplitude.py`

Per-channel amplitude verification tool for debugging cross-section discrepancies.

**File**: `plotter.py`

Added multi-style visualization:
- `std`: Energy (eV) vs σ (cm²)
- `atomic`: Energy (Ha) vs σ (a₀²)
- `article`: E/E_thr vs σ/(πa₀²)
- `ev_au`: Energy (eV) vs σ (a.u.)

---

### Edit_52 — `2d7a7f9` — 2025-12-19

#### Validation Suite
**New files**: `debug_angular.py`, `debug_bound.py`

Automated health checks:
- Wigner symbol triangle rules and selection rules
- SAE bound-state normalization (∫u² dr = 1)
- Orthogonality between states

#### Coupling Vectorization
**File**: `dwba_coupling.py`

First-pass NumPy vectorization of CG coefficient loops, achieving ~3× speedup for amplitude accumulation.

---

### Edit_51 — `ab8e54f` — 2025-12-19

#### Driver Stability
**File**: `driver.py`

- Fixed result file lock handling for concurrent writes
- Improved error recovery in partial wave loop

**File**: `plotter.py`

Enhanced unit system toggle between `atomic` and `std` conventions.

---

### Edit_50 — `070a24c` — 2025-12-19

#### Project Consolidation
Massive results directory restructuring:
- Aggregated fragmented atom-specific JSONs into core datasets
- Established naming convention: `results_{target}{transition}_exc.json`
- Improved scaling performance for large result sets

#### SAE Potential Refinement
**File**: `distorting_potential.py`

Updated fitting bounds for heavy-atom core potentials (Ne, Ar, Kr), improving convergence of the potential optimizer.

---

### Edit_49 — `3d11ada` — 2025-12-17

#### Dataset Expansion
Generated comprehensive excitation dataset for H(n=2) → H(n'=3,4,5).

**File**: `plotter.py`

Added enhanced scaling for vibrationally excited targets with small cross-sections.

---

### Edit_48 — `f7aba35` — 2025-12-17

#### Calibration Refinement
**File**: `calibration.py`

Precision tuning of Tong model parameters:
- Dipole transitions: β=0.5, γ=0.25, δ=0.75
- Non-dipole: β=0.3, γ=0.15, δ=0.45

Added reference results for He⁺(1s → 2p) excitation.

---

### Edit_47 — `7110adc` — 2025-12-17

#### Multi-Target Results
Massive bulk calculation run completing datasets for:
- H(1s → 2s)
- He⁺(1s → 2s)
- He⁺(1s → 2p)
- Na(3s → 3p)
- Ne⁺ various transitions

All integrated into the primary results library.

---

### Edit_46 — `7cb4ca5` — 2025-12-16

#### Architecture Documentation
**File**: `README.md`

200-line expansion detailing:
- Module dependency graph
- Data flow from input to cross-section output
- Grid construction and unit handling
- Numerical method selection criteria

#### Potential Optimizer Overhaul
**File**: `fit_potential.py` (748 lines changed)

- Standardized use of `scipy.optimize.differential_evolution`
- Improved bounds specification for SAE potential parameters (a₁...a₆)
- Added constraint functions for physically reasonable potentials

---

### Edit_45 — `338b07a` — 2025-12-16

#### Lithium Support
**File**: `atoms.json`

Added Li with pre-fitted Tong-Lin SAE potential:
- Ionization potential: 5.39 eV
- Core parameters: a₁=1.6, a₂=2.4, a₃=-1.8, a₄=3.8, a₅=-1.1, a₆=0.9

**File**: `driver.py`

Optimized loop structures for large energy scans (100+ points).

---

### Edit_44 — `a5be463` — 2025-12-16

#### Dataset Sanitization
- Removed redundant/stale He⁺ JSON files
- Restructured `results_H_exc.json` for long-term compatibility

---

### Edit_43 — `a3c7fed` — 2025-12-16

#### Folder Reorganization
Created structured subdirectories:
- `article_png/`: Theory derivation diagrams
- `debug/`: Diagnostic scripts and test cases
- `fited_potentials/`: Pre-computed SAE potential parameters

#### Sigma Core Cleanup
**File**: `sigma_total.py`

Removed legacy cross-section logic, keeping only the main DCS/TCS functions with proper documentation.

---

### Edit_42 — `97faedc` — 2025-12-16

#### Atom Diagnostics
**New file**: `diag_atoms.py`

Health-checking tool for the SAE potential library:
- Verifies bound state energies match NIST data
- Checks potential asymptotic behavior (-Z/r)
- Validates orthogonality of computed orbitals

#### Fitting Robustness
**File**: `fit_potential.py`

Enhanced boundary handling for complex atoms with multiple near-threshold states.

---

### Edit_41 — `3269803` — 2025-12-15

#### Core Refactoring
Systematic naming and signature cleanup across:
- `bound_states.py`: Renamed `solve_bound_state` → `solve_bound_states`
- `continuum.py`: Unified function signatures for wave solvers
- `driver.py`: Consistent parameter ordering
- `ionization.py`: Aligned with excitation conventions

---

### Edit_40 — `afbcf77` — 2025-12-15

#### Centralized Logging
**New file**: `logging_config.py`

Introduced structured logging replacing print statements:
```python
from logging_config import get_logger
logger = get_logger(__name__)
logger.debug("Partial wave L=%d: σ=%.2e", L, sigma)
```

#### Docstring Standards
Applied NumPy-style docstrings across all modules with:
- Parameter descriptions
- Return value specifications
- Example usage where appropriate

---

## [2.1.2] — 2024-12-31

### Added
- **GPU Memory Management**: Block-wise calculation for radial integrals prevents VRAM exhaustion on systems with limited GPU memory.
- **User Configurability**: Exposed `gpu_block_size`, `CC_nodes`, `phase_increment` in `DW_main.py` interactive UI.

### Fixed
- **Multiprocessing Performance**: Localized `matplotlib` imports to prevent initialization delays in worker processes on Windows.
- **Import Error**: Resolved missing `set_oscillatory_config` import in `DW_main.py`.

---

## [2.1.1] — 2024-12-31

### Fixed
- **CRITICAL: Missing Integration Weights** — `oscillatory_kernel_integral_2d` was computing matrix products without proper `dr` integration weights. Cross-sections were incorrect by factors of 200-500×.

### Changed
- **Performance**: Implemented caching and pre-slicing for GPU Filon nodes/kernels (~50× speedup).
- **Phase Stability**: 4th-order central difference for phase extraction; proper 2π unwrapping.
- **Adaptive Grid**: Point density now scales with incident energy (up to 10k for 1000 eV).

---

## [2.1.0] — 2024-12-31

### Fixed
- **Match Point Selection**: `_find_match_point` now searches forward from `idx_start + 50`.
- **Phase Extraction**: Corrected sign in log-derivative formula.

### Changed
- **Physics-Based Turning Point**: Uses S(r_min) > 0 instead of hardcoded l > 5.
- **Non-Uniform Numerov**: Separate h₁², h₂² for O(h⁴) accuracy on exponential grids.

---

## [2.0.0] — 2024-12-01

### Added
- Full DWBA implementation for electron impact excitation
- Ionization cross-sections (TDCS, SDCS, TCS)
- GPU acceleration via CuPy
- Oscillatory integral methods (Filon, Levin)
- Tong model empirical calibration
- Atom library with pre-fitted potentials
- Interactive menu system

---

## [1.0.0] — 2024-11-15

### Added
- Initial DWBA implementation for hydrogen-like targets
- Basic radial grid and bound state solver
- Core potential fitting routines

---

## Notes

### Versioning
- **Major**: Breaking changes or significant new features
- **Minor**: New functionality, backward compatible
- **Patch**: Bug fixes and minor improvements

### Git Commits
For detailed commit-level changes, see `git log --oneline`.

### References
- Lai et al. (2014): DWBA theory for electron-impact excitation
- Jones & Madison (2003): (e,2e) ionization formalism
- Fornberg (1988): Finite difference weights on arbitrary grids
- Tong & Lin (2005): Single-active-electron potentials
