# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### [Code Review Audit] - 2026-01-01
- **Exchange Phase Convention Verification**: Added comprehensive documentation in `dwba_coupling.py` (lines 388-404) explicitly verifying the Condon-Shortley phase convention for exchange amplitudes per Article Eq. 448. Cross-checked against Khakoo et al. experimental DCS.
- **Ionization L_max Floor**: Added `L_floor=3` parameter to `_auto_L_max()` in `ionization.py` to ensure minimum partial wave coverage (s, p, d contributions) at very low energies near threshold where k → 0.
- **Oscillatory Quadrature Threshold Documentation**: Enhanced `k_threshold` parameter documentation in `dwba_matrix_elements.py` explaining the physics rationale for switching between Filon/Levin and Simpson integration.
- **Comprehensive Code Audit**: Verified all DWBA modules against Lai et al. (2014) article and ionization literature (Jones/Madison, Bote/Salvat). No fundamental errors found.

### [Refinement Audit] - Current session
- **CRITICAL: Integration Weight Fix**: Corrected missing integration weights (`w2_lim`) in 2D oscillatory integrals for exchange terms in `oscillatory_integrals.py`. Previous code computed `kernel × rho2` without proper `dr` weights, causing inaccurate results.
- **GPU Cleanup Bug Fix**: Fixed potential `NameError` in `radial_ME_all_L_gpu` when using Filon mode. The `del inv_gtr, ratio, log_ratio` cleanup now only executes in non-Filon mode where these variables exist.
- **Multipole Moment Parity**: Corrected `moment_L` calculation to use full-grid weights (`w`) instead of partitioned weights (`w_limited`) for higher accuracy in analytical tails (CPU).
- **GPU Integral Parity**: Updated `radial_ME_all_L_gpu` to transfer full bound state data and use full-grid integration for the inner $r_2$ integral, matching CPU "Full-Split" logic.
- **Scalable GPU Architecture**: Increased default `gpu_block_size` to **8192** and implemented explicit pool cleanup to ensure constant VRAM footprint for large grids ($N=10000$).
- **Physics-Based Numerov**: Improved non-uniform grid handling with separate $h_1^2$, $h_2^2$ coefficients instead of averaged $h^2$, preserving O(h⁴) accuracy on exponential grids.
- **Phase Unwrapping**: Added 2π unwrapping to phase stability check in `continuum.py` to prevent false instability warnings.

### [Edit_62] (042b044) - 2026-01-01
- **High-Order Phase Extraction**: Replaced inconsistent derivative logic in `continuum.py` with a stable **5-point Fornberg finite-difference stencil**. Ensures $O(h^4)$ accuracy on non-uniform grids.
- **Advanced Parameter Control**: Wired `min_grid_fraction` and `k_threshold` through the DWBA integral pipeline to allow fine-tuning of oscillatory quadrature activation.
- **Progress Tracking & UX**: Added real-time ETA and per-wave timing metrics to `driver.py`. Refactored `DW_main.py` to hide verbose configuration tables behind an optional toggle.

### [Edit_61] (9b77978) - 2025-12-31
- **Documentation Engine**: Initialized `CHANGELOG.md` and performed a comprehensive `README.md` update to reflect the v2.1 feature set.
- **Grid Adaptivity**: Implemented `min_grid_fraction` logic to prevent match points from occurring too close to the origin, ensuring stable asymptotic matching for high-L waves.

### [Edit_60] (4317871) - 2025-12-31
- **Node Caching Optimization**: Introduced `_CC_CACHE` in `oscillatory_integrals.py` for precomputed Clenshaw-Curtis nodes/weights, reducing trig overhead in radial loops.
- **Coupling Refinement**: Fine-tuned angular amplitude contributions in `dwba_coupling.py` for better integration with the new oscillatory integrals.

### [Edit_59] (926b99a) - 2025-12-24
- **GPU Filon Engine**: Comprehensive implementation of Filon-type quadrature for GPU. Features vectorized interpolation and memory-efficient kernel construction.
- **2D Quadrature API**: Standardized `oscillatory_kernel_integral_2d` to handle complex density products and multi-method selection (filon/exchange).

### [Edit_58] (e642f8c) - 2025-12-23
- **Architectural Shift**: Introduced the dedicated `oscillatory_integrals.py` module, centralizing Levin, Filon, and Clenshaw-Curtis algorithms.
- **Domain Decomposition**: Implemented the `Full-Split` integration paradigm ($I_{in} + I_{out}$) for handling high-frequency oscillations in the asymptotic region.

### [Edit_57] (7f18342) - 2025-12-22
- **Continuum Solver Overhaul**: Upgraded `continuum.py` with a physics-driven Numerov $O(h^4)$ propagator.
- **Turning Point Intelligence**: Implemented automated turning point detection ($S(r_{min}) > 0$) and WKB initialization for suppressed regions, significantly improving the stability of high-L partial waves.

### [Edit_56] (d9a1286) - 2025-12-22
- **Result Normalization Audit**: Large-scale update to `results_H2p_exc.json` to reflect corrected normalization factors and kinematic terms across the entire excitation dataset.

### [Edit_55] (4fb7a03) - 2025-12-21
- **Visualization Sync**: Updated plot generation scripts to align with new result formats. Corrected minor metadata inconsistencies in JSON export routines.

### [Edit_54] (42eecc7) - 2025-12-21
- **Ionization Engine Refactor**: Major 370-line logic overhaul in `ionization.py`. Standardized kinetic factors and (2π) normalization for TDCS/SDCS calculations.
- **Distortion Potential Unification**: Refactored `distorting_potential.py` to ensure consistent core and static potential handling across both excitation and ionization channels.

### [Edit_53] (3b70228) - 2025-12-20
- **Coupling Logic Extraction**: Transitioned angular momentum coupling and Z-matrix algebra into the dedicated `dwba_coupling.py` module.
- **Diagnostic Tooling**: Introduced `debug_amplitude.py` for per-channel amplitude verification. Optimized `plotter.py` with new multi-style visualization support.

### [Edit_52] (2d7a7f9) - 2025-12-19
- **Validation Suite**: Added `debug_angular.py` and `debug_bound.py` for automated health checks of Wigner symbols and SAE bound-state normalization.
- **Coupling Optimization**: First-pass vectorization of the new coupling layer in `dwba_coupling.py`.

### [Edit_51] (ab8e54f) - 2025-12-19
- **Driver Stability**: Minor bugfixes in `driver.py` regarding result file lock handling. Expanded `plotter.py` with enhanced `atomic` vs `std` unit toggling.

### [Edit_50] (070a24c) - 2025-12-19
- **Project Consolidation**: Massive restructuring of the results directory. Aggregated fragmented atom-specific JSONs into core datasets (`results_H2p_exc.json` etc.) to improve scaling performance.
- **SAE Potential Refinement**: Updated `distorting_potential.py` with improved fitting bounds for heavy-atom core potentials.

### [Edit_49] (3d11ada) - 2025-12-17
- **Dataset Expansion**: Generation of comprehensive dataset for $H(v=2)$ excitation. Updated `plotter.py` with enhanced scaling for vibrationally excited targets.

### [Edit_48] (f7aba35) - 2025-12-17
- **Calibration Refinement**: Precision tuning of the Tong calibration factors in `calibration.py`. Added reference results for $He^+ (2p)$ excitation.

### [Edit_47] (7110adc) - 2025-12-17
- **Multi-Target Results**: Massive bulk run completing datasets for $H(2s)$, $He^+(2s)$, $Na$, and $Ne^+$. Integrated these into the primary results library.

### [Edit_46] (7cb4ca5) - 2025-12-16
- **Architecture Documentation**: 200-line expansion of `README.md` detailings system internals.
- **Parametric Optimizer**: Major overhaul of `fit_potential.py` (748 lines changed). Standardized the use of `differential_evolution` for SAE potential parameter fitting.

### [Edit_45] (338b07a) - 2025-12-16
- **Lithium Support**: Added full support for $Li$ excitation. Optimized `driver.py` loop structures to improve throughput for large energy scans.

### [Edit_44] (a5be463) - 2025-12-16
- **Dataset Sanitization**: Cleaned up redundant and stale $He^+$ JSON files. Restructured `results_H_exc.json` for long-term archival.

### [Edit_43] (a3c7fed) - 2025-12-16
- **Folder Reorganization**: Moved binary assets (plots/images) and configuration aids into structured subdirectories (`article_png/`, `debug/`, `fited_potentials/`).
- **Sigma Core Cleanup**: Refactored `sigma_total.py` to remove legacy cross-section logic.

### [Edit_42] (97faedc) - 2025-12-16
- **Atom Diagnostics**: Introduced `diag_atoms.py` for health-checking the SAE potential library.
- **Fitting Robustness**: Enhanced `fit_potential.py` with better boundary handling for complex atoms.

### [Edit_41] (3269803) - 2025-12-15
- **Core Refactoring**: Systematic naming and signature cleanup across `bound_states.py`, `continuum.py`, `driver.py`, and `ionization.py` to improve maintainability.

### [Edit_40] (afbcf77) - 2025-12-15
- **Centralized Logging**: Introduced `logging_config.py`, replacing print-based debugging with a structured logger across all modules.
- **Docstring Standards**: Standardized module-level documentation across the entire toolkit.

---

## [2.1.2] - 2024-12-31

### Added
- **GPU Memory Management**: Implemented block-wise calculation for direct radial integrals, allowing stable execution on systems with limited VRAM (prevents "Pagefile too small" errors).
- **User Configurability**: Integrated `gpu_block_size`, `CC_nodes`, and `phase_increment` into the `DW_main.py` interactive UI.

### Fixed
- **Multiprocessing Performance**: Optimized imports in `DW_main.py` by localizing `matplotlib` and `plotter` calls. This eliminates initialization "hangs" in worker processes on Windows.
- **NameError**: Resolved a missing import for `set_oscillatory_config` in `DW_main.py`.

---

## [2.1.1] - 2024-12-31

### Fixed
- **CRITICAL: Missing integration weights in 2D oscillatory integrals** - `oscillatory_kernel_integral_2d` was performing matrix dot product without `w_grid` integration weights, causing radial integrals to be ~200-500× too large. This affected all cross-section calculations.
  - Added `w_grid` parameter to `oscillatory_kernel_integral_2d` and all GPU variants
  - Inner integral now correctly uses `kernel @ (rho2 * w)` instead of `kernel @ rho2`
  - Fallback paths now correctly apply `rho1 * w` for outer integral
  - GPU functions `_gpu_filon_direct` and `_gpu_filon_exchange` updated similarly

### Changed
- **Performance Optimization**: Implemented caching and pre-slicing for GPU Filon quadrature nodes and kernels in `dwba_matrix_elements.py`.
  - Dramatically reduces GPU memory pressure by avoiding large kernel matrix allocations for L > 0.
  - Estimated total speedup for oscillatory radial integrals: up to 50x compared to v2.1.0.
- **Phase Stability & Accuracy**: Upgraded continuum wave analysis in `continuum.py`.
  - Implemented 4th-order (5-point) central difference for phase extraction derivatives.
  - Corrected phase unwrapping logic to handle 2π jumps at high energies (1000 eV+).
- **Adaptive Grid Density**: Increased default point density and scaled it with incident energy to improve high-frequency wave representation (reaching 10k points for 1000 eV).
- **UX Improvements**: Added real-time progress logging and ETA to the partial wave summation in `driver.py`.

---

## [2.1.0] - 2024-12-31

### Fixed
- **Critical: Match point selection** - Function `_find_match_point` now searches FORWARD from `idx_start + 50`, preventing "all solvers failed" errors for high partial waves (L > 50)
- **Match point threshold** - Relaxed from 0.01% to 1% (|U|/(k²/2) < 0.01 instead of 0.0001)
- **Phase extraction formula** - Corrected sign in denominator: `[n̂' - Y·n̂]` instead of `[Y·n̂ - n̂']`

### Changed
- **Physics-based turning point detection** - Now uses `S(r_min) > 0` criterion instead of hardcoded `l > 5` threshold
- **Numerov for non-uniform grids** - Uses separate step sizes h₁², h₂² instead of averaged (h_avg)² for O(h⁴) accuracy on exponential grids
- **Adaptive initial conditions** - Always evaluates S(r_start) to choose between WKB and regular boundary conditions

### Improved
- Documentation in `continuum.py` module docstring updated to reflect v2.1 implementation
- README.md updated with detailed description of radial solver methods

---

## [2.0.0] - 2024-12-01

### Added
- Full DWBA implementation for electron impact excitation
- Ionization cross-section calculations (TDCS, SDCS, TCS)
- GPU acceleration via CuPy for partial wave summation
- Oscillatory integral methods (Filon, Levin quadrature)
- Tong model empirical calibration
- Atom library with pre-fitted potentials (H, He, Ne, Ar, etc.)
- Interactive menu system in `DW_main.py`
- Comprehensive logging system

### Technical Details
- Numerov propagator with fallback to Johnson log-derivative and RK45
- Asymptotic stitching for continuum wavefunctions
- Split radial integrals (numerical + analytic tail)
- Phase-adaptive quadrature with sinA×sinB decomposition

---

## [1.0.0] - 2024-11-15

### Added
- Initial implementation of DWBA for hydrogen-like targets
- Basic radial grid and bound state solver
- Core potential fitting routines

---

## Notes

### Versioning
- Major version: Breaking changes or significant new features
- Minor version: New functionality, backward compatible
- Patch version: Bug fixes and minor improvements

### Git Commits
For detailed commit-level changes, see `git log --oneline`.
