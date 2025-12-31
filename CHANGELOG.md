# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed
- Ongoing improvements and bug fixes

---

## [2.1.1] - 2024-12-31

### Fixed
- **CRITICAL: Missing integration weights in 2D oscillatory integrals** - `oscillatory_kernel_integral_2d` was performing matrix dot product without `w_grid` integration weights, causing radial integrals to be ~200-500× too large. This affected all cross-section calculations.
  - Added `w_grid` parameter to `oscillatory_kernel_integral_2d` and all GPU variants
  - Inner integral now correctly uses `kernel @ (rho2 * w)` instead of `kernel @ rho2`
  - Fallback paths now correctly apply `rho1 * w` for outer integral
  - GPU functions `_gpu_filon_direct` and `_gpu_filon_exchange` updated similarly

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
