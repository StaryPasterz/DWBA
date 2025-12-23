# DW\_antigravity\_v2 – Distorted Wave Born Approximation Toolkit

Comprehensive Python suite for computing electron–atom excitation and ionization cross sections using the Distorted Wave Born Approximation (DWBA), with optional empirical calibration (Tong model) and plotting/diagnostic tools.

## Table of Contents
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Theory Snapshot](#theory-snapshot)
- [Units and Normalization](#units-and-normalization)
- [Requirements](#requirements)
- [Setup](#setup)
- [Core Workflows](#core-workflows)
  - [Interactive Driver (`DW_main.py`)](#interactive-driver-dw_mainpy)
  - [Excitation Scan](#excitation-scan)
  - [Ionization Scan](#ionization-scan)
  - [Plotting Results](#plotting-results)
  - [Potential Fitting](#potential-fitting)
- [Atom Library (`atoms.json`)](#atom-library-atomsjson)
- [Calibration (Tong / DWBA)](#calibration-tong--dwba)
- [Debugging and Logging](#debugging-and-logging)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Features
- DWBA excitation cross sections (total and differential) with static + optional polarization potentials; exchange via T-matrix.
- DWBA ionization cross sections (TICS, SDCS, optional TDCS) using (2π)⁴ kinematic factor (Jones & Madison 2003), with auto-scaled L_max and cached scattered waves.
- Empirical calibration (Tong model) for excitation, with per-energy normalization factors.
- Plotting in multiple unit conventions (cm², a.u., article E/E\_thr, mixed).
- Partial-wave convergence diagnostics and Born top-up (DCS scaled to match top-up TCS).
- Potential fitting utility with Tong-Lin (2005) methodology.
- GPU acceleration via CuPy (optional).

## Repository Layout
```
DW_antigravity_v2/
├── DW_main.py              # Main interactive CLI
├── driver.py               # Excitation computation pipeline
├── ionization.py           # Ionization (TICS/SDCS) pipeline
├── continuum.py            # Distorted-wave continuum solver
├── bound_states.py         # Bound-state solver in SAE potential
├── distorting_potential.py # Construction of U_i, U_f
├── dwba_matrix_elements.py # Radial DWBA integrals (CPU/GPU)
├── dwba_coupling.py        # Angular coupling and amplitudes
├── oscillatory_integrals.py # Filon/phase-adaptive quadrature
├── sigma_total.py          # DCS/TCS assembly
├── calibration.py          # Tong model implementation
├── potential_core.py       # SAE core potential V(r)
├── grid.py                 # Radial grid and quadrature
├── atom_library.py         # Atom database interface
├── atoms.json              # SAE parameters for atoms/ions
├── nist_data.json          # NIST reference energies
├── fit_potential.py        # Potential parameter fitting tool
├── plotter.py              # Result plotting utility
├── logging_config.py       # Centralized logging
├── debug.py                # Health checks and diagnostics
├── article.md              # Reference paper (theory)
└── README.md               # This file
```

## Theory Snapshot
- DWBA amplitudes follow Eq. (216), (412), (448) of the article: direct `f` and exchange `g` from multipole expansion of `1/r₁₂`, with distorted continuum waves.
- Distorting potentials: `U_j(r) = V_core(r) + V_H(r)` (static), with optional `V_pol`.
- SAE core potential (Eq. 69):
  ```
  V(r) = -[Zc + a₁e^(-a₂r) + a₃re^(-a₄r) + a₅e^(-a₆r)] / r
  ```
- Calibration: Tong model (Eq. ~34, 493) provides TCS; normalization factor `C(E)` rescales DWBA TCS. For DCS, multiply by `calibration_factor` if you need calibrated angular distributions.

### Cross Section Formulas
- **Excitation DCS**: `dσ/dΩ = (2π)⁴ × (k_f/k_i) × |T|²` (Eq. 216)
- **Ionization TDCS**: `d³σ/(dΩ₁dΩ₂dE) = (2π)⁴ × (k_f×k_ej/k_i) × |T|²` (Jones/Madison 2003)

## Units and Normalization
| Quantity | Internal Unit | Output Unit |
|----------|---------------|-------------|
| Distance | bohr (a₀) | — |
| Energy | Hartree (Ha) | eV |
| Cross Section | a.u. | cm² |

- Conversion: 1 a.u. area = 2.8003×10⁻¹⁷ cm²
- Continuum waves: unit asymptotic amplitude (normalized via `DELTA_NORM_CORRECTION`)

## Requirements
- Python ≥ 3.9
- NumPy, SciPy, Matplotlib
- (Optional) CuPy for GPU acceleration

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install numpy scipy matplotlib
```

For GPU support:
```bash
pip install cupy-cuda12x      # Match your CUDA version
```

## Core Workflows

### Interactive Driver (`DW_main.py`)
```bash
python DW_main.py
```
Menu:
1. Excitation Cross Sections
2. Ionization Cross Sections
3. Generate Plots
4. Partial Wave Analysis
5. Fit Potential (new atom)
6. Change Run Name

### Excitation Scan
- Select target from `atoms.json` or enter custom parameters
- Configure initial/final states (n, l)
- Choose model: DWBA (static) or DWBA + polarization
- Numerics: L_max (multipole), L_max_projectile (base, auto-scaled), n_theta (DCS grid)
- Results saved to `results_<run>_exc.json`

### Ionization Scan
- Similar setup to excitation
- Uses (2π)⁴ kinematic factor for proper continuum normalization (TDWBA convention)
- Results include TICS and SDCS data; optional TDCS for specified angle triplets
- SDCS is integrated over both outgoing electron angles using Y_lm orthonormality
- Tong calibration applies to excitation only; ionization plots show DWBA only.
- Partial-wave diagnostics are stored (integrated over ejected energy).
- TDCS uses angles (theta_scatt, theta_eject, phi_eject) with phi_scatt = 0 (scattering plane).
- Exchange term uses swapped detection angles for indistinguishable electrons (Jones/Madison).
- Numerics: l_eject_max, L_max (multipole), L_max_projectile (base, auto-scaled from k_i), n_energy_steps.
- Polarization option is heuristic and not part of the article DWBA.

### Centralized Default Parameters

All numerical defaults are organized by category and displayed before calculation:

```
  ┌─ GRID ─────────────────────────────
  │  r_max                  = 200
  │  n_points               = 3000
  │  r_max_scale_factor     = 2.5
  │  n_points_max           = 8000
  └────────────────────────────────────

  ┌─ EXCITATION ───────────────────────
  │  L_max_integrals        = 15
  │  L_max_projectile       = 5
  │  n_theta                = 200
  │  pilot_energy_eV        = 1000
  └────────────────────────────────────

  ┌─ OSCILLATORY ──────────────────────
  │  CC_nodes               = 5
  │  phase_increment        = 1.571
  │  min_grid_fraction      = 0.1
  │  k_threshold            = 0.5
  └────────────────────────────────────
```

When prompted, enter:
- **Y** (or Enter): Use all defaults unchanged
- **n**: Edit any parameter individually

Outputs: `results_<run>_exc.json`, `results_<run>_ion.json` in project root. Excitation entries include angular grids (`theta_deg`) and both raw/calibrated DCS in a.u. for later plotting. Ionization entries include SDCS data and optional TDCS entries (`angles_deg`, `values`).

### Plotting Results
```bash
python plotter.py [style] [results_file.json]
```
Styles: `std` (eV/cm²), `atomic` (Ha/a₀²), `article` (E/E_thr), `ev_au`

### Potential Fitting
```bash
python fit_potential.py
```
Features:
- **Reference Protection**: Parameters from Tong-Lin (2005) are marked with `source: "Tong-Lin (2005) Table 1"` and show a warning if you try to re-fit them
- **Global Optimizer**: Uses `differential_evolution` for robust fitting
- **Proper Bounds**: Allows negative a₃, a₅ (required for He, Ne, Ar)
- Auto-saves to `atoms.json`

## Atom Library (`atoms.json`)

Contains SAE model potential parameters for various atoms:

### Tong-Lin (2005) Reference Parameters
| Atom | Zc | a₁ | a₂ | a₃ | a₄ | a₅ | a₆ |
|------|-----|--------|-------|---------|-------|---------|-------|
| H | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 |
| He | 1.0 | 1.231 | 0.662 | -1.325 | 1.236 | -0.231 | 0.480 |
| Ne | 1.0 | 8.069 | 2.148 | -3.570 | 1.986 | 0.931 | 0.602 |
| Ar | 1.0 | 16.039 | 2.007 | -25.543 | 4.525 | 0.961 | 0.443 |
| Rb | 1.0 | 24.023 | 11.107 | 115.200 | 6.629 | 11.977 | 1.245 |
| Ne+ | 2.0 | 8.043 | 2.715 | 0.506 | 0.982 | -0.043 | 0.401 |
| Ar+ | 2.0 | 14.989 | 2.217 | -23.606 | 4.585 | 1.011 | 0.551 |

### Adding New Atoms
1. Run `python fit_potential.py`
2. Enter atomic data (Z, Zc, n, l, binding energy)
3. Wait for optimization (~30-60s)
4. Parameters are automatically saved to `atoms.json`

## Calibration (Tong / DWBA)
- Calibration applies to excitation only; ionization uses raw DWBA results.
- Tong model parameters:
  - Dipole-allowed (|Δl|=1): Set “np” (β=1.32, γ=−1.08, δ=−0.04)
  - Dipole-forbidden (|Δl|≠1): Set “ns” (β=0.7638, γ=1.1759, δ=0.6706)
- Classification is automatic based on |l_f−l_i|.
- α fitted per channel at 1000 eV.
- Per-energy normalization: `C(E) = σ_Tong(E) / σ_DWBA(E)`. TCS are saved as raw and calibrated; to obtain calibrated DCS multiply raw DCS by `calibration_factor`.

## Debugging and Logging

### Log Levels
Set via environment variable:
```bash
set DWBA_LOG_LEVEL=DEBUG    # Windows
export DWBA_LOG_LEVEL=DEBUG # Linux/Mac
```
Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Health Checks
```bash
python debug.py
```
Verifies bound-state norms, continuum waves, and partial-wave convergence.

## Performance Tips

### Grid Numerics: L_max and r_max Selection

The classical turning point constraint is critical for numerical stability:

```
r_t(L) = (L + 0.5) / k      # Turning point for partial wave L
L_max ≤ k × (r_max / C) - 0.5   # Safe L_max for given r_max (C ≈ 2.5)
```

The code automatically enforces this via `compute_safe_L_max()`:
- At **low energies** (small k): L_max is automatically reduced
- If you see **warnings about turning point limits**: increase r_max or accept fewer partial waves

| Energy (eV) | k (a.u.) | r_max=200 | r_max=500 | r_max=1000 |
|-------------|----------|-----------|-----------|------------|
| 15          | 1.05     | L≤33      | L≤83      | L≤167      |
| 50          | 1.92     | L≤61      | L≤153     | L≤306      |
| 100         | 2.71     | L≤86      | L≤216     | L≤433      |

### Adaptive Grid (Automatic r_max Scaling)

The interactive scans (`DW_main.py`) now **automatically compute optimal r_max** based on:
1. Minimum energy in the scan (lowest kinetic energy = most restrictive)
2. Requested L_max_projectile
3. Classical turning point formula

```
E_min_scan → k_min = sqrt(2 × (E_min - threshold))
r_max_optimal = max(200, C × (L_max + 0.5) / k_min)    # C ≈ 2.5
n_points = 3000 × (r_max / 200)                        # Scale proportionally
```

This ensures **all energies in the scan** have sufficient grid extent. The console shows:
```
  • Adaptive Grid: E_min=15.0 eV, k_min=0.27 -> r_max=480, n_points=7200
```

### Radial Solver: Numerical Methods

The continuum wave solver uses **Numerov propagation** with asymptotic stitching:

**Numerov Method** (Primary):
- Solves χ''(r) = Q(r)·χ(r) where Q = l(l+1)/r² + 2U(r) - k²
- O(h⁶) accuracy, more stable than RK45 for oscillatory solutions
- Periodic renormalization prevents over/underflow
- 5-point derivative formula for accurate χ'(r)

**Initial Conditions**:
- Origin (l≤5): Regular boundary χ ~ r^(l+1)
- Turning point (l>5): WKB-like χ ~ exp(κ·r) inside barrier

**Phase Extraction**:
```
tan(δ_l) = [Y_m · ĵ_l - ĵ_l'] / [Y_m · n̂_l - n̂_l']
```
- Matches to Riccati-Bessel (neutral) or Coulomb F,G (ionic) at r_m
- r_m chosen where |2U(r)| << k² and l(l+1)/r²

**Asymptotic Stitching**:
- Numerical χ scaled to match asymptotic amplitude at r_m
- Pure analytic solution A·[ĵ cos(δ) - n̂ sin(δ)] used for r > r_m
- Amplitude A = √(2/π) for δ(k-k') normalization
- Eliminates numerical noise in oscillatory tail

**Split Radial Integrals**:
- Integration uses numerical χ for [0, r_m] and analytic for [r_m, ∞)
- Match point stored in `ContinuumWave.idx_match`

### Oscillatory Radial Integrals

For high partial waves and energies, the radial integrands oscillate rapidly:

```
χ_i(k_i, r) × χ_f(k_f, r) × K_L(r₁, r₂)
```

The **oscillatory_integrals.py** module provides specialized quadrature:

**Phase Sampling Diagnostic**:
- Checks if grid spacing satisfies Nyquist: Δφ < π/4 per step
- Logs warnings when undersampling detected (helps debugging p-orbital issues)

**Filon + Clenshaw-Curtis Method** (Active in Main Path):
- Uses constant phase splitting: Δφ = π/2 per sub-interval
- 5 Chebyshev nodes per sub-interval for accuracy
- Integrand interpolated to CC nodes, summed with CC weights
- **Direct integrals**: Use `filon` method (CC outer integral only)
- **Exchange integrals**: Use `filon_exchange` method (CC both inner AND outer)
- Per instruction: "rozbij całkę na przedziały, na których faza robi stały przyrost"

**Exchange Inner Integral Treatment**:
- Exchange densities contain continuum waves in both rho1 and rho2
- `filon_exchange` applies CC to the inner integral (over r2) for each r1
- Vectorized kernel interpolation using searchsorted + linear interp
- ~10x slower than `filon` but handles oscillatory inner integrals correctly

**Analytical Multipole Tail** (L ≥ 1, Excitation Only):
- **Excitation only**: Disabled for ionization (where bound_f is ContinuumWave)
- **Correct multipole moment**: Uses ∫r^L × u_f × u_i dr (not ∫u_f × u_i)
- **Correct envelope decay**: 1/r^(L+1) kernel, so L=1 uses 1/r², L=2 uses 1/r³
- **Coulomb asymptotic phase**: For ionic targets, includes:
  - η ln(2kr) logarithmic correction (Sommerfeld parameter η = -z_ion/k)
  - σ_l = arg(Γ(l+1+iη)) Coulomb phase shift
  - Full phase: kr + η ln(2kr) - lπ/2 + σ_l + δ
- L=1: Exact integration using Si(x), Ci(x) special functions
- L>1: Asymptotic expansion with boundary term
- **Exchange integrals**: No analytical tail (oscillatory quadrature via CC)

**GPU Acceleration**:
- `radial_ME_all_L_gpu()` supports `use_oscillatory_quadrature` parameter
- Match point domain splitting reduces GPU memory transfer
- Analytical tails computed on CPU, kernel integration on GPU
- **Automatic GPU selection**: `HAS_CUPY and check_cupy_runtime()`
- CPU and GPU versions produce identical results

**Advanced Quadrature API**:
- `clenshaw_curtis_nodes(n, a, b)` - returns Chebyshev nodes and weights
- `generate_phase_nodes(r_start, r_end, k_total, Δφ)` - constant phase grid
- `oscillatory_kernel_integral_2d(..., method="filon")` - direct integrals
- `oscillatory_kernel_integral_2d(..., method="filon_exchange")` - exchange integrals

**Usage**:
```python
# CPU version with oscillatory quadrature
integrals = radial_ME_all_L(
    grid, V_core, U_i, orb_i, orb_f, chi_i, chi_f, L_max,
    use_oscillatory_quadrature=True
)

# GPU version with oscillatory quadrature (auto-selected when available)
integrals_gpu = radial_ME_all_L_gpu(
    grid, V_core, U_i, orb_i, orb_f, chi_i, chi_f, L_max,
    use_oscillatory_quadrature=True
)

# Advanced: Clenshaw-Curtis nodes for custom integration
from oscillatory_integrals import clenshaw_curtis_nodes, generate_phase_nodes
nodes, weights = clenshaw_curtis_nodes(7, 0, 100)  # 7 CC nodes on [0, 100]
phase_grid = generate_phase_nodes(0, 200, k_i + k_f)  # Constant Δφ grid
```

### General Performance Tips

- **Near-threshold**: Grid auto-scales, but consider using log energy grid
- **keV energies**: Ensure sufficient `L_max_projectile`
- **Turning point warning**: Logs show when L_max is limited; r_max is auto-increased
- **GPU**: Install CuPy for 5-10× speedup on radial integrals (auto-detected)
- **Parallel**: Code auto-detects CPU cores for multiprocessing




### Numerical Safeguards

The oscillatory integral module includes automatic safeguards:

| Safeguard | Limit | Purpose |
|-----------|-------|---------|
| MAX_REFINE_INTERVALS | 100 | Prevents infinite refinement loops |
| MAX_SUBDIVISIONS | 50/interval | Caps memory usage per interval |
| MAX_ARGUMENT (Si/Ci) | 10⁶ | Prevents special function overflow |
| NaN/Inf checking | Automatic | Falls back to standard method |

**Logging levels**: Set `DWBA_LOG_LEVEL=DEBUG` to see phase-adaptive details.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cross sections too small | Check (2π)⁴ factor in excitation and ionization |
| Near-threshold zeros | Use log grid, ensure energy > threshold + 0.5 eV |
| Slow runs | Reduce `n_points`, enable GPU |
| Solver failures (L>50) | Expected for very high L; code uses analytical bypass |
| "All solvers failed" | Check grid r_max vs turning point; increase r_max |
| Non-finite integral warnings | Indicates numerical issues; check input wavefunctions |
| Fit gives different results | Reference params are protected; fitted params may vary |

## References

1. **Article**: Lai et al., J. At. Mol. Sci. 5, 311-323 (2014)
2. **SAE Potential**: X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)
3. **Ionization TDCS**: S. Jones, D.H. Madison et al., Phys. Rev. A 48, 2285 (1993)
4. **Inner-shell ionization**: D. Bote, F. Salvat, Phys. Rev. A 77, 042701 (2008)

---
*Happy computing!*
