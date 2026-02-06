# DWBA – Distorted Wave Born Approximation Toolkit

Comprehensive Python suite for computing electron–atom excitation and ionization cross sections using the Distorted Wave Born Approximation (DWBA), with optional empirical calibration (Tong model) and plotting/diagnostic tools.

## Table of Contents
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Theory Snapshot](#theory-snapshot)
- [Units and Normalization](#units-and-normalization)
- [Requirements](#requirements)
- [Setup](#setup)
- [Core Workflows](#core-workflows)
  - [Batch Mode](#batch-mode-config-file)
  - [Interactive Driver](#interactive-driver-dw_mainpy)
  - [Excitation Scan](#excitation-scan)
  - [Ionization Scan](#ionization-scan)
  - [Plotting Results](#plotting-results)
  - [Partial Wave Analysis](#partial-wave-analysis-partial_wave_plotterpy)
  - [Potential Fitting](#potential-fitting)
- [Numerics and Parameters](#numerics-and-parameters)
  - [Centralized Default Parameters](#centralized-default-parameters)
  - [Parameter Reference](#parameter-reference)
    - [Grid Parameters](#grid-parameters)
    - [Excitation Parameters](#excitation-parameters)
    - [Ionization Parameters](#ionization-parameters)
    - [Oscillatory Integral Parameters](#oscillatory-integral-parameters)
    - [Hardware Parameters](#hardware-parameters)
    - [Output Parameters](#output-parameters)
  - [Adaptive Grid Strategies](#adaptive-grid-strategies-v26)
- [Atom Library (`atoms.json`)](#atom-library-atomsjson)
- [Debugging and Logging](#debugging-and-logging)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [References](#references)

## Features
- **GPU Acceleration**: Core radial matrix element calculations are GPU-accelerated via CuPy.
- **Scalable GPU Memory**: Implemented **block-wise construction** for radial kernels. This ensures a constant VRAM footprint, allowing large grids ($N_{grid}=15000$) to run on standard hardware.
- **Improved Phase Extraction**: Uses **5-point Fornberg stencils** for finite derivatives on non-uniform grids, ensuring stable phase extraction for high energies (1.0 keV+).
- **High-Accuracy Integrals**: Support for **Full-Split** quadrature with full-grid integration parity across CPU and GPU paths.
- **Bound State Extent Handling** *(v2.8)*: Automatic detection of bound state extent ensures match point is beyond 99% of density, critical for accurate oscillatory integral factorization.
- **Centrifugal Phase Corrections** *(v2.8)*: First-order centrifugal terms in asymptotic phase for stable high-L partial wave calculations.
- **LOCAL Adaptive Grid Strategy** *(v2.12)*: Per-energy optimal grid sizing now fully functional with turning point bounds safeguards.
- **Wavelength-Aware Grid Scaling** *(v2.14)*: Automatic grid density increase for high energies, ensuring adequate phase sampling in bound state and match point regions.
- **Physics-Based Convergence** *(v2.15)*: Partial wave sum convergence based on per-angle DCS stability, correctly handling interference-induced non-monotonicity.
- **Dual Top-Up Strategy** *(v2.16)*: Physics-based tail extrapolation using Coulomb-Bethe for E1 (dipole) transitions and Born geometric series for forbidden transitions.
- **Optimized Defaults** *(v2.16)*: RK45 solver as default (correct for exponential grids), reduced base `n_points` (adaptive scaling handles high energies).
- **Diagnostic Tools**: Comprehensive scripts in `debug/` for analyzing partial wave convergence, radial integrals, and method comparisons.
- **Progress Reporting**: Real-time feedback and ETA for long-running partial wave summations.


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
├── config_loader.py        # YAML configuration handling
├── logging_config.py       # Centralized logging configuration
├── output_utils.py         # Result directory management
├── atom_library.py         # Atom database interface
├── atoms.json              # SAE parameters for atoms/ions
├── nist_data.json          # NIST experimental benchmarks
├── fit_potential.py        # Potential parameter fitting tool
├── plotter.py              # Result plotting utility
├── partial_wave_plotter.py # Partial wave analysis plotter
├── debug/                  # Unified diagnostic framework
│   ├── debug.py            # Main diagnostic menu
│   ├── diag_amplitude.py   # Amplitude logic verification
│   ├── diag_angular.py     # Coupling coefficients check
│   ├── diag_bound.py       # Bound state solver health
│   ├── diag_upturn.py      # High-E convergence analysis
│   ├── diag_radial_integrals.py  # I_L multipole diagnostics
│   └── diag_oscillatory.py # Integral stability tests
├── article.md              # Reference theory paper
├── results/                # All output files (auto-created)
│   ├── results_*.json      # Calculation results
│   └── plot_*.png          # Generated plots
└── README.md               # This file
```

## Theory Snapshot

The core of the package implements the **Distorted-wave Born Approximation (DWBA)** for electron-impact transitions in the Single-Active-Electron (SAE) approximation.

### 1. Transition Amplitudes
The calculation evaluates both **direct ($f$)** and **exchange ($g$)** scattering amplitudes (Eq. 120, 134). These are computed via partial-wave expansion of the distorted continuum waves and the multipole-expanded interaction $1/r_{12}$:
- **Direct ($f$):** Evaluated using Eq. (412) of the article.
- **Exchange ($g$):** Evaluated using Eq. (448) of the article.

### 2. Differential Cross Section (DCS)
The unpolarized DCS for excitation (Eq. 226) includes proper spin-weighting for singlet and triplet channels:
$$\frac{d\sigma}{d\Omega} = N (2\pi)^4 \frac{k_f}{k_i} \frac{1}{2L_i+1} \sum_{M_i, M_f} \left[ \frac{3}{4}|f-g|^2 + \frac{1}{4}|f+g|^2 \right]$$
*Note: For Ionization, the (e, 2e) TDCS follows a similar $(2\pi)^4$ kinematic convention (Jones/Madison 1993).*

### 3. Potential Models
- **SAE Core Potential (Eq. 69):** Used to solve for target bound states $\Phi_j$ and the core potential:
  $$V_{A^+}(r) = -\frac{1 + a_1 e^{-a_2 r} + a_3 r e^{-a_4 r} + a_5 e^{-a_6 r}}{r}$$
- **Distorting Potentials (Eq. 457):** The distorted waves $\chi_k$ for the projectile are solved in a static-exchange potential $U_j$:
  $$U_j(r) = V_{A^+}(r) + \int \frac{|\Phi_j(\mathbf{r}')|^2}{|\mathbf{r} - \mathbf{r}'|} d\mathbf{r}' + V_{pol}(r)$$
  *Where $V_{pol}$ is an optional polarization term.*

### 4. Calibration (Tong / DWBA)

The calibration procedure corrects the overestimation of Total Cross Sections (TCS) at low energies by scaling DWBA results using an empirical BE-scaled formula.

#### Theoretical Basis
The empirical TCS for a given transition follows Eq. (493):
$$\sigma_{\text{Tong}}(E) = \alpha \cdot \frac{\pi}{\Delta E_{thr}^2} \exp\left[ \frac{1.5(\Delta E_{thr} - \epsilon)}{E} \right] f\left( \frac{E}{\Delta E_{thr}} \right) $$

where the shape function $f(x)$ represents the energy dependence (Eq. 480):
$$f(x) = \frac{1}{x} \left[ \beta \ln x + \gamma \left( 1 - \frac{1}{x} \right) + \delta \frac{\ln x}{x} \right]$$

**Parameters**:
- $\Delta E_{thr}$: Threshold excitation energy.
- $\epsilon$: Eigenenergy of the final excited state (e.g., $-0.125$ a.u. for H $2s$).
- $\alpha$: Matching coefficient, determined at $E_{ref} = 1000$ eV.
- **Sets**:
  - **Dipole-allowed** ($|\Delta l|=1$): $\beta=1.32, \gamma=-1.08, \delta=-0.04$
  - **Dipole-forbidden** ($|\Delta l|\neq 1$): $\beta=0.7638, \gamma=1.1759, \delta=0.6706$

#### Normalization Procedure
1. **Alpha Matching**: A "Pilot Run" is performed at 1000 eV to find $\alpha = \sigma_{DWBA}(1000) / \sigma_{\text{Tong}, \alpha=1}(1000)$.
2. **Per-energy scaling**: For each energy $E$, a normalization factor $C(E) = \sigma_{\text{Tong}}(E) / \sigma_{\text{DWBA}}(E)$ is derived.
3. **Calibrated Results**: Both raw and calibrated TCS are saved. Calibrated DCS are obtained via: $(d\sigma/d\Omega)_{\text{cal}} = C(E) \cdot (d\sigma/d\Omega)_{\text{raw}}$.

> [!NOTE]
> Calibration applies to **excitation only**. Ionization calculations use raw DWBA results.

> [!TIP]
> **Disabling Calibration (v2.10+):** Set `output.calibrate: false` in config or interactive mode to skip the pilot run. Results will use α = 1.0 and plots will show only the raw DWBA curve.

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

### Output Directory Structure

All output files are organized in dedicated directories:
- `results/` — Calculation results (JSON) and plots (PNG)
- `fited_potentials/` — Fitted potential plots from potential fitting tool

The directories are created automatically on first use.

### Batch Mode (Config File)

For automated calculations, use a YAML configuration file:

```bash
# Run with config file
python DW_main.py -c config.yaml

# Verbose mode
python DW_main.py -c config.yaml -v

# Generate template config
python DW_main.py --generate-config -o my_config.yaml
python DW_main.py --generate-config --config-type ionization -o ion_config.yaml
```

**Example Configuration**: See `examples/` directory for complete templates (`examples/config_excitation.yaml` and `examples/config_ionization.yaml`).

**Config File Discovery**: When starting an excitation or ionization calculation interactively, the program will prompt to use existing `.yaml` config files if found.

### Interactive Driver (`DW_main.py`)
```bash
python DW_main.py
```
Menu:
1. Excitation Cross Sections
2. Ionization Cross Sections
3. Total Cross Sections Plots
4. Angular DCS Plots
5. Partial Wave Analysis
6. Fit Potential (new atom)
7. Change Run Name

### Excitation Scan

Interactive workflow for calculating electron-impact excitation transitions.
- **Workflow**: Select target atom/ion $\to$ Define $(n_i, l_i) \to (n_f, l_f)$ transition $\to$ Choose physics model.
- **Calibration**: Automatic **Tong Model** pilot run at 1 keV used to derive the normalization factor $\alpha$.
- **Physics**: Hybrid DWBA treatment of exchange and static-exchange potentials.
- **Diagnostics**: Convergence of partial waves is monitored and logged in real-time.
- **Results**: Cross sections (TCS, DCS) are saved to `results/results_<run>_exc.json`.

### Ionization Scan

Comprehensive (e, 2e) pipeline for computing ionization cross sections.
- **Quantities**:
  - **SDCS**: Single Differential Cross Section $d\sigma/dE_{ej}$, integrated over both electron angles.
  - **TICS**: Total Ionization Cross Section, integrated over ejected energy.
  - **TDCS**: Triple Differential Cross Section for specified scattering/ejection angle triplets $(\theta_s, \theta_e, \phi_e)$ in the scattering plane ($\phi_s=0$).
- **Key Features**:
  - **Normalization**: Follows TDWBA convention with proper $(2\pi)^4$ kinematic factors.
  - **Exchange**: Handled via angle-swapping logic (Jones/Madison) for indistinguishable electrons.
  - **Automation**: Minimum $L_{max}=3$ floor and adaptive $L_{max}$ scaling based on incident momentum.
- **Calibration**: Normalization applies to **excitation only**; ionization plots and results show raw DWBA data.
- **Results**: Integrated and differential data saved to `results/results_<run>_ion.json`.

### Plotting Results

Generate plots from calculation results:
```bash
python plotter.py [style] [results_file.json]
```

Styles: `std` (eV/cm²), `atomic` (Ha/a₀²), `article` (E/E_thr), `ev_au`, `dcs`

### Partial Wave Analysis (`partial_wave_plotter.py`)

Interactive tool for analyzing partial wave contributions:
```bash
python partial_wave_plotter.py
```

Features:
- **File selection** — Choose from available result files in `results/`.
- **Convergence analysis** — Cumulative sum vs L across energies.
- **L\_90% indicator** — Shows L value at which sum reaches 90% of total.
- **Energy dependence** — Partial cross sections $\sigma_L(E)$ for selected waves.
- **Distribution plots** — Bar chart of contributions at specific energy.
- **Summary statistics** — Energy range, max L, total cross sections.

### Potential Fitting

Interactive tool for calibrating atomic core potentials to match NIST binding energies:
```bash
python fit_potential.py
```
Features:
- **Reference Protection**: Parameters from Tong-Lin (2005) are marked and protected.
- **Global Optimizer**: Uses `differential_evolution` for robust fitting.
- **Proper Bounds**: Support for complex core structures (negative $a_3, a_5$).
- **Auto-save**: Updates `atoms.json` directly upon successful fit.


## Numerics and Parameters

### Centralized Default Parameters

All numerical defaults are organized by category and displayed before calculation:

| Category | Key Parameters |
|----------|----------------|
| **Grid** | `strategy`, `r_max`, `n_points`, scale factors |
| **Excitation** | `L_max_integrals`, `L_max_projectile`, pilot settings |
| **Ionization** | `l_eject_max`, `L_max`, `n_energy_steps` |
| **Oscillatory** | `method`, `CC_nodes`, thresholds |
| **Hardware** | `gpu_block_size`, `gpu_memory_mode`, `n_workers` |
| **Output** | `save_dcs`, `save_partial`, `calibrate` |

<details>
<summary>📋 Full Parameter Display (click to expand)</summary>

```
┌─ GRID ─────────────────────────────────┐
│  strategy               = "global"     │
│  r_max                  = 200          │
│  n_points               = 1000         │
│  r_max_scale_factor     = 2.5          │
│  n_points_max           = 15000        │
│  min_points_per_wavelength = 15        │
└────────────────────────────────────────┘

┌─ EXCITATION ───────────────────────────┐
│  L_max_integrals        = 15           │
│  L_max_projectile       = 5            │
│  n_theta                = 200          │
│  pilot_energy_eV        = 1000         │
│  pilot_L_max_integrals  = "auto"       │
│  pilot_L_max_projectile = "auto"       │
│  pilot_n_theta          = 50           │
└────────────────────────────────────────┘

┌─ IONIZATION ───────────────────────────┐
│  l_eject_max            = 3            │
│  L_max                  = 15           │
│  L_max_projectile       = 50           │
│  n_energy_steps         = 10           │
└────────────────────────────────────────┘

┌─ OSCILLATORY ──────────────────────────┐
│  method                 = "advanced"   │
│  CC_nodes               = 5            │
│  phase_increment        = 1.571 (π/2)  │
│  min_grid_fraction      = 0.1          │
│  k_threshold            = 0.5          │
│  max_chi_cached         = 20           │
└────────────────────────────────────────┘

┌─ HARDWARE ─────────────────────────────┐
│  gpu_block_size         = "auto"       │
│  gpu_memory_mode        = "auto"       │
│  gpu_memory_threshold   = 0.8          │
│  n_workers              = "auto"       │
└────────────────────────────────────────┘

┌─ OUTPUT ───────────────────────────────┐
│  save_dcs               = true         │
│  save_partial           = true         │
│  calibrate              = true         │
└────────────────────────────────────────┘
```
</details>

## Parameter Reference

### Grid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `"global"` | Grid mode: `"manual"` = fixed params, `"global"` = adaptive for $E_{min}$, `"local"` = per-energy |
| `r_max` | 200 a.u. | Base maximum radius (`float`) or `"auto"` (adaptive only). For manual mode: must be numeric. |
| `n_points` | 1000 | Base grid points. For manual: fixed. For adaptive: minimum value. |
| `r_max_scale_factor` | 2.5 | Safety factor for turning point: $r_{max} = factor \times r_{turn}(L_{max})$ |
| `n_points_max` | 15000 | Maximum grid points (caps adaptive scaling, limits memory). *(v2.8: increased from 8000)* |
| `min_points_per_wavelength` | 15 | **(v2.7+)** Min pts/λ at large r. Increases $n_{points}$ for high energies. Set to 0 to disable. |

### Excitation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L_max_integrals` | 15 | Maximum multipole order L for radial Coulomb integrals. |
| `L_max_projectile` | 5 | Base partial wave $L_{max}$ for projectile. Automatically increased relative to $k \cdot r_{max}$ at runtime. |
| `n_theta` | 200 | Number of scattering angle points for DCS calculation ($0^\circ-180^\circ$). |
| `pilot_energy_eV` | 1000 | Reference energy for pre-calculating calibration $\alpha$. |
| `pilot_L_max_integrals` | `"auto"` | Dynamically scales based on `k × r_max × 0.6` |
| `pilot_L_max_projectile`| `"auto"` | Dynamically scales based on `k × r_max × 0.6` |
| `pilot_n_theta` | 50 | Angular grid for pilot run (TCS only). |

### Ionization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l_eject_max` | 3 | Maximum angular momentum of ejected electron ($0=s, 1=p, 2=d, 3=f$). |
| `L_max` | 15 | Maximum multipole order for ionization integrals. |
| `L_max_projectile` | 50 | Maximum projectile angular momentum (ionization requires more partial waves). |
| `n_energy_steps` | 10 | Number of ejected electron energy points for $d\sigma/dE_{ej}$ integration. |

### Oscillatory Integral Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"advanced"` | Quadrature method: `legacy` (CC only), `advanced` (CC+Filon), `full_split` (most accurate) |
| `CC_nodes` | `5` | Clenshaw-Curtis nodes per oscillation interval |
| `phase_increment` | `π/2` | Phase increment for sub-interval division |
| `min_grid_fraction` | `0.1` | Minimum match point fraction: `r_m ≥ 0.1 × r_max` |
| `k_threshold` | `0.5` | Momentum threshold for Filon/Levin activation |
| `max_chi_cached` | `20` | LRU cache size for continuum waves (v2.5+) |
| `phase_extraction` | `"hybrid"` | Phase extraction method: `hybrid` (cross-validated), `logderiv`, `lsq` *(v2.11+)* |
| `solver` | `"rk45"` | ODE solver: `auto` (physics-based), `rk45` (recommended), `johnson`, `numerov` *(v2.13+)* |

---

### Hardware Parameters

> [!TIP]
> These parameters are now in a separate `hardware:` section in YAML configuration (v2.10+).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_block_size` | `"auto"` | GPU block size. `"auto"` = tune based on VRAM, or explicit `int` |
| `gpu_memory_mode` | `"auto"` | GPU strategy: `auto` / `full` / `block` |
| `gpu_memory_threshold` | `0.8` | Max VRAM fraction for matrix allocation |
| `n_workers` | `"auto"` | CPU workers: `"auto"` / `"max"` / explicit `int` |

**GPU Memory Modes:**
- **auto** — Check VRAM, use full matrix if fits, else block-wise *(recommended)*
- **full** — Force full matrix construction *(fastest, may OOM)*
- **block** — Force block-wise processing *(slowest, constant memory)*

---

### Output Parameters

> [!NOTE]
> New in v2.10. Controls what gets saved and whether calibration is applied.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_dcs` | `true` | Save differential cross section (θ, dσ/dΩ) data |
| `save_partial` | `true` | Save partial wave contributions (σ_L) |
| `calibrate` | `true` | Apply Tong model calibration (pilot run + α matching) |

**When `calibrate: false`:**
- Pilot calculation is skipped (faster)
- Uses α = 1.0 (no empirical correction)
- Plots show only raw DWBA curve (no Tong reference)

### Adaptive Grid Strategies (v2.6+)

The code supports three strategies for determining the radial grid (`r_max`, `n_points`):

1. **Global (Default)**
   - Calculates optimal grid parameters based on the *lowest* energy in the scan.
   - Pre-calculates target properties once.
   - **Pros**: Balanced performance/accuracy, single target prep.
   - **Cons**: Excessive grid size for high-energy points in wide scans.

2. **Local** *(v2.12 bounds fix)*
   - Recalculates optimal grid parameters for *each* energy point.
   - Re-runs target preparation (bound states, core potential) for every point.
   - **Pros**: Most accurate, optimal grid size per energy.
   - **Cons**: Slower due to repeated target preparation.
   - *Note: v2.12 fixed turning point overflow when high-L waves extend beyond small grids.*

3. **Manual**
   - Uses fixed `r_max` and `n_points` defined in configuration.
   - **Pros**: Complete user control, predictable memory usage.
   - **Cons**: Risk of insufficient grid for low energies or high partial waves.

### GPU Computation Modes

When using Filon quadrature with GPU acceleration, three computation modes are available:

1. **Filon/full-matrix** — Extended kernel matrix `(idx_limit × N_grid)` fits in VRAM. Single `cp.dot()` for entire integration. *Fastest.*

2. **Filon/hybrid** — Standard matrix `(idx_limit × idx_limit)` built, tail computed block-wise. Falls back when extended matrix doesn't fit.

3. **Filon/block-wise** — No prebuilt matrix, all computations block-by-block. Used when memory is very limited.

The mode is logged once at the start of GPU calculations:
```
GPU mode: Filon/full-matrix
```

Outputs: `results_<run>_exc.json`, `results_<run>_ion.json` in project root. Excitation entries include angular grids (`theta_deg`) and both raw/calibrated DCS in a.u. for later plotting. Ionization entries include SDCS data and optional TDCS entries (`angles_deg`, `values`).

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

## Debugging and Logging

### Log Levels
Set via environment variable:
```bash
set DWBA_LOG_LEVEL=DEBUG    # Windows
export DWBA_LOG_LEVEL=DEBUG # Linux/Mac
```
Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Comprehensive Diagnostic Suite

The `debug/` folder contains a unified diagnostic framework for verifying all aspects of DWBA calculations.

**Launch the diagnostic menu:**
```bash
python debug/debug.py
```
<details>
<summary>📋 Full Menu Structure (click to expand)</summary>

**Menu Structure:**
```
DWBA COMPREHENSIVE DIAGNOSTIC SUITE
=====================================
[QUICK TESTS]
  1. Quick Health Check         - Runs excitation + ionization test
  2. Convergence Study          - Theta and grid convergence analysis

[BOUND STATES]
  3. Bound State Analysis       - Norms, energies, nodes, orthogonality

[CONTINUUM WAVES]
  4. Continuum Wave Analysis    - Phase shifts, asymptotic amplitude
  5. Phase Extraction           - Multi-point phase stability diagnostic
  6. High-L Stability Scan      - Scan up to L=30 for instabilities

[POTENTIALS]
  7. Potential Analysis         - Core potential V(r) and Z_eff(r)
  8. Multi-Atom Comparison      - Compare H, Li, Na potentials

[RADIAL INTEGRALS]
  9. Radial Integral Breakdown  - I_L per multipole analysis
 10. Method Comparison          - Legacy vs Advanced oscillatory methods

[ANGULAR COUPLING]
 11. Angular Coefficient Check  - Verify CG and Racah coefficients

[CROSS SECTIONS]
 12. Cross Section Analysis     - σ(E) and partial wave contributions
 13. L0/L1 Anomaly              - Investigate 11-17 eV anomaly
 14. High-Energy Upturn         - Analyze 69.26 eV upturn phenomenon

[FULL TRACES]
 15. Full Physics Trace (Exc)   - Complete excitation pipeline trace
 16. Full Physics Trace (Ion)   - Complete ionization pipeline trace

[BATCH]
 20. Run ALL Diagnostics        - Execute all tests with summary

  q. Quit
```
</details>

**Output:**
- Results saved to `debug/results/` as timestamped JSON files
- Plots saved to `debug/plots/`

### Individual Diagnostic Scripts

| Script | Purpose |
|--------|---------|
| `diag_bound.py` | Standalone bound state analysis |
| `diag_angular.py` | Angular coupling coefficient tests |
| `diag_amplitude.py` | Amplitude contribution per M channel |
| `diag_phase_extraction.py` | Phase shift analysis with plots |
| `diag_radial_integrals.py` | I_L breakdown for various energies |
| `diag_method_compare.py` | Legacy vs Advanced method comparison |
| `diag_upturn.py` | High-energy partial wave convergence |
| `diag_L0_L1_anomaly.py` | L0/L1 ratio investigation |
| `diag_atoms.py` | Multi-atom potential comparison |
| `diag_oscillatory.py` | Oscillatory integral function tests |
| `diag_partial_waves.py` | Analyze partial waves from result JSON |


## Performance Tips

### Grid Numerics: $L_{max}$ and $r_{max}$ Selection

The classical turning point constraint is critical for numerical stability:

$$r_t(L) = \frac{L + 0.5}{k}$$    

$$L_{max} \le k \cdot \frac{r_{max}}{C} - 0.5 \quad (C \approx 2.5)$$    

The code automatically enforces this via `compute_safe_L_max()`:
- At **low energies** (small $k$): $L_{max}$ is automatically reduced
- If you see **warnings about turning point limits**: increase $r_{max}$ or accept fewer partial waves

| Energy (eV) | $k$ (a.u.) | $r_{max}=200$ | $r_{max}=500$ | $r_{max}=1000$ |
|-------------|------------|---------------|---------------|----------------|
| 15          | 1.05       | $L \le 33$    | $L \le 83$    | $L \le 167$    |
| 50          | 1.92       | $L \le 61$    | $L \le 153$   | $L \le 306$    |
| 100         | 2.71       | $L \le 86$    | $L \le 216$   | $L \le 433$    |

### Adaptive Grid (Automatic $r_{max}$ Scaling)

The code automatically computes optimal grid parameters via `calculate_optimal_grid_params()`:

**Step 1: Turning Point + Coulomb Requirement**
$$r_{max} \ge C \cdot \frac{L_{eff} + 0.5}{k} \quad (C = 2.5)$$

For ionic targets, also enforce Coulomb asymptotic validity:
$$r_{max} \ge \frac{3\max(L_{eff}, |\eta|)}{k}, \quad \eta = \frac{|z_{ion}|}{k}$$

**Step 2: Wavelength Sampling (v2.7+)**

For high-energy scans, ensure sufficient points per wavelength:
$$\lambda = \frac{2\pi}{k}, \quad dr_{max} = \frac{\lambda}{N_{pts/\lambda}}$$

At $r_{check} = 15$ a.u. (bound-state region), the exponential grid step is:
$$dr(r) \approx r \cdot \frac{\ln(r_{max}/r_{min})}{n_{points}}$$

Required points: $n_{pts} \ge r_{check} \cdot \ln(ratio) / dr_{max}$

A relaxed secondary check is applied near $r \approx 50$ a.u. for match-point stability.

**Step 3: Density Scaling**

If $r_{max}$ increases beyond base, scale points proportionally:
$$n_{points} = \max\left(n_{base}, \; \frac{n_{base}}{r_{base}} \cdot r_{max}\right)$$

For `r_max: "auto"`, density scaling uses a fixed reference $r_{base}=200$ a.u.

**Final**: Apply memory cap `n_points_max` (default 15000).

### Radial Solver: Numerical Methods

The continuum wave solver uses multiple methods with physics-based selection:

**RK45 Method** (Default for most cases):
- Standard Runge-Kutta-Fehlberg 4(5) with adaptive step control
- Best accuracy on **non-uniform (exponential) grids**
- Automatic error estimation and step adjustment

**Johnson Log-Derivative** (For tunneling regions):
- Propagates $Y = \chi'/\chi$ directly
- Numerically stable when $\chi \to 0$ (inside barrier)
- Preferred for high-L or low-energy cases

**Numerov Method** (Uniform grids only):
- Solves: $\chi''(r) = Q(r) \cdot \chi(r)$
- $O(h^4)$ accuracy but requires **constant step size**
- Not recommended for standard exponential grids

**Fornberg Phase Extraction** (v2.2+):
- Replaced 3-point central differences with a **5-point Fornberg stencil**
- Correctly computes weights for arbitrary grid spacing, suppressing numerical noise in the logarithmic derivative $Y(r) = \chi'(r)/\chi(r)$ at match points
- Essential for stable extraction of phase shifts $\delta_l$ at high energies ($k \gg 1$)

**Physics-Based Turning Point Detection**:
- Checks $S(r_{min}) = l(l+1)/r^2 + 2U - k^2$ at grid origin
- If $S > 0$ (inside barrier): starts propagation at $r = 0.9 \times r_{turn}$
- Works for **any $l$** when physics requires it (not just $l > 5$)

**Adaptive Initial Conditions**:
- Always evaluates $S(r_{start})$ to choose between:
  - **WKB**: $\chi \sim \exp(\kappa r)$ when $S > 0$ (inside barrier)
  - **Regular**: $\chi \sim r^{l+1}$ when $S < 0$ (oscillatory region)

**Match Point Selection** (Critical for high $L$):
- Searches **forward** from `idx_start + 50` to guarantee valid wavefunction
- Ensures $r_m > r_{turn}$ (past classical turning point)
- Uses relaxed threshold: $|U|/(k^2/2) < 1\%$ (was 0.01%)
- Prevents "all solvers failed" errors for high partial waves

**Phase Extraction**:
$$\tan(\delta_l) = \frac{Y_m \cdot \hat{j}_l - \hat{j}_l'}{\hat{n}_l' - Y_m \cdot \hat{n}_l}$$
- Matches to Riccati-Bessel (neutral) or Coulomb $F$, $G$ (ionic) at $r_m$
- Log-derivative $Y_m = \chi'/\chi$ used for numerical stability

**Asymptotic Stitching**:
- Numerical $\chi$ scaled to match asymptotic amplitude at $r_m$
- Pure analytic solution $A[\hat{j}\cos\delta - \hat{n}\sin\delta]$ used for $r > r_m$
- Amplitude $A = \sqrt{2/\pi}$ for $\delta(k-k')$ normalization
- Eliminates numerical noise in oscillatory tail

**Solver Selection** *(v2.13+)*:

| Grid Type | Recommended | Notes |
|-----------|-------------|-------|
| **Exponential** (default) | `"rk45"` | Only correct solver for variable step size |
| **Uniform** (`linspace`) | `"numerov"` | O(h⁴) accuracy, fastest |

**Johnson Renormalized Numerov** *(v2.13 rewrite)*:
- Based on B.R. Johnson, J. Chem. Phys. 69, 4678 (1978)
- Propagates ratio $R_n = \psi_{n-1}/\psi_n$ via $R_{n+1} = 1/(T_n - R_n)$
- Stable for high-L but requires uniform grid

> [!CAUTION]
> Numerov and Johnson require uniform grids. On exponential grids, errors up to 1.3 rad.

**Fallback Chain**:
If primary fails → next solver in chain (auto-selected order)

**Split Radial Integrals**:
- Integration uses numerical $\chi$ for $[0, r_m]$ and analytic for $[r_m, \infty)$
- Match point stored in `ContinuumWave.idx_match`

### Oscillatory Radial Integrals

For high partial waves and energies, the radial integrands oscillate rapidly:

$$\chi_i(k_i, r) \times \chi_f(k_f, r) \times K_L(r_1, r_2)$$

The **oscillatory_integrals.py** module provides advanced oscillatory quadrature methods following best practices for high-accuracy scattering calculations.

#### Domain Splitting: $I_{in} + I_{out}$

All integrals are split at the match point $r_m$:
- **$I_{in}$ $[0, r_m]$**: Clenshaw-Curtis / Simpson quadrature with proper integration weights
- **$I_{out}$ $[r_m, \infty)$**: Specialized oscillatory methods (Levin/Filon) using asymptotic wave forms

#### High-Accuracy Integration (v2.2+)

Recent audits (Edit_62+) have standardized the **Full-Split** and **Advanced** methods for maximum parity between CPU and GPU:

1.  **Multipole Moment ($M_L$)**: The asymptotic envelope coefficient is now computed using the **full grid** weights `w`:
    $M_L = \int_0^{R_{max}} w(r) \cdot r^L \cdot u_f(r) u_i(r) dr$.
    This ensures that the bound-state contribution living beyond the match point $r_m$ is fully captured.
2.  **Inner Integral Parity**: The inner $r_2$ integral for both **Direct** and **Exchange** terms is now computed over the full range $[0, R_{max}]$ on both CPU and GPU. This ensures that the entire localized potential of the target atom is integrated when computing the field felt by the projectile.

**Important**: 2D kernel integrals use:
$$I = \iint \rho_1(r_1) \cdot K(r_1, r_2) \cdot \rho_2(r_2) \, dr_1 \, dr_2$$
Integration weights `w_grid` (Simpson's rule) are applied to both dimensions.

#### $\sin A \times \sin B$ Decomposition

For products of continuum waves $\chi_a \times \chi_b$, the identity is applied:
$$\sin(\Phi_a) \times \sin(\Phi_b) = \frac{1}{2}\left[\cos(\Phi_a - \Phi_b) - \cos(\Phi_a + \Phi_b)\right]$$

This separates the integral into two cosine terms $I_{-}$ and $I_{+}$, each computed using complex exponentials:
$$I = \text{Re} \int f(r) \exp(i\Phi(r)) \, dr$$

**Key functions:**
- `compute_product_phases()` — decomposes wave parameters into $k_\pm$, $\varphi_\pm$, $\eta_\pm$
- `dwba_outer_integral_1d()` — full $\sin A \times \sin B$ integral using Filon/Levin

#### Filon Quadrature (Linear Phase)

For regions where phase is approximately linear ($|\Phi''| \times h^2 < 0.1$):
- Divides into segments with constant phase increment $\Delta\Phi = \pi/4$
- Uses complex exponential form: $\int f(r) \exp(i\omega r + \varphi_0) \, dr$
- Taylor expansion for small $\omega h$ to avoid division issues
- Polynomial interpolation of envelope $f(r)$

```python
from oscillatory_integrals import filon_oscillatory_integral
I = filon_oscillatory_integral(f_func, omega, phase_offset, r_start, r_end)
```

#### Levin Collocation (Nonlinear Phase)

For Coulomb phases where $\eta \ln(2kr)$ makes $\Phi$ nonlinear:
- Solves ODE: $u'(r) + i\Phi'(r)u(r) = f(r)$
- Uses boundary formula: $I = u(b)e^{i\Phi(b)} - u(a)e^{i\Phi(a)}$
- Chebyshev collocation with 8 nodes per segment
- Robust for highly oscillatory integrands with variable frequency

```python
from oscillatory_integrals import levin_oscillatory_integral
I = levin_oscillatory_integral(f_func, phi_func, phi_prime_func, r_start, r_end)
```

#### Automatic Method Selection

The unified interface `compute_outer_integral_oscillatory()` automatically selects:
- **Filon** when $|\Phi''| \times h^2 < 0.1$ (constant frequency)
- **Levin** when $|\Phi''| \times h^2 \ge 0.1$ (variable frequency)

```python
from oscillatory_integrals import compute_outer_integral_oscillatory
I = compute_outer_integral_oscillatory(f, phi, phi_prime, phi_prime2, r_m, r_max)
```

#### Numerical Stability

**Kahan Summation**: Used for segment accumulation to reduce roundoff errors:
```python
from oscillatory_integrals import _kahan_sum_complex
total = _kahan_sum_complex(segment_contributions)  # Compensated summation
```

**Phase Computation Helpers**:
```python
from oscillatory_integrals import compute_asymptotic_phase, compute_phase_derivative
phi = compute_asymptotic_phase(r, k, l, delta, eta, sigma)  # Full Coulomb phase
phi_prime = compute_phase_derivative(r, k, eta)  # Φ'(r) = k + η/r
```

#### Advanced API

| Function | Purpose |
|----------|---------|
| `clenshaw_curtis_nodes(n, a, b)` | Chebyshev nodes and weights |
| `generate_phase_nodes(r_start, r_end, k_total, dphi)` | Constant phase grid |
| `_filon_segment_complex(f, r, omega, phi0)` | Single segment Filon |
| `_levin_segment_complex(f, r, phi, phi_prime)` | Single segment Levin |
| `compute_product_phases(...)` | $\sin A \times \sin B \to \cos$ terms |
| `dwba_outer_integral_1d(...)` | Full DWBA outer integral |

**Usage**:
```python
# Full DWBA calculation with advanced quadrature
integrals = radial_ME_all_L(
    grid, V_core, U_i, orb_i, orb_f, chi_i, chi_f, L_max,
    use_oscillatory_quadrature=True
)
```

### General Performance Tips

- **Near-threshold**: Grid auto-scales, but consider using log energy grid
- **keV energies**: Ensure sufficient `L_max_projectile`
- **Turning point warning**: Logs show when L_max is limited; r_max is auto-increased
- **GPU**: Install CuPy for 5-10× speedup on radial integrals (auto-detected)
- **Parallel**: Code auto-detects CPU cores for multiprocessing

### Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| CC weights (vectorized) | ~0.1 ms | Cached at module level |
| Filon integration | ~0.9 ms | 1000 points |
| Filon Exchange | ~3.3 ms | 500 points |
| GPU radial integrals | 5-10× faster | Pure GPU path |

#### GPU Block-wise Architecture

To prevent system memory exhaustion on large grids:
- Kernels are built in **blocks of 8192 rows** (default).
- Each block calculates a subset of the $r_1$ outer integral.
- Memory for each block is explicitly freed from the CuPy pool before the next block begins.
- Result: **Constant VRAM usage**, independent of $N_{grid}$.

#### GPU Cache Architecture (v2.5+)

The `GPUCache` class provides **energy-level resource reuse** to minimize GPU transfers:

**Cached Resources:**
- `r_gpu`, `w_gpu` — Persistent grid arrays (full grid)
- `chi_cache` — LRU-managed continuum wave cache (max 20 entries)

**Synchronization Reduction:**
- Radial matrix elements accumulated in GPU arrays (`I_L_dir_gpu`, `I_L_exc_gpu`)
- Single `.get()` transfer at end of L-loop instead of ~2×L_max syncs per call
- L=0 correction terms precomputed on GPU

**Usage:**
```python
# Created once per energy point in driver.py
gpu_cache = GPUCache.from_grid(grid, max_chi_cached=20)

# Passed to all radial_ME_all_L_gpu calls
integrals = radial_ME_all_L_gpu(..., gpu_cache=gpu_cache)

# Cleanup at end of energy point
gpu_cache.clear()
```

**Configuration:**
- `max_chi_cached: 20` — Maximum cached continuum waves (LRU eviction)
- Automatically adapts to available VRAM

**Algorithmic Optimizations:**
- Module-level CC weight caching (~25% speedup)
- Vectorized kernel interpolation (searchsorted + linear)
- Pre-computed ratio/log_ratio matrices for kernel construction
- Phase-adaptive subdivision with constant Δφ = π/2
- **GPU:** Pure-GPU interpolation via `cp.interp` (no CPU transfers)

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
| Solver failures (L>50) | Normal for very high L; code uses analytical bypass |
| "All solvers failed" | Fixed in v2.1 - match point now guaranteed valid |
| Phase unstable warnings | Usually near L cutoff or undersampled grid; diagnostics compare post-match asymptotic points (`r_m` vs `r_m + 5 a.u.`) |
| "r_m not in asymptotic region" | Fixed in v2.1 - relaxed threshold to 1% |
| Non-finite integral warnings | Check input wavefunctions; reduce L_max |
| LOCAL "index out of bounds" | Fixed in v2.12 - turning point bounds now clamped |

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and detailed changes.

## References

1. **Article**: Lai et al., J. At. Mol. Sci. 5, 311-323 (2014)
2. **SAE Potential**: X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)
3. **Ionization TDCS**: S. Jones, D.H. Madison et al., Phys. Rev. A 48, 2285 (1993)
4. **Inner-shell ionization**: D. Bote, F. Salvat, Phys. Rev. A 77, 042701 (2008)

---
*Happy computing!*
