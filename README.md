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
- **GPU Acceleration**: Core radial matrix element calculations are GPU-accelerated via CuPy.
- **Scalable GPU Memory**: Implemented **block-wise construction** for radial kernels (default 8192 points). This ensures a constant VRAM footprint, allowing large grids ($N_{grid}=10000$) to run on standard hardware without "Pagefile too small" errors.
- **Improved Phase Extraction**: Uses **5-point Fornberg stencils** for finite derivatives on non-uniform grids, ensuring stable phase extraction for high energies (1.0 keV+).
- **High-Accuracy Integrals**: Support for **Full-Split** quadrature with full-grid integration parity across CPU and GPU paths.
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
├── output_utils.py         # Centralized output path management
├── atom_library.py         # Atom database interface
├── atoms.json              # SAE parameters for atoms/ions
├── nist_data.json          # NIST reference energies
├── fit_potential.py        # Potential parameter fitting tool
├── plotter.py              # Result plotting utility
├── partial_wave_plotter.py # Partial wave analysis plotter
├── logging_config.py       # Centralized logging
├── debug.py                # Health checks and diagnostics
├── article.md              # Reference paper (theory)
├── results/                # All output files (auto-created)
│   ├── results_*.json      # Calculation results
│   └── plot_*.png          # Generated plots
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
3. Total Cross Sections Plots
4. Angular DCS Plots
5. Partial Wave Analysis
6. Fit Potential (new atom)
7. Change Run Name

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

**Example Configuration** (`examples/config_excitation.yaml`):
```yaml
run_name: "H_1s2s_batch"

calculation:
  type: "excitation"

target:
  atom: "H"

states:
  initial: {n: 1, l: 0}
  final: {n: 2, l: 0}

energy:
  type: "log"
  start_eV: 10.2
  end_eV: 300
  density: 0.5

oscillatory:
  method: "advanced"
  gpu_memory_mode: "auto"
```

See `examples/` directory for complete templates.

**Config File Discovery**: When starting an excitation or ionization calculation interactively, the program will prompt to use existing `.yaml` config files if found.

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
- **L_max floor**: Minimum L_max=3 ensures s-, p-, d-wave contributions at all energies, even near threshold.
- Polarization option is heuristic and not part of the article DWBA.
- **k_threshold**: When k_total > 0.5 a.u., specialized Filon/Levin oscillatory quadrature is used; below this threshold, standard Simpson integration is faster and sufficiently accurate.

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
- **File selection** — Choose from available result files in `results/`
- **Convergence analysis** — Cumulative sum vs L across energies
- **L_90% indicator** — Shows L value at which sum reaches 90% of total
- **Energy dependence** — Partial cross sections σ_L(E) for selected waves
- **Distribution plots** — Bar chart of contributions at specific energy
- **Summary statistics** — Energy range, max L, total cross sections


### Centralized Default Parameters

All numerical defaults are organized by category and displayed before calculation:

```
  ┌─ GRID ─────────────────────────────
  │  r_max                  = 200
  │  n_points               = 3000
  │  r_max_scale_factor     = 2.5
  │  n_points_max           = 10000
  └────────────────────────────────────

  ┌─ EXCITATION ───────────────────────
  │  L_max_integrals        = 15
  │  L_max_projectile       = 5
  │  n_theta                = 200
  │  pilot_energy_eV        = 1000
  └────────────────────────────────────

  ┌─ OSCILLATORY ──────────────────────
  │  method                 = "advanced"
  │  CC_nodes               = 5
  │  phase_increment        = 1.571
  │  min_grid_fraction      = 0.1
  │  k_threshold            = 0.5
  │  gpu_block_size         = auto
  │  gpu_memory_mode        = "auto"
  │  gpu_memory_threshold   = 0.7
  │  n_workers              = auto
  └────────────────────────────────────
```

When prompted, enter:
- **Y** (or Enter): Use all defaults unchanged
- **n**: Edit any parameter individually
- **d**: Display parameter details before deciding

---

## Parameter Reference

### Grid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_max` | 200 a.u. | Maximum radius of radial grid. Automatically scaled based on projectile wavelength. |
| `n_points` | 3000 | Base number of radial grid points. Uses exponential spacing for efficiency. |
| `r_max_scale_factor` | 2.5 | Multiplier for adaptive r_max scaling: `r_max = factor × max(classical_turning_point, wavelength)` |
| `n_points_max` | 10000 | Maximum allowed grid points (limits memory usage for high-energy calculations). |

### Excitation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L_max_integrals` | 15 | Maximum multipole order L for radial Coulomb integrals. Higher values improve accuracy for monopole/dipole/quadrupole coupling. Convergence is typically achieved at L=10-20. |
| `L_max_projectile` | 5 | Maximum angular momentum for projectile partial waves (initial/final). Automatically increased based on classical turning point at runtime. |
| `n_theta` | 200 | Number of scattering angle points for DCS calculation (0°-180°). |
| `pilot_energy_eV` | 1000 | Reference energy for pre-calculating convergence parameters. |

### Ionization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l_eject_max` | 3 | Maximum angular momentum of ejected electron (0=s, 1=p, 2=d, 3=f). |
| `L_max` | 15 | Maximum multipole order for ionization integrals. |
| `L_max_projectile` | 50 | Maximum projectile angular momentum (ionization requires more partial waves). |
| `n_energy_steps` | 10 | Number of ejected electron energy points for SDCS integration. |

### Oscillatory Integral Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | "advanced" | Oscillatory quadrature method: |
| | | • **legacy**: Clenshaw-Curtis only (fastest, less accurate at high k) |
| | | • **advanced**: CC + Levin/Filon tail correction (balanced, recommended) |
| | | • **full_split**: Full I_in/I_out separation with oscillatory tails (most accurate) |
| `CC_nodes` | 5 | Clenshaw-Curtis nodes per oscillation interval. Higher = more accurate but slower. |
| `phase_increment` | π/2 | Phase increment for sub-interval division in oscillatory integrals. |
| `min_grid_fraction` | 0.1 | Minimum match point as fraction of grid: `r_m ≥ 0.1 × r_max`. |
| `k_threshold` | 0.5 | Total momentum threshold for Filon quadrature: if `k_i + k_f > threshold`, use Filon. |

### GPU Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_block_size` | auto (0) | GPU block size for matrix operations. `auto` computes optimal size based on VRAM. Explicit value (e.g., 4096) overrides. |
| `gpu_memory_mode` | "auto" | GPU memory strategy: |
| | | • **auto**: Check VRAM, use full matrix if possible, else block-wise (recommended) |
| | | • **full**: Force full matrix construction (fastest, may OOM) |
| | | • **block**: Force block-wise (slowest, constant memory) |
| `gpu_memory_threshold` | 0.7 | Maximum fraction of free GPU memory to use for matrix allocation. |
| `n_workers` | "auto" | CPU worker count for multiprocessing: |
| | | • **auto**: Auto-detect (`min(cpu_count, 8)`) |
| | | • **N**: Explicit count (capped at cpu_count) |

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
- O(h⁴) accuracy for non-uniform (exponential) grids
- Uses **separate step sizes** h₁², h₂² instead of averaged h² for better accuracy
- Periodic renormalization prevents over/underflow

**Fornberg Phase Extraction** (v2.2+):
- Replaced 3-point central differences with a **5-point Fornberg stencil**.
- Correctly computes weights for arbitrary grid spacing, suppressing numerical noise in the logarithmic derivative $Y(r) = \chi'(r)/\chi(r)$ at match points.
- Essential for stable extraction of phase shifts $\delta_l$ at high energies ($k \gg 1$).

**Physics-Based Turning Point Detection**:
- Checks S(r_min) = l(l+1)/r² + 2U - k² at grid origin
- If S > 0 (inside barrier): starts propagation at r = 0.9 × r_turn
- Works for **any l** when physics requires it (not just l > 5)

**Adaptive Initial Conditions**:
- Always evaluates S(r_start) to choose between:
  - **WKB**: χ ~ exp(κ·r) when S > 0 (inside barrier)
  - **Regular**: χ ~ r^(l+1) when S < 0 (oscillatory region)

**Match Point Selection** (Critical for high L):
- Searches **forward** from idx_start + 50 to guarantee valid wavefunction
- Ensures r_m > r_turn (past classical turning point)
- Uses relaxed threshold: |U|/(k²/2) < 1% (was 0.01%)
- Prevents "all solvers failed" errors for high partial waves

**Phase Extraction**:
```
tan(δ_l) = [Y_m · ĵ_l - ĵ_l'] / [n̂_l' - Y_m · n̂_l]
```
- Matches to Riccati-Bessel (neutral) or Coulomb F,G (ionic) at r_m
- Log-derivative Y_m = χ'/χ used for numerical stability

**Asymptotic Stitching**:
- Numerical χ scaled to match asymptotic amplitude at r_m
- Pure analytic solution A·[ĵ cos(δ) - n̂ sin(δ)] used for r > r_m
- Amplitude A = √(2/π) for δ(k-k') normalization
- Eliminates numerical noise in oscillatory tail

**Fallback Chain**:
If Numerov fails → Johnson log-derivative → RK45

**Split Radial Integrals**:
- Integration uses numerical χ for [0, r_m] and analytic for [r_m, ∞)
- Match point stored in `ContinuumWave.idx_match`

### Oscillatory Radial Integrals

For high partial waves and energies, the radial integrands oscillate rapidly:

```
χ_i(k_i, r) × χ_f(k_f, r) × K_L(r₁, r₂)
```

The **oscillatory_integrals.py** module provides advanced oscillatory quadrature methods following best practices for high-accuracy scattering calculations.

#### Domain Splitting: I_in + I_out

All integrals are split at the match point r_m:
- **I_in [0, r_m]**: Clenshaw-Curtis / Simpson quadrature with proper integration weights.
- **I_out [r_m, ∞)**: Specialized oscillatory methods (Levin/Filon) using asymptotic wave forms.

#### High-Accuracy Integration (v2.2+)

Recent audits (Edit_62+) have standardized the **Full-Split** and **Advanced** methods for maximum parity between CPU and GPU:

1.  **Multipole Moment ($M_L$)**: The asymptotic envelope coefficient is now computed using the **full grid** weights `w`:
    $M_L = \int_0^{R_{max}} w(r) \cdot r^L \cdot u_f(r) u_i(r) dr$.
    This ensures that the bound-state contribution living beyond the match point $r_m$ is fully captured.
2.  **Inner Integral Parity**: The inner $r_2$ integral for both **Direct** and **Exchange** terms is now computed over the full range $[0, R_{max}]$ on both CPU and GPU. This ensures that the entire localized potential of the target atom is integrated when computing the field felt by the projectile.

**Important**: 2D kernel integrals use:
```
I = ∫∫ ρ₁(r₁) · K(r₁,r₂) · ρ₂(r₂) dr₁ dr₂
```
Integration weights `w_grid` (Simpson's rule) are applied to both dimensions to ensure proper ∫dr integration.

#### sinA × sinB Decomposition

For products of continuum waves χ_a × χ_b, the identity is applied:
```
sin(Φ_a) × sin(Φ_b) = ½[cos(Φ_a - Φ_b) - cos(Φ_a + Φ_b)]
```
This separates the integral into two cosine terms `I_minus` and `I_plus`, each computed using complex exponentials:
```python
I = Re ∫ f(r) exp(iΦ(r)) dr
```

**Key functions:**
- `compute_product_phases()` - decomposes wave parameters into k_±, φ_±, η_±
- `dwba_outer_integral_1d()` - full sinA×sinB integral using Filon/Levin

#### Filon Quadrature (Linear Phase)

For regions where phase is approximately linear (|Φ''| × h² < 0.1):
- Divides into segments with constant phase increment ΔΦ = π/4
- Uses complex exponential form: ∫ f(r) exp(iωr + φ₀) dr
- Taylor expansion for small ω·h to avoid division issues
- Polynomial interpolation of envelope f(r)

```python
from oscillatory_integrals import filon_oscillatory_integral
I = filon_oscillatory_integral(f_func, omega, phase_offset, r_start, r_end)
```

#### Levin Collocation (Nonlinear Phase)

For Coulomb phases where η·ln(2kr) makes Φ nonlinear:
- Solves ODE: u'(r) + iΦ'(r)u(r) = f(r)
- Uses boundary formula: I = u(b)exp(iΦ(b)) - u(a)exp(iΦ(a))
- Chebyshev collocation with 8 nodes per segment
- Robust for highly oscillatory integrands with variable frequency

```python
from oscillatory_integrals import levin_oscillatory_integral
I = levin_oscillatory_integral(f_func, phi_func, phi_prime_func, r_start, r_end)
```

#### Automatic Method Selection

The unified interface `compute_outer_integral_oscillatory()` automatically selects:
- **Filon** when |Φ''| × h² < 0.1 (constant frequency)
- **Levin** when |Φ''| × h² ≥ 0.1 (variable frequency)

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
| `generate_phase_nodes(r_start, r_end, k_total, Δφ)` | Constant phase grid |
| `_filon_segment_complex(f, r, ω, φ₀)` | Single segment Filon |
| `_levin_segment_complex(f, r, Φ, Φ')` | Single segment Levin |
| `compute_product_phases(...)` | sinA×sinB → cos terms |
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
| Phase unstable warnings | Normal for L near cutoff; results still usable |
| "r_m not in asymptotic region" | Fixed in v2.1 - relaxed threshold to 1% |
| Non-finite integral warnings | Check input wavefunctions; reduce L_max |

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and detailed changes.

## References

1. **Article**: Lai et al., J. At. Mol. Sci. 5, 311-323 (2014)
2. **SAE Potential**: X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)
3. **Ionization TDCS**: S. Jones, D.H. Madison et al., Phys. Rev. A 48, 2285 (1993)
4. **Inner-shell ionization**: D. Bote, F. Salvat, Phys. Rev. A 77, 042701 (2008)

---
*Happy computing!*
