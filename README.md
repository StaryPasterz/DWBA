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
- **Near-threshold**: Increase `n_points` and `r_max`
- **keV energies**: Ensure sufficient `L_max_projectile`
- **Auto-L cap**: If logs show L_max_proj hit cap=100, raise `L_max_projectile`
- **GPU**: Install CuPy for 5-10× speedup on radial integrals
- **Parallel**: Code auto-detects CPU cores for multiprocessing

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cross sections too small | Check (2π)⁴ factor in excitation and ionization |
| Near-threshold zeros | Use log grid, ensure energy > threshold + 0.5 eV |
| Slow runs | Reduce `n_points`, enable GPU |
| Fit gives different results | Reference params are protected; fitted params may vary |

## References

1. **Article**: Lai et al., J. At. Mol. Sci. 5, 311-323 (2014)
2. **SAE Potential**: X.M. Tong, C.D. Lin, J. Phys. B 38, 2593 (2005)
3. **Ionization TDCS**: S. Jones, D.H. Madison et al., Phys. Rev. A 48, 2285 (1993)
4. **Inner-shell ionization**: D. Bote, F. Salvat, Phys. Rev. A 77, 042701 (2008)

---
*Happy computing!*
