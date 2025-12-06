# Project Knowledge Base: electron-impact scattering in DWBA

## 1. Project Objective
Implementation of the **Distorted Wave Born Approximation (DWBA)** to calculate cross sections for:
- **Electron-Impact Excitation** (e + A -> e + A*)
- **Electron-Impact Ionization** (e + A -> e + A+ + e)

The goal is to reproduce the methodology and accuracy of the Bray et al. article ("Electron-impact ionization and excitation of hydrogen in a distorted-wave model"), creating a robust Python pipeline.

## 2. Theoretical Framework (Based on Article)

### 2.1 Physics Model
- **Single Active Electron (SAE) Approximation**: The target is modeled as a frozen core (charge $Z_c$) plus one active electron. For Hydrogen, $Z_c=1$.
- **Distorting Potential ($U$)**: The projectile electron moves in a "distorted" orbit determined by the **Static-Exchange** potential of the target.
  - Unlike simple models that use only the static potential, this project implements the full static-exchange interaction to ensure orthogonality of continuum and bound states.

### 2.2 T-Matrix Formulation
The Transition Matrix element for a transition $i \to f$ is calculated as:
$$ T_{fi} = \langle \chi_f^{(-)} \Phi_f | V - U_f | \mathcal{A} \{ \chi_i^{(+)} \Phi_i \} \rangle $$

Where:
- $\chi$: Distorted continuum waves.
- $\Phi$: Target bound states.
- $V$: Coulomb interaction ($1/r_{12}$).
- $\mathcal{A}$: Antisymmetrization operator (leads to Direct and Exchange terms).

The implementation splits this into:
1.  **Direct Amplitude**: $f(\theta) \sim \langle \chi_f(1) \Phi_f(2) | \frac{1}{|r_1-r_2|} | \chi_i(1) \Phi_i(2) \rangle$
2.  **Exchange Amplitude**: $g(\theta) \sim \langle \chi_f(2) \Phi_f(1) | \frac{1}{|r_1-r_2|} | \chi_i(1) \Phi_i(2) \rangle$

### 2.3 Approximations & Corrections
- **Exchange Potential**: Exact non-local exchange is expensive. We use the **Furness-McCarthy (FM)** local equivalent potential:
  $$ V_{ex}(r) = \frac{1}{2} \left[ (E - V_{static}) - \sqrt{(E - V_{static})^2 + 4\pi \rho(r)} \right] $$
- **BE-Scaling (M-Tong)**: To correct the DWBA's behavior near the threshold (where it often overestimates), we apply empirical scaling:
  $$ \sigma_{scaled}(E) = \frac{E}{E + I_p + E_{exc}} \sigma_{DWBA}(E) \approx \frac{E_{inc}}{E_{inc} + E_{exc}} \sigma_{DWBA} $$

## 3. Implementation Logic

### 3.1 Numerical Pipeline
1.  **Grid**: non-linear radial grid (dense near nucleus, sparse at large $r$) to capture core dynamics.
2.  **Bound States**: Solves radial Schrödinger equation using finite difference + sparse diagonalization (`eigsh`).
3.  **Distorting Potentials**:
    - **Static**: $V_{core} + V_{Hartree}$
    - **Exchange**: Furness-McCarthy (dependent on beam energy $k^2/2$).
4.  **Continuum Waves**: Integrates radial ODE outward using `solve_ivp` (LSODA/RK), matching asymptotic Coulomb/Bessel functions to extract phase shifts $\delta_l$ and normalize to unit amplitude.
5.  **Matrix Elements**:
    - **Direct Integrals**: Overlap of projectile densities $\rho_{proj} = \chi_f \chi_i$ and target densities $\rho_{targ} = u_f u_i$.
    - **Exchange Integrals**: Overlap of mixed pairs $\rho_1 = \chi_f u_i$ and $\rho_2 = \chi_i u_f$.
    - Integration uses vectorized broadcasting for speed.
6.  **Cross Sections**:
    - $d\sigma/d\Omega$ computed from partial wave sums of $f(\theta)$ and $g(\theta)$.
    - Integrated $\sigma_{total}$ obtained by angular integration.

### 3.2 File Architecture
- **`DW_main.py`**: Unified interface for running calculations and plotting.
- **`driver.py`**: Orchestrates excitation calculations (potentials -> waves -> integrals -> sigma).
- **`dwba_matrix_elements.py`**: Computes radial integrals $I_L$.
- **`dwba_coupling.py`**: Handles angular momentum algebra coefficients ($F_L, G_L$).
- **`distorting_potential.py`**: Implements $V_{static}$ and $V_{exchange}$ (FM).
- **`continuum.py`**: Solves scattering states.
- **`bound_states.py`**: Solves target structure.
- **`ionization.py`**: Implements ionization via projection onto continuum (EII).

## 4. Verification Status
- **Excitation H(1s->2s)**: Verified. Code runs stably and produces non-zero cross-sections ($2.76 \times 10^{-19}$ cm$^2$ at 15 eV).
- **Exchange Implementation**: Confirmed working. Furness-McCarthy potential is generated and affects the wavefunction.
- **Separation of Integrals**: Direct and Exchange radial integrals are computed separately, correcting previous physics error.

## 5. Units & Conventions
- **Internal**: Hartree Atomic Units ($e=m_e=\hbar=1$). Energy in Hartree, Length in Bohr ($a_0$).
- **Input/Output**: Energy in eV, Cross Sections in cm$^2$ (typically).
- **Conversions**:
  - 1 Ha = 27.211 eV
  - 1 $a_0$ = 0.529 $\mathring{A}$
  - $\pi a_0^2 \approx 8.797 \times 10^{-17}$ cm$^2$

## 6. Known Limitations
- **High-L Convergence**: The partial wave series sums only up to user-defined $L_{max}$. For high energies or dipole transitions, this may need a "Born Top-Up" correction (currently documented but not implemented).
- **Ionization**: Treated as excitation to positive energy states. This is a standard approximation but neglects some 3-body Coulomb boundary conditions (Peterkop).
