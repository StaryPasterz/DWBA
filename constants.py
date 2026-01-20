# constants.py
"""
Physical and Numerical Constants for DWBA Calculations
=======================================================

This module centralizes all magic numbers used throughout the codebase,
providing named constants with documentation for maintainability.

Physics Constants
-----------------
- HARTREE_TO_EV: Conversion factor between atomic units and electronvolts
- EV_TO_HARTREE: Inverse conversion

Numerical Limits
----------------
- L_MAX_HARD_CAP: Maximum angular momentum to prevent runaway computation
- TURNING_POINT_SCALE: Conservative factor for classical turning point estimates

Grid Defaults
-------------
- Default values for radial grid construction
"""

from __future__ import annotations

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Energy conversion factors (CODATA values)
HARTREE_TO_EV: float = 27.211386245988
"""Hartree to eV conversion: 1 Ha = 27.211... eV"""

EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
"""eV to Hartree conversion: 1 eV = 0.0367... Ha"""

# =============================================================================
# NUMERICAL LIMITS
# =============================================================================

L_MAX_HARD_CAP: int = 150
"""Maximum angular momentum quantum number.

Computational ceiling to prevent runaway partial wave summation.
Based on practical limits for numerical stability and memory usage.
"""

L_MAX_INTEGRALS_CAP: int = 25
"""Maximum L for multipole expansion of 1/r12.

Higher values rarely contribute significantly but increase computation.
"""

TURNING_POINT_SCALE: float = 0.6
"""Conservative scaling factor for classical turning point estimate.

Used in: L_max ~ k * r_max * TURNING_POINT_SCALE
Value 0.6 provides margin for asymptotic fitting.
"""

SAFETY_FACTOR_R_MAX: float = 2.5
"""Safety factor for r_max beyond classical turning point.

Ensures grid extends well past r_t(L_max) for accurate phase fitting.
"""

# =============================================================================
# GRID DEFAULTS
# =============================================================================

DEFAULT_R_MIN: float = 1e-5
"""Default minimum radius in atomic units.

Small but non-zero to avoid division by zero in centrifugal term.
"""

DEFAULT_R_MAX: float = 200.0
"""Default maximum radius in atomic units.

Sufficient for most low-excited states. Adaptive scaling may extend this.
"""

DEFAULT_N_POINTS: int = 1000
"""Default number of radial grid points.

Baseline value; adaptive scaling adjusts based on energy.
"""

N_POINTS_CAP: int = 15000
"""Maximum grid points (memory cap).

Prevents excessive memory usage at high energies.
"""

MIN_POINTS_PER_WAVELENGTH: int = 15
"""Minimum grid points per de Broglie wavelength.

Ensures adequate sampling of oscillatory wavefunctions.
"""

# =============================================================================
# PILOT CALIBRATION DEFAULTS
# =============================================================================

PILOT_ENERGY_EV: float = 1000.0
"""Default pilot calibration energy in eV.

High energy chosen for Born-like behavior where Tong model is most accurate.
"""

PILOT_N_THETA: int = 50
"""Number of angular points for pilot DCS.

Reduced from production (200) for speed; sufficient for TCS integration.
"""

# =============================================================================
# CONVERGENCE THRESHOLDS
# =============================================================================

K_HIGH_ENERGY_THRESHOLD: float = 5.0
"""Wave number threshold for high-energy regime (a.u.).

Corresponds to ~340 eV. Above this, DWBA accuracy decreases.
"""

MONOTONIC_DECAY_TOLERANCE: float = 0.2
"""Maximum allowed fluctuation in decay rate for top-up.

Larger values = more lenient monotonicity check.
"""

TOP_UP_FRACTION_LIMIT: float = 0.20
"""Maximum top-up contribution as fraction of base sigma.

Safety limit: top-up should not dominate the result.
"""

# =============================================================================
# GPU/PARALLEL DEFAULTS
# =============================================================================

GPU_MEMORY_THRESHOLD: float = 0.8
"""Maximum fraction of GPU memory to use.

Conservative value to leave headroom for other processes.
"""

MAX_CPU_WORKERS: int = 8
"""Maximum CPU workers for parallel processing.

Caps parallelism to avoid diminishing returns and memory contention.
"""
"""
Compatibility note: These constants replace magic numbers scattered 
throughout the codebase. See CHANGELOG v2.17 for migration details.
"""
