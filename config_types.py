# config_types.py
"""
Configuration Dataclasses for DWBA Calculations
================================================

This module provides typed configuration containers that replace
scattered dict access patterns throughout the codebase.

Benefits:
- Type safety and IDE autocompletion
- Default values documented in one place
- Validation at construction time
- Easy serialization to/from dicts

Usage:
    grid_cfg = GridConfig.from_params(params)
    pilot_cfg = PilotConfig.from_params(params)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal, Any

from constants import (
    DEFAULT_R_MAX,
    DEFAULT_N_POINTS,
    SAFETY_FACTOR_R_MAX,
    N_POINTS_CAP,
    MIN_POINTS_PER_WAVELENGTH,
    PILOT_ENERGY_EV,
    PILOT_N_THETA,
    L_MAX_HARD_CAP,
    L_MAX_INTEGRALS_CAP,
    GPU_MEMORY_THRESHOLD,
    MAX_CPU_WORKERS,
)


@dataclass
class GridConfig:
    """Configuration for radial grid construction.
    
    Attributes
    ----------
    strategy : str
        Grid sizing strategy: "manual", "global", or "local".
    r_max : float
        Maximum radius in atomic units.
    n_points : int
        Number of radial grid points.
    r_max_scale_factor : float
        Safety factor for adaptive r_max calculation.
    n_points_max : int
        Memory cap for grid points.
    min_points_per_wavelength : int
        Minimum sampling density for oscillatory functions.
    """
    strategy: Literal["manual", "global", "local"] = "global"
    r_max: float = DEFAULT_R_MAX
    n_points: int = DEFAULT_N_POINTS
    r_max_scale_factor: float = SAFETY_FACTOR_R_MAX
    n_points_max: int = N_POINTS_CAP
    min_points_per_wavelength: int = MIN_POINTS_PER_WAVELENGTH
    
    @classmethod
    def from_params(cls, params: dict) -> GridConfig:
        """Create GridConfig from params dict."""
        grid = params.get('grid', {})
        return cls(
            strategy=grid.get('strategy', 'global'),
            r_max=grid.get('r_max', DEFAULT_R_MAX),
            n_points=grid.get('n_points', DEFAULT_N_POINTS),
            r_max_scale_factor=grid.get('r_max_scale_factor', SAFETY_FACTOR_R_MAX),
            n_points_max=grid.get('n_points_max', N_POINTS_CAP),
            min_points_per_wavelength=grid.get('min_points_per_wavelength', MIN_POINTS_PER_WAVELENGTH),
        )
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            'strategy': self.strategy,
            'r_max': self.r_max,
            'n_points': self.n_points,
            'r_max_scale_factor': self.r_max_scale_factor,
            'n_points_max': self.n_points_max,
            'min_points_per_wavelength': self.min_points_per_wavelength,
        }


@dataclass
class PilotConfig:
    """Configuration for pilot calibration calculation.
    
    Pilot calculations run at high energy (typically 1000 eV) to
    calibrate the Tong model α parameter.
    
    Attributes
    ----------
    energy_eV : float
        Pilot calculation energy.
    n_theta : int
        Angular grid points (can be reduced for speed).
    L_max_projectile : str or int
        "auto" for dynamic scaling, or explicit integer.
    L_max_integrals : str or int
        "auto" for dynamic scaling, or explicit integer.
    """
    energy_eV: float = PILOT_ENERGY_EV
    n_theta: int = PILOT_N_THETA
    L_max_projectile: str | int = "auto"
    L_max_integrals: str | int = "auto"
    
    @classmethod
    def from_params(cls, params: dict) -> PilotConfig:
        """Create PilotConfig from params dict."""
        exc = params.get('excitation', {})
        return cls(
            energy_eV=exc.get('pilot_energy_eV', PILOT_ENERGY_EV),
            n_theta=exc.get('pilot_n_theta', PILOT_N_THETA),
            L_max_projectile=exc.get('pilot_L_max_projectile', 'auto'),
            L_max_integrals=exc.get('pilot_L_max_integrals', 'auto'),
        )
    
    def calculate_dynamic_L_max(
        self, 
        k_pilot: float, 
        r_max: float,
        base_L_proj: int,
        base_L_int: int
    ) -> tuple[int, int]:
        """Calculate dynamic L_max values based on energy.
        
        Parameters
        ----------
        k_pilot : float
            Wave number at pilot energy (a.u.).
        r_max : float
            Grid maximum radius.
        base_L_proj : int
            Base L_max_projectile from channel spec.
        base_L_int : int
            Base L_max_integrals from channel spec.
            
        Returns
        -------
        tuple[int, int]
            (L_proj, L_int) values to use.
        """
        from constants import TURNING_POINT_SCALE
        
        # Dynamic estimate based on classical turning point
        L_proj_dynamic = int(k_pilot * r_max * TURNING_POINT_SCALE)
        
        # Determine L_proj
        if self.L_max_projectile == 'auto':
            L_proj = max(base_L_proj, min(L_proj_dynamic, L_MAX_HARD_CAP))
        else:
            L_proj = int(self.L_max_projectile)
        
        # Determine L_int
        if self.L_max_integrals == 'auto':
            L_int = max(base_L_int, min(L_MAX_INTEGRALS_CAP, L_proj // 4))
        else:
            L_int = int(self.L_max_integrals)
        
        return L_proj, L_int


@dataclass
class HardwareConfig:
    """Configuration for GPU/CPU hardware utilization.
    
    Attributes
    ----------
    gpu_block_size : str or int
        "auto" for dynamic tuning, or explicit block size.
    gpu_memory_mode : str
        "auto", "full", or "block" strategy.
    gpu_memory_threshold : float
        Maximum fraction of GPU memory to use.
    n_workers : str or int
        "auto", "max", or explicit worker count.
    """
    gpu_block_size: str | int = "auto"
    gpu_memory_mode: Literal["auto", "full", "block"] = "auto"
    gpu_memory_threshold: float = GPU_MEMORY_THRESHOLD
    n_workers: str | int = "auto"
    
    @classmethod
    def from_params(cls, params: dict) -> HardwareConfig:
        """Create HardwareConfig from params dict."""
        hw = params.get('hardware', {})
        return cls(
            gpu_block_size=hw.get('gpu_block_size', 'auto'),
            gpu_memory_mode=hw.get('gpu_memory_mode', 'auto'),
            gpu_memory_threshold=hw.get('gpu_memory_threshold', GPU_MEMORY_THRESHOLD),
            n_workers=hw.get('n_workers', 'auto'),
        )
    
    def get_worker_count(self) -> int:
        """Get actual worker count based on configuration."""
        import os
        cpu_count = os.cpu_count() or 4
        
        if self.n_workers in ("auto", 0, "0"):
            return min(cpu_count, MAX_CPU_WORKERS)
        elif self.n_workers == "max":
            return cpu_count
        else:
            try:
                val = int(self.n_workers)
                return min(val, cpu_count) if val > 0 else min(cpu_count, MAX_CPU_WORKERS)
            except (ValueError, TypeError):
                return min(cpu_count, MAX_CPU_WORKERS)


@dataclass
class OscillatoryConfig:
    """Configuration for oscillatory integral methods.
    
    Attributes
    ----------
    method : str
        Integration method: "legacy", "advanced", "full_split".
    CC_nodes : int
        Clenshaw-Curtis quadrature nodes per interval.
    phase_increment : float
        Phase step for interval subdivision (radians).
    min_grid_fraction : float
        Minimum match point fraction.
    k_threshold : float
        Wave number threshold for Filon integration.
    """
    method: Literal["legacy", "advanced", "full_split"] = "advanced"
    CC_nodes: int = 5
    phase_increment: float = 1.5708  # π/2
    min_grid_fraction: float = 0.1
    k_threshold: float = 0.5
    
    @classmethod
    def from_params(cls, params: dict) -> OscillatoryConfig:
        """Create OscillatoryConfig from params dict."""
        osc = params.get('oscillatory', {})
        return cls(
            method=osc.get('method', 'advanced'),
            CC_nodes=osc.get('CC_nodes', 5),
            phase_increment=osc.get('phase_increment', 1.5708),
            min_grid_fraction=osc.get('min_grid_fraction', 0.1),
            k_threshold=osc.get('k_threshold', 0.5),
        )


@dataclass
class CalculationContext:
    """Context object for a single DWBA calculation run.
    
    Encapsulates configuration and state that was previously stored
    in global variables. Pass this explicitly to functions instead
    of relying on global state.
    
    Attributes
    ----------
    grid : GridConfig
        Radial grid configuration.
    oscillatory : OscillatoryConfig
        Oscillatory integral settings.
    hardware : HardwareConfig
        GPU/CPU utilization settings.
    pilot : PilotConfig
        Pilot calibration settings.
    scan_logged : bool
        Flag to prevent repetitive logging during energy scans.
    solver : str
        ODE solver preference: "auto", "rk45", "johnson", "numerov".
    """
    grid: GridConfig = field(default_factory=GridConfig)
    oscillatory: OscillatoryConfig = field(default_factory=OscillatoryConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    scan_logged: bool = False
    solver: str = "rk45"
    
    @classmethod
    def from_params(cls, params: dict) -> CalculationContext:
        """Create CalculationContext from params dict."""
        return cls(
            grid=GridConfig.from_params(params),
            oscillatory=OscillatoryConfig.from_params(params),
            hardware=HardwareConfig.from_params(params),
            pilot=PilotConfig.from_params(params),
            solver=params.get('hardware', {}).get('solver', 'rk45'),
        )
    
    def reset_scan_logging(self) -> None:
        """Reset scan-level logging flags."""
        self.scan_logged = False
    
    def to_oscillatory_config_dict(self) -> dict:
        """Convert oscillatory settings to dict for backward compatibility."""
        return {
            "method": self.oscillatory.method,
            "CC_nodes": self.oscillatory.CC_nodes,
            "phase_increment": self.oscillatory.phase_increment,
            "min_grid_fraction": self.oscillatory.min_grid_fraction,
            "k_threshold": self.oscillatory.k_threshold,
            "gpu_block_size": self.hardware.gpu_block_size,
            "gpu_memory_mode": self.hardware.gpu_memory_mode,
            "gpu_memory_threshold": self.hardware.gpu_memory_threshold,
            "n_workers": self.hardware.n_workers,
            "solver": self.solver,
        }

