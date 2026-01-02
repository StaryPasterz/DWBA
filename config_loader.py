# config_loader.py
"""
Configuration File Loader for DWBA Calculation Suite
=====================================================

Provides YAML-based configuration file loading and validation for 
automated batch calculations without interactive prompts.

Usage
-----
```python
from config_loader import load_config, DWBAConfig

config = load_config("my_calculation.yaml")
```

Configuration Format
--------------------
See examples/config_excitation.yaml for a complete template.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any, Union
import json

from logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class TargetConfig:
    """Target atom configuration."""
    atom: str = "H"
    Z: Optional[float] = None  # Only needed for custom atoms
    
@dataclass
class StateConfig:
    """Quantum state configuration."""
    n: int = 1
    l: int = 0

@dataclass
class StatesConfig:
    """Initial and final state configuration."""
    initial: StateConfig = field(default_factory=lambda: StateConfig(n=1, l=0))
    final: StateConfig = field(default_factory=lambda: StateConfig(n=2, l=0))

@dataclass
class EnergyConfig:
    """Energy grid configuration."""
    type: Literal["single", "linear", "log", "list"] = "log"
    start_eV: float = 10.2
    end_eV: float = 1000.0
    step_eV: Optional[float] = None  # For linear grid
    density: float = 1.0  # For log grid
    values: Optional[List[float]] = None  # For list type

@dataclass
class GridConfig:
    """Radial grid configuration."""
    r_max: float = 200.0
    n_points: int = 3000
    r_max_scale_factor: float = 2.5
    n_points_max: int = 8000

@dataclass
class ExcitationConfig:
    """Excitation-specific parameters."""
    L_max_integrals: int = 15
    L_max_projectile: int = 5
    n_theta: int = 200
    pilot_energy_eV: float = 1000.0

@dataclass
class IonizationConfig:
    """Ionization-specific parameters."""
    l_eject_max: int = 3
    L_max: int = 15
    L_max_projectile: int = 50
    n_energy_steps: int = 10

@dataclass
class OscillatoryConfig:
    """Oscillatory integral configuration."""
    method: Literal["legacy", "advanced", "full_split"] = "advanced"
    CC_nodes: int = 5
    phase_increment: float = 1.5708
    min_grid_fraction: float = 0.1
    k_threshold: float = 0.5
    gpu_block_size: Union[int, str] = "auto"  # "auto" or explicit int
    gpu_memory_mode: Literal["auto", "full", "block"] = "auto"
    gpu_memory_threshold: float = 0.7

@dataclass
class OutputConfig:
    """Output configuration."""
    save_dcs: bool = True
    save_partial: bool = True
    calibrate: bool = True

@dataclass
class DWBAConfig:
    """
    Complete DWBA calculation configuration.
    
    This dataclass represents all parameters needed to run a DWBA 
    calculation without interactive prompts.
    """
    run_name: str = "batch_run"
    calculation_type: Literal["excitation", "ionization"] = "excitation"
    physics_model: Literal["static", "polarization"] = "static"
    
    target: TargetConfig = field(default_factory=TargetConfig)
    states: StatesConfig = field(default_factory=StatesConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    excitation: ExcitationConfig = field(default_factory=ExcitationConfig)
    ionization: IonizationConfig = field(default_factory=IonizationConfig)
    oscillatory: OscillatoryConfig = field(default_factory=OscillatoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# =============================================================================
# CONFIG LOADING AND VALIDATION
# =============================================================================

def _dict_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.
    Handles nested dataclasses and provides defaults for missing fields.
    """
    if data is None:
        return cls()
    
    # Get field info from the dataclass
    field_types = {}
    if hasattr(cls, '__dataclass_fields__'):
        field_types = {
            name: field.type 
            for name, field in cls.__dataclass_fields__.items()
        }
    
    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            
            # Check if field type is a dataclass
            origin_type = getattr(field_type, '__origin__', None)
            if origin_type is None and hasattr(field_type, '__dataclass_fields__'):
                # Nested dataclass
                kwargs[field_name] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def _parse_gpu_block_size(value) -> int:
    """Parse gpu_block_size from config: 'auto'/0 = auto-tune, int > 0 = explicit."""
    if isinstance(value, str):
        if value.lower() == "auto":
            return 0  # Internal representation for auto-tune
        try:
            return int(value)
        except ValueError:
            return 0
    return int(value) if value else 0


def load_config(path: Union[str, Path]) -> DWBAConfig:
    """
    Load and validate a YAML configuration file.
    
    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
        
    Returns
    -------
    DWBAConfig
        Validated configuration object.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    logger.info("Loading configuration from: %s", path)
    
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)
    
    if raw_data is None:
        raise ValueError(f"Configuration file is empty: {path}")
    
    # Parse top-level fields
    config = DWBAConfig()
    
    # Simple fields
    if 'run_name' in raw_data:
        config.run_name = raw_data['run_name']
    
    # Calculation type (handle nested 'calculation' key)
    if 'calculation' in raw_data:
        calc = raw_data['calculation']
        if isinstance(calc, dict) and 'type' in calc:
            config.calculation_type = calc['type']
        elif isinstance(calc, str):
            config.calculation_type = calc
    
    # Physics model
    if 'physics' in raw_data:
        phys = raw_data['physics']
        if isinstance(phys, dict) and 'model' in phys:
            config.physics_model = phys['model']
        elif isinstance(phys, str):
            config.physics_model = phys
    
    # Target
    if 'target' in raw_data:
        t = raw_data['target']
        config.target = TargetConfig(
            atom=t.get('atom', 'H'),
            Z=t.get('Z', None)
        )
    
    # States
    if 'states' in raw_data:
        s = raw_data['states']
        initial = s.get('initial', {})
        final = s.get('final', {})
        config.states = StatesConfig(
            initial=StateConfig(n=initial.get('n', 1), l=initial.get('l', 0)),
            final=StateConfig(n=final.get('n', 2), l=final.get('l', 0))
        )
    
    # Energy
    if 'energy' in raw_data:
        e = raw_data['energy']
        config.energy = EnergyConfig(
            type=e.get('type', 'log'),
            start_eV=e.get('start_eV', 10.2),
            end_eV=e.get('end_eV', 1000.0),
            step_eV=e.get('step_eV'),
            density=e.get('density', 1.0),
            values=e.get('values')
        )
    
    # Grid
    if 'grid' in raw_data:
        g = raw_data['grid']
        config.grid = GridConfig(
            r_max=g.get('r_max', 200.0),
            n_points=g.get('n_points', 3000),
            r_max_scale_factor=g.get('r_max_scale_factor', 2.5),
            n_points_max=g.get('n_points_max', 8000)
        )
    
    # Excitation
    if 'excitation' in raw_data:
        ex = raw_data['excitation']
        config.excitation = ExcitationConfig(
            L_max_integrals=ex.get('L_max_integrals', 15),
            L_max_projectile=ex.get('L_max_projectile', 5),
            n_theta=ex.get('n_theta', 200),
            pilot_energy_eV=ex.get('pilot_energy_eV', 1000.0)
        )
    
    # Ionization
    if 'ionization' in raw_data:
        ion = raw_data['ionization']
        config.ionization = IonizationConfig(
            l_eject_max=ion.get('l_eject_max', 3),
            L_max=ion.get('L_max', 15),
            L_max_projectile=ion.get('L_max_projectile', 50),
            n_energy_steps=ion.get('n_energy_steps', 10)
        )
    
    # Oscillatory
    if 'oscillatory' in raw_data:
        osc = raw_data['oscillatory']
        config.oscillatory = OscillatoryConfig(
            method=osc.get('method', 'advanced'),
            CC_nodes=osc.get('CC_nodes', 5),
            phase_increment=osc.get('phase_increment', 1.5708),
            min_grid_fraction=osc.get('min_grid_fraction', 0.1),
            k_threshold=osc.get('k_threshold', 0.5),
            gpu_block_size=_parse_gpu_block_size(osc.get('gpu_block_size', 'auto')),
            gpu_memory_mode=osc.get('gpu_memory_mode', 'auto'),
            gpu_memory_threshold=osc.get('gpu_memory_threshold', 0.7)
        )
    
    # Output
    if 'output' in raw_data:
        out = raw_data['output']
        config.output = OutputConfig(
            save_dcs=out.get('save_dcs', True),
            save_partial=out.get('save_partial', True),
            calibrate=out.get('calibrate', True)
        )
    
    # Validate
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    logger.info("Configuration loaded successfully: run_name='%s', type='%s'", 
                config.run_name, config.calculation_type)
    
    return config


def validate_config(config: DWBAConfig) -> List[str]:
    """
    Validate a DWBAConfig and return a list of errors.
    
    Returns
    -------
    List[str]
        List of validation error messages. Empty if valid.
    """
    errors = []
    
    # Calculation type
    if config.calculation_type not in ("excitation", "ionization"):
        errors.append(f"Invalid calculation_type: '{config.calculation_type}'. Must be 'excitation' or 'ionization'.")
    
    # Physics model
    if config.physics_model not in ("static", "polarization"):
        errors.append(f"Invalid physics_model: '{config.physics_model}'. Must be 'static' or 'polarization'.")
    
    # States
    ni, li = config.states.initial.n, config.states.initial.l
    nf, lf = config.states.final.n, config.states.final.l
    
    if ni < 1:
        errors.append(f"Initial state n must be >= 1, got {ni}")
    if nf < 1:
        errors.append(f"Final state n must be >= 1, got {nf}")
    if li < 0 or li >= ni:
        errors.append(f"Initial state l must satisfy 0 <= l < n, got l={li}, n={ni}")
    if lf < 0 or lf >= nf:
        errors.append(f"Final state l must satisfy 0 <= l < n, got l={lf}, n={nf}")
    
    # Energy
    if config.energy.type == "list" and not config.energy.values:
        errors.append("Energy type 'list' requires 'values' to be specified")
    if config.energy.type in ("linear", "log"):
        if config.energy.start_eV >= config.energy.end_eV:
            errors.append(f"Energy start ({config.energy.start_eV}) must be < end ({config.energy.end_eV})")
    
    # Grid
    if config.grid.n_points < 100:
        errors.append(f"Grid n_points must be >= 100, got {config.grid.n_points}")
    if config.grid.r_max < 10:
        errors.append(f"Grid r_max must be >= 10, got {config.grid.r_max}")
    
    # Oscillatory
    if config.oscillatory.method not in ("legacy", "advanced", "full_split"):
        errors.append(f"Invalid oscillatory method: '{config.oscillatory.method}'")
    if config.oscillatory.gpu_memory_mode not in ("auto", "full", "block"):
        errors.append(f"Invalid gpu_memory_mode: '{config.oscillatory.gpu_memory_mode}'")
    
    return errors


def config_to_params_dict(config: DWBAConfig) -> Dict[str, Dict[str, Any]]:
    """
    Convert DWBAConfig to the params dictionary format used by DW_main.py.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary compatible with prompt_use_defaults() return format.
    """
    return {
        'grid': {
            'r_max': config.grid.r_max,
            'n_points': config.grid.n_points,
            'r_max_scale_factor': config.grid.r_max_scale_factor,
            'n_points_max': config.grid.n_points_max,
        },
        'excitation': {
            'L_max_integrals': config.excitation.L_max_integrals,
            'L_max_projectile': config.excitation.L_max_projectile,
            'n_theta': config.excitation.n_theta,
            'pilot_energy_eV': config.excitation.pilot_energy_eV,
        },
        'ionization': {
            'l_eject_max': config.ionization.l_eject_max,
            'L_max': config.ionization.L_max,
            'L_max_projectile': config.ionization.L_max_projectile,
            'n_energy_steps': config.ionization.n_energy_steps,
        },
        'oscillatory': {
            'method': config.oscillatory.method,
            'CC_nodes': config.oscillatory.CC_nodes,
            'phase_increment': config.oscillatory.phase_increment,
            'min_grid_fraction': config.oscillatory.min_grid_fraction,
            'k_threshold': config.oscillatory.k_threshold,
            'gpu_block_size': config.oscillatory.gpu_block_size,
            'gpu_memory_mode': config.oscillatory.gpu_memory_mode,
            'gpu_memory_threshold': config.oscillatory.gpu_memory_threshold,
        },
    }


def generate_template_config(output_path: Union[str, Path], 
                             calc_type: str = "excitation") -> None:
    """
    Generate a template configuration file.
    
    Parameters
    ----------
    output_path : str or Path
        Where to save the template.
    calc_type : str
        "excitation" or "ionization"
    """
    template = f'''# DWBA Calculation Configuration
# Generated template for {calc_type} calculations

run_name: "{calc_type}_batch"

calculation:
  type: "{calc_type}"

target:
  atom: "H"           # Atom from atoms.json: H, He, Li, Ne, Ar, etc.
  # Z: 1.0            # Only needed for custom atoms

states:
  initial:
    n: 1
    l: 0
  final:
    n: 2
    l: 0              # 0 for s-wave, 1 for p-wave

physics:
  model: "static"     # "static" or "polarization"

energy:
  type: "log"         # "single", "linear", "log", or "list"
  start_eV: 10.2      # First energy point (eV above threshold)
  end_eV: 1000.0      # Last energy point
  density: 1.0        # Log grid density (higher = more points)
  # step_eV: 10.0     # For linear grid only
  # values: [10, 20, 50, 100, 200, 500, 1000]  # For list type only

grid:
  r_max: 200
  n_points: 3000
  r_max_scale_factor: 2.5
  n_points_max: 8000

{calc_type}:
  L_max_integrals: 15
  L_max_projectile: 5
  n_theta: 200
  pilot_energy_eV: 1000

oscillatory:
  method: "advanced"          # "legacy", "advanced", "full_split"
  CC_nodes: 5
  phase_increment: 1.5708     # π/2
  min_grid_fraction: 0.1
  k_threshold: 0.5
  gpu_block_size: \"auto\"        # \"auto\" = auto-tune, int = explicit size
  gpu_memory_mode: "auto"     # "auto", "full", "block"
  gpu_memory_threshold: 0.7

output:
  save_dcs: true
  save_partial: true
  calibrate: true
'''
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    logger.info("Template configuration saved to: %s", path)


# =============================================================================
# CLI UTILITIES
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DWBA Configuration File Utilities")
    parser.add_argument("--generate", "-g", type=str, metavar="PATH",
                        help="Generate template config at PATH")
    parser.add_argument("--type", "-t", choices=["excitation", "ionization"],
                        default="excitation", help="Template type")
    parser.add_argument("--validate", "-v", type=str, metavar="PATH",
                        help="Validate config file at PATH")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_template_config(args.generate, args.type)
        print(f"Template saved to: {args.generate}")
    elif args.validate:
        try:
            config = load_config(args.validate)
            print(f"✓ Configuration is valid: {args.validate}")
            print(f"  Run name: {config.run_name}")
            print(f"  Type: {config.calculation_type}")
            print(f"  Target: {config.target.atom}")
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    else:
        parser.print_help()
