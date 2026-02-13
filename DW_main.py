"""
DW_main.py - DWBA Calculation Suite 
=======================================

Unified interactive interface for Distorted Wave Born Approximation (DWBA)
calculations of electron-atom collision cross sections.

Main Features
-------------
1. Excitation Cross Sections - Calculate e + A → e + A* transitions
2. Ionization Cross Sections - Calculate e + A → e + A⁺ + e' transitions  
3. Visualization - Generate plots from saved result files
4. Partial Wave Analysis - Detailed convergence diagnostics
5. Potential Fitting - Tool for calibrating atomic potentials

Output
------
All results and plots are saved to the `results/` directory:
    results/results_[RUN_NAME]_exc.json - Excitation cross sections
    results/results_[RUN_NAME]_ion.json - Ionization cross sections
    results/plot_*.png - Generated visualization plots

The results/ directory is created automatically if it doesn't exist.

Usage
-----
    python DW_main.py

Environment Variables
--------------------
    DWBA_LOG_LEVEL : Set to DEBUG for verbose output

Dependencies
-----------
This module integrates: driver, ionization, calibration, plotter, atom_library
"""

import numpy as np
import json
import os
import sys
import glob
import argparse
import builtins
from dataclasses import asdict

from driver import (
    compute_total_excitation_cs,
    ExcitationChannelSpec,
    ev_to_au,
    set_oscillatory_method,
    set_oscillatory_config,
    reset_scan_logging
)
from grid import (
    k_from_E_eV,
    compute_safe_L_max,
    compute_required_r_max,
    validate_high_energy,
)

from ionization import (
    compute_ionization_cs,
    IonizationChannelSpec
)
from potential_core import CorePotentialParams
import atom_library
# import plotter  <-- Moved to local functions to avoid multiprocessing overhead
from calibration import TongModel
from output_utils import get_results_dir, get_output_path, get_json_path, find_result_files
from logging_config import get_logger

# -----------------------------------------------------------------------------
# Console output safety (Windows code pages)
# -----------------------------------------------------------------------------
# Some terminals (e.g. cp1250/cp1252) cannot encode box-drawing symbols used by
# this CLI, which may raise UnicodeEncodeError and abort batch runs.
#
# Strategy:
# 1) Try UTF-8 reconfigure when available.
# 2) Wrap module-local print() with graceful ASCII fallback on encoding errors.
# -----------------------------------------------------------------------------
if os.name == "nt":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _safe_print(*args, **kwargs) -> None:
    """Print with fallback for non-UTF console encodings."""
    try:
        builtins.print(*args, **kwargs)
        return
    except UnicodeEncodeError:
        pass

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    def _normalize(s: str) -> str:
        # Replace decorative glyphs with ASCII-safe equivalents.
        s = (
            s.replace("→", "->")
            .replace("✓", "[OK]")
            .replace("✗", "[X]")
            .replace("⚠", "[!]")
            .replace("•", "-")
            .replace("╔", "+").replace("╗", "+").replace("╚", "+").replace("╝", "+")
            .replace("═", "=").replace("─", "-")
            .replace("│", "|")
            .replace("┌", "+").replace("┐", "+").replace("└", "+").replace("┘", "+")
        )
        return s.encode(encoding, errors="replace").decode(encoding, errors="replace")

    safe_args = [_normalize(str(a)) for a in args]
    builtins.print(*safe_args, **kwargs)


# Module-local override: all print() calls in this file use safe output.
print = _safe_print

# Initialize logger for debug messages
logger = get_logger(__name__)

# =============================================================================
# CENTRALIZED DEFAULT PARAMETERS
# =============================================================================
# All numerical defaults organized by category. These are displayed to users
# and can be modified before calculation.

DEFAULTS = {
    # --- Grid Parameters ---
    "grid": {
        "strategy": "global",     # "manual" / "global" / "local" - v2.6+
        "r_max": 200.0,           # Maximum radius (a.u.)
        "n_points": 1000,         # Number of radial grid points
        "r_max_scale_factor": 2.5, # Safety factor for adaptive r_max
        "n_points_max": 15000,    # Maximum grid points (memory cap)
        "min_points_per_wavelength": 15,  # Minimum points per λ for high-E (v2.7+)
    },
    
    # --- Excitation-specific ---
    "excitation": {
        "L_max_integrals": 15,    # Maximum multipole L for Coulomb expansion
        "L_max_projectile": 5,    # Base partial wave L_max for projectile
        "n_theta": 200,           # Angular grid for DCS
        "pilot_energy_eV": 1000.0, # Calibration energy
        # Pilot light mode (v2.5+) - "auto" = dynamic scaling, int = explicit value
        "pilot_L_max_integrals": "auto",    # "auto" or int (e.g., 8)
        "pilot_L_max_projectile": "auto",   # "auto" or int (e.g., 30)
        "pilot_n_theta": 50,                # TCS only, DCS not needed
    },
    
    # --- Ionization-specific ---
    "ionization": {
        "l_eject_max": 3,         # Maximum ejected electron l
        "L_max": 15,              # Multipole L_max
        "L_max_projectile": 50,   # Base partial wave L_max
        "n_energy_steps": 10,     # SDCS integration steps
        "energy_quadrature": "gauss_legendre",  # "gauss_legendre" / "trapz_linear"
    },
    
    # --- Oscillatory Integrals ---
    "oscillatory": {
        "method": "advanced",         # "legacy" / "advanced" / "full_split"
        "CC_nodes": 5,                # Clenshaw-Curtis nodes per interval
        "phase_increment": 1.5708,    # π/2 radians per sub-interval
        "min_grid_fraction": 0.10,    # Minimum match point fraction
        "k_threshold": 0.5,           # k_total threshold for Filon
        "max_chi_cached": 20,         # v2.5+: LRU cache size for GPU continuum waves
        "phase_extraction": "hybrid", # v2.11+: "hybrid", "logderiv", "lsq"
        "solver": "rk45",             # v2.13+: "auto", "rk45" (recommended), "johnson", "numerov"
        "analytic_bypass": True,      # v2.34+: enable early analytic bypass in continuum solver
    },
    
    # --- Hardware (GPU/CPU) ---
    "hardware": {
        "gpu_block_size": "auto",     # "auto" = auto-tune based on VRAM, int = explicit size
        "gpu_memory_mode": "auto",    # "auto" / "full" / "block" - GPU matrix strategy
        "gpu_memory_threshold": 0.8,  # Max fraction of GPU memory to use (for auto mode)
        "n_workers": "auto",          # "auto" (balanced), "max" (all cores), or int count
    },
    
    # --- Output Options ---
    "output": {
        "save_dcs": True,             # Save differential cross section data
        "save_partial": True,         # Save partial wave contributions
        "calibrate": True,            # Apply Tong empirical calibration
    },
    
    # --- Energy Grid ---
    "energy_grid": {
        "start_epsilon_eV": 0.5,  # Start above threshold (eV)
        "log_density_factor": 1.0, # Log grid density multiplier
        "points_per_decade": 10,  # Base points per log decade
    },
}


def display_defaults(category: str = None) -> None:
    """Display default parameters, optionally filtered by category."""
    print()
    if category and category in DEFAULTS:
        categories = {category: DEFAULTS[category]}
    else:
        categories = DEFAULTS
    
    for cat_name, params in categories.items():
        print(f"  ┌─ {cat_name.upper()} ─────────────────────────────")
        for key, val in params.items():
            # Special display for gpu_block_size: 0 means auto
            if key == "gpu_block_size" and val == 0:
                print(f"  │  {key:<22} = auto")
            elif isinstance(val, float):
                print(f"  │  {key:<22} = {val:.4g}")
            else:
                print(f"  │  {key:<22} = {val}")
        print(f"  └{'─' * 40}")


def get_defaults_copy() -> dict:
    """Return a deep copy of defaults for modification."""
    import copy
    return copy.deepcopy(DEFAULTS)


def prompt_use_defaults(categories: list = None) -> dict:
    """
    Ask user if they want to use defaults. Only displays parameters if requested or changed.
    Returns the (possibly modified) defaults dict.
    """
    print_subheader("Numerical Parameters")
    
    # Compact prompt first
    use_defaults_raw = input("  Use default parameters? [Y/n, d=show details]: ").strip().lower()
    
    if use_defaults_raw == 'd':
        if categories:
            for cat in categories:
                if cat in DEFAULTS: display_defaults(cat)
        else:
            display_defaults()
        use_defaults_raw = input("\n  Use these defaults? [Y/n]: ").strip().lower()
    
    params = get_defaults_copy()
    
    if use_defaults_raw == 'n':
        print("\n  Edit parameters (press Enter to keep default):")
        changed = False
        
        if categories:
            cats_to_edit = [c for c in categories if c in DEFAULTS]
        else:
            cats_to_edit = list(DEFAULTS.keys())
        
        for cat_name in cats_to_edit:
            print(f"\n  ── {cat_name.upper()} ──")
            for key, default_val in DEFAULTS[cat_name].items():
                old_val = params[cat_name][key]
                if isinstance(default_val, bool):
                    # Special handling for boolean values
                    current = "Y" if default_val else "N"
                    val = input(f"    {key} [{current}]: ").strip().lower()
                    if val in ('y', 'yes', 'true', '1'):
                        params[cat_name][key] = True
                    elif val in ('n', 'no', 'false', '0'):
                        params[cat_name][key] = False
                    else:
                        params[cat_name][key] = default_val
                elif isinstance(default_val, int):
                    params[cat_name][key] = get_input_int(f"    {key}", default_val)
                elif isinstance(default_val, float):
                    params[cat_name][key] = get_input_float(f"    {key}", default_val)
                elif isinstance(default_val, str):
                    if key == "strategy" and cat_name == "grid":
                        print(f"\n    Grid Strategy:")
                        print("      1. global - Adaptive based on min energy (recommended)")
                        print("      2. local  - Adaptive per energy point (slower, accurate)")
                        print("      3. manual - Fixed r_max/n_points")
                        choice = input(f"    Select [1-3, default={default_val}]: ").strip()
                        if choice == '1': params[cat_name][key] = "global"
                        elif choice == '2': params[cat_name][key] = "local"
                        elif choice == '3': params[cat_name][key] = "manual"
                        else: params[cat_name][key] = default_val
                    elif key == "method" and cat_name == "oscillatory":
                        print(f"\n    Oscillatory method:")
                        print("      1. legacy     - Clenshaw-Curtis (fastest)")
                        print("      2. advanced   - CC + Levin/Filon tail (balanced)")
                        print("      3. full_split - Full I_in/I_out separation (most accurate)")
                        choice = input(f"    Select [1-3, default={default_val}]: ").strip()
                        if choice == '1': params[cat_name][key] = "legacy"
                        elif choice == '2': params[cat_name][key] = "advanced"
                        elif choice == '3': params[cat_name][key] = "full_split"
                        else: params[cat_name][key] = default_val
                    elif key == "gpu_memory_mode" and cat_name == "hardware":
                        print(f"\n    GPU Memory Strategy:")
                        print("      1. auto  - Check memory, fast if possible (recommended)")
                        print("      2. full  - Force full matrix (fastest, may OOM)")
                        print("      3. block - Force block-wise (slowest, safest)")
                        choice = input(f"    Select [1-3, default={default_val}]: ").strip()
                        if choice == '1': params[cat_name][key] = "auto"
                        elif choice == '2': params[cat_name][key] = "full"
                        elif choice == '3': params[cat_name][key] = "block"
                        else: params[cat_name][key] = default_val
                    else:
                        val = input(f"    {key} [{default_val}]: ").strip()
                        params[cat_name][key] = val if val else default_val
                
                if params[cat_name][key] != old_val:
                    changed = True
        
        if changed:
            print_success("Parameters updated")
            # Show summary of updated parameters only
            print("\n  Updated Parameters Summary:")
            if categories:
                for cat in categories:
                    if cat in params:
                        # Display only Categories that were requested
                        display_params_custom(cat, params[cat])
                        # Log the configuration change
                        logger.info("%s configuration: %s", cat.upper(), params[cat])
            else:
                for cat_name, val_dict in params.items():
                    display_params_custom(cat_name, val_dict)
                    logger.info("%s configuration: %s", cat_name.upper(), val_dict)
        else:
            print_info("No changes made. Using defaults.")
    else:
        # User chose 'y' or entered nothing
        # Log that defaults are being used (only once, concisely)
        if categories:
            for cat in categories:
                if cat in params:
                    logger.debug("%s configuration (default): %s", cat.upper(), params[cat])
        pass
    
    return params

def display_params_custom(cat_name, val_dict) -> None:
    """Helper to display a specific set of parameters."""
    print(f"  ┌─ {cat_name.upper()} ─────────────────────────────")
    for key, val in val_dict.items():
        if isinstance(val, float):
            print(f"  │  {key:<22} = {val:.4g}")
        else:
            print(f"  │  {key:<22} = {val}")
    print(f"  └{'─' * 40}")


def log_active_configuration(params: dict, context: str = "calculation") -> None:
    """
    Log the complete active configuration being used for calculation.
    
    This provides users visibility into exactly what parameters are being used,
    especially important when parameters are auto-calculated or derive from defaults.
    
    Parameters
    ----------
    params : dict
        The complete parameters dictionary.
    context : str
        Context description (e.g., "excitation scan", "ionization").
    """
    logger.info("=" * 60)
    logger.info("ACTIVE CONFIGURATION for %s", context)
    logger.info("=" * 60)
    
    # Grid configuration (always important)
    if 'grid' in params:
        g = params['grid']
        strategy = g.get('strategy', 'global').upper()
        logger.info("Grid: strategy=%s, r_max=%.1f a.u., n_points=%d", 
                   strategy, g.get('r_max', 200), g.get('n_points', 1000))
    
    # Excitation-specific
    if 'excitation' in params:
        e = params['excitation']
        logger.info("Excitation: L_max_int=%s, L_max_proj=%d, n_theta=%d",
                   e.get('L_max_integrals', 15), e.get('L_max_projectile', 5), 
                   e.get('n_theta', 200))
    
    # Ionization-specific
    if 'ionization' in params:
        ion = params['ionization']
        logger.info("Ionization: l_eject_max=%d, L_max=%d, n_energy_steps=%d, quadrature=%s",
                   ion.get('l_eject_max', 3), ion.get('L_max', 15),
                   ion.get('n_energy_steps', 10), ion.get('energy_quadrature', 'gauss_legendre'))
    
    # Oscillatory configuration
    if 'oscillatory' in params:
        o = params['oscillatory']
        logger.info("Oscillatory: method=%s, CC_nodes=%d, gpu_mode=%s, workers=%s",
                   o.get('method', 'advanced'), o.get('CC_nodes', 5),
                   o.get('gpu_memory_mode', 'auto'), o.get('n_workers', 'auto'))
    
    logger.info("=" * 60)


# =============================================================================
# UI Formatting Helpers
# =============================================================================

def print_header(title: str, width: int = 50) -> None:
    """Print a prominent section header."""
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")

def print_subheader(title: str, width: int = 50) -> None:
    """Print a subsection header."""
    print()
    padding = width - len(title) - 4
    print("── " + title + " " + "─" * max(padding, 3))

def print_menu(options: list, prompt: str = "Select", default: str = None) -> str:
    """Print a numbered menu and get selection."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    if default:
        raw = input(f"\n{prompt} [default={default}]: ").strip()
        if not raw:
            return default
    else:
        raw = input(f"\n{prompt}: ").strip()
    return raw

def print_progress(msg: str, end: str = ""):
    """Print a progress message."""
    print(f"  → {msg}", end=end, flush=True)

def print_success(msg: str):
    """Print a success message."""
    print(f"  ✓ {msg}")

def print_error(msg: str):
    """Print an error message."""
    print(f"  ✗ {msg}")

def print_warning(msg: str):
    """Print a warning message."""
    print(f"  ⚠ {msg}")

def print_info(msg: str):
    """Print an info message."""
    print(f"  • {msg}")

def print_result(energy: float, sigma: float, extra: str = ""):
    """Print a calculation result in compact format."""
    extra_str = f" {extra}" if extra else ""
    print(f"  {energy:8.2f} eV │ σ = {sigma:.3e} cm²{extra_str}")


# --- Input Helpers ---

def get_input_float(prompt, default=None) -> float:
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    try:
        return float(val)
    except ValueError:
        print("Invalid number.")
        return get_input_float(prompt, default)

def get_input_int(prompt, default=None) -> int:
    if default is not None:
        val = input(f"{prompt} [{default}]: ")
        if not val.strip(): return default
    else:
        val = input(f"{prompt}: ")
    try:
        return int(val)
    except ValueError:
        print("Invalid integer.")
        return get_input_int(prompt, default)

def select_target() -> atom_library.AtomEntry:
    print_subheader("Target Selection")
    atoms = atom_library.get_atom_list()
    
    # Display available atoms
    for i, name in enumerate(atoms):
        entry = atom_library.get_atom(name)
        print(f"  {i+1}. {entry.name:<5} │ {entry.description}")
    
    print(f"  {len(atoms)+1}. Custom │ Manual Z input")
    
    raw = input(f"\n  Select [1-{len(atoms)+1}, default=1]: ").strip()
    if not raw: raw = "1"
    
    try:
        idx = int(raw) - 1
    except ValueError:
        idx = 0
        
    if 0 <= idx < len(atoms):
        name = atoms[idx]
        entry = atom_library.get_atom(name)
        print_info(f"Selected: {entry.name} (Z={entry.Z})")
        return entry
    else:
        z_in = get_input_float("  Nuclear Charge Z")
        return atom_library.AtomEntry(
            name=f"Z={z_in}",
            Z=z_in,
            core_params=CorePotentialParams(Zc=z_in, a1=0, a2=1, a3=0, a4=1, a5=0, a6=1),
            default_n=1,
            default_l=0,
            ip_ev=13.6*z_in**2,
            description="Custom Hydrogen-like"
        )

def generate_flexible_energy_grid(start_eV: float, end_eV: float, density_factor: float = 1.0) -> np.ndarray:
    """
    Generates a grid that is dense at low energies (start) and sparse at high energies (end).
    Logic: Uses geometric progression for 'excess energy' (E - start).
    """
    epsilon_min = 0.5 # 500 meV above threshold (Relaxed from 0.05)
    if end_eV <= start_eV + epsilon_min:
        return np.array([start_eV + epsilon_min])
        
    epsilon_max = end_eV - start_eV
    
    # Estimate decent number of points
    # Log range is np.log10(epsilon_max / epsilon_min)
    # E.g. 1000/0.5 = 2000 -> 3.3 decades.
    # Base density = 10 points/decade (Relaxed from 15).
    
    decades = np.log10(epsilon_max / epsilon_min)
    n_points = int(decades * 10.0 * density_factor)
    n_points = max(n_points, 10) # Minimum safety
    
    # Generate excess
    excess = np.geomspace(epsilon_min, epsilon_max, n_points)
    
    energies = start_eV + excess
    
    # Ensure they are unique and sorted (geomspace should be, but safety first)
    return np.unique(np.round(energies, 3))


def build_energy_grid_above_threshold(
    energy_grid,
    threshold_eV: float,
    energy_type: str | None = None,
    log_end_eV: float | None = None,
    log_density: float = 1.0,
    threshold_margin_eV: float = 0.5,
) -> tuple[list[float], bool]:
    """
    Ensure batch energy grid is physically above threshold.

    For log grids, if user-provided start includes sub-threshold points, regenerate
    the log grid from (threshold + margin) to preserve intended log-point density,
    matching interactive behavior.
    """
    arr = np.asarray(energy_grid, dtype=float)
    regenerated = False

    if energy_type == "log" and log_end_eV is not None and np.any(arr <= threshold_eV):
        new_start = threshold_eV + threshold_margin_eV
        arr = generate_flexible_energy_grid(new_start, log_end_eV, log_density)
        regenerated = True

    filtered = [float(E) for E in arr if E > threshold_eV]
    return filtered, regenerated


def get_energy_list_interactive() -> np.ndarray | tuple:
    print_subheader("Energy Grid")
    print("  1. Single Energy")
    print("  2. Linear Grid (Start, End, Step)")
    print("  3. Custom List (comma separated)")
    print("  4. Log Grid (Dense at threshold)")
    
    choice = input("\n  Select [default=2]: ").strip()
    if not choice: choice = '2'
    
    if choice == '1':
        E = get_input_float("  Energy [eV]", 50.0)
        print_info(f"Single energy: {E} eV")
        return np.array([E])
        
    elif choice == '2':
        start = get_input_float("  Start [eV]", 10.0)
        end   = get_input_float("  End   [eV]", 200.0)
        step  = get_input_float("  Step  [eV]", 5.0)
        if step <= 0: step = 1.0
        grid = np.arange(start, end + 0.0001, step)
        print_info(f"Grid: {len(grid)} points from {start} to {end} eV")
        return grid
        
    elif choice == '3':
        raw = input("  Energies (e.g. 10, 15.5, 20): ")
        try:
            vals = [float(x.strip()) for x in raw.split(',')]
            print_info(f"Custom list: {len(vals)} points")
            return np.array(vals)
        except ValueError:
            print_warning("Invalid format. Using 50.0 eV.")
            return np.array([50.0])

    elif choice == '4':
        start = get_input_float("  Start [eV]", 10.0)
        end   = get_input_float("  End [eV]", 1000.0)
        dens  = get_input_float("  Density (1.0=normal)", 1.0)
        grid = generate_flexible_energy_grid(start, end, dens)
        print_info(f"Log grid: {len(grid)} points")
        # Return tuple with params for threshold-based regeneration
        return (grid, 'log', {'end': end, 'density': dens})
    
    print_warning("Invalid choice. Using default grid.")
    return np.arange(10.0, 201.0, 5.0)

# --- Data Management ---

def load_results(filename) -> dict:
    """
    Load existing results from a JSON file.
    
    Checks both results/ directory and root for backward compatibility.
    Returns an empty dict if file does not exist or is corrupt.
    
    Parameters
    ----------
    filename : str
        Filename (may or may not include 'results/' prefix).
    """
    from pathlib import Path
    
    # Normalize filename (remove any existing results/ prefix)
    base_name = Path(filename).name
    
    # Check results/ directory first (preferred location)
    results_path = get_results_dir() / base_name
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    # Fallback: check root directory for backward compatibility
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    return {}

def save_results(filename, new_data_dict) -> None:
    """
    Update and save results to JSON in the results/ directory.
    
    Merges new_data_dict into existing data (by key) to prevent data loss.
    Creates the results/ directory if it doesn't exist.
    
    Parameters
    ----------
    filename : str
        Filename (will be saved to results/ directory).
    new_data_dict : dict
        New data to merge into existing results.
    """
    from pathlib import Path
    
    # Ensure we save to results/ directory
    base_name = Path(filename).name
    output_path = get_output_path(base_name)
    
    # Load existing data (checks both locations)
    current = load_results(base_name)
    current.update(new_data_dict)
    
    with open(output_path, "w") as f:
        json.dump(current, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")


# --- Configuration File Discovery ---

def discover_config_files() -> list:
    """
    Find all .yaml config files in current directory and examples/.
    Excludes template files.
    """
    configs = glob.glob("*.yaml") + glob.glob("examples/*.yaml")
    # Filter out templates
    configs = [c for c in configs if "template" not in c.lower()]
    return sorted(configs)


def prompt_use_config_file(calc_type: str = "excitation") -> str | None:
    """
    Ask user if they want to use an existing config file.
    
    Parameters
    ----------
    calc_type : str
        "excitation" or "ionization" - filters relevant configs
        
    Returns
    -------
    str or None
        Path to selected config file, or None to continue interactively.
    """
    from config_loader import load_config
    
    configs = discover_config_files()
    if not configs:
        return None
    
    # Load and filter configs (single load per file)
    valid_configs = []  # List of (path, config_object)
    for cfg_path in configs:
        try:
            config = load_config(cfg_path)
            if config.calculation_type == calc_type:
                valid_configs.append((cfg_path, config))
        except Exception as e:
            logger.debug("Failed to load config %s: %s", cfg_path, e)
            # Don't include invalid configs
    
    if not valid_configs:
        return None
    
    # Clean UI display
    print_subheader("Configuration Files")
    for i, (path, cfg) in enumerate(valid_configs, 1):
        # Compact display: path and key info
        state_info = f"{cfg.states.initial.n}{['s','p','d','f'][min(cfg.states.initial.l, 3)]}"
        if calc_type == "excitation":
            state_info += f"→{cfg.states.final.n}{['s','p','d','f'][min(cfg.states.final.l, 3)]}"
        print(f"  {i}. {cfg.run_name} ({cfg.target.atom} {state_info}) - {path}")
    print(f"  {len(valid_configs)+1}. Configure interactively")
    
    choice = input(f"\n  Select [1-{len(valid_configs)+1}, default={len(valid_configs)+1}]: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(valid_configs):
        return valid_configs[int(choice)-1][0]
    
    return None

def check_file_exists_warning(filename) -> bool:
    """
    Warn the user if the output file already exists.
    Returns True if user wants to continue (append), False to abort.
    """
    # Check in results/ directory where files are actually saved
    results_path = os.path.join("results", filename)
    if os.path.exists(results_path):
        print(f"\n[WARNING] File '{results_path}' already exists!")
        print("New results will be appended/merged into this file.")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    return True

# --- Calculation Routines ---

def run_scan_excitation(run_name) -> None:
    """
    Interactive workflow for Excitation Cross Section calculation.
    """
    filename = f"results_{run_name}_exc.json"
    
    if not check_file_exists_warning(filename):
        print_error("Aborting calculation.")
        return

    print_header("EXCITATION CALCULATION")
    
    # --- Check for existing config files ---
    config_path = prompt_use_config_file("excitation")
    if config_path:
        print_info(f"Using config file: {config_path}")
        run_from_config(config_path, verbose=False)
        return
    
    # --- Continue with interactive mode ---
    atom_entry = select_target()
    Z = atom_entry.Z
      
    print_subheader("State Configuration")
    print("  Initial State:")
    ni = get_input_int("    n", atom_entry.default_n)
    li = get_input_int("    l", atom_entry.default_l)
    
    print("  Final State:")
    nf = get_input_int("    n", ni + 1)
    lf = get_input_int("    l", li)  # Default: same l (e.g., 1s→2s)
    
    # --- VALIDATION ---
    if li >= ni:
        print_error(f"Initial state impossible: l({li}) >= n({ni}).")
        return
    if lf >= nf:
        print_error(f"Final state impossible: l({lf}) >= n({nf}).")
        return
    if ni == nf and li == lf:
        print_error("Initial and Final states are identical (elastic scattering).")
        return
    if nf < ni:
        print_warning(f"De-excitation: n_f({nf}) < n_i({ni})")
        confirm = input("    Continue? [y/N]: ").strip().lower()
        if confirm != 'y': return
    if abs(lf - li) > 3:
        print_warning(f"Large ΔL = {abs(lf-li)} - cross sections may be very small.")
    
    print_subheader("Physics Model")
    print("  1. Static DWBA")
    print("  2. Static DWBA + Polarization")
    choice = input("\n  Select [default=1]: ").strip()
    
    use_pol = False
    
    if choice == '2':
        use_pol = True
        print_info("Model: Static DWBA + Polarization")
        print_warning("Polarization potential is heuristic (not in the article DWBA).")
        logger.warning("Excitation: polarization potential is heuristic (not in the article DWBA).")
    else:
        print_info("Model: Static DWBA")
    
    # Get energy grid (may return tuple for log grid with regeneration params)
    energy_result = get_energy_list_interactive()
    if isinstance(energy_result, tuple):
        energies, grid_type, grid_params = energy_result
    else:
        energies = energy_result
        grid_type, grid_params = None, None
    
    # Convert quantum number n to sorting index n_index
    # For a given l, the states are n=l+1, l+2, ...
    # So index = n - l.
    n_idx_i = ni - li
    n_idx_f = nf - lf
    
    if n_idx_i < 1 or n_idx_f < 1:
        print("Error: Invalid n, l combination (n must be > l).")
        return

    # Use centralized defaults with improved UI
    params = prompt_use_defaults(categories=['grid', 'excitation', 'oscillatory', 'hardware', 'output'])
    
    L_max_integrals = params['excitation']['L_max_integrals']
    L_max_proj = params['excitation']['L_max_projectile']
    n_theta = params['excitation']['n_theta']
    pilot_E = params['excitation']['pilot_energy_eV']
    do_calibrate = params['output']['calibrate']
    
    # Set oscillatory configuration globally (merge oscillatory + hardware for backward compat)
    osc_config = {**params['oscillatory'], **params['hardware']}
    set_oscillatory_config(osc_config)


    spec = ExcitationChannelSpec(
        l_i=li, l_f=lf,
        n_index_i=n_idx_i, n_index_f=n_idx_f,
        N_equiv=1, # SAE approximation
        L_max_integrals=L_max_integrals, 
        L_target_i=li,
        L_target_f=lf,
        L_max_projectile=L_max_proj
    )
    
    # Use parameters from library
    core_params = atom_entry.core_params

    key = f"Excitation_Z{Z}_{atom_entry.name}_n{ni}l{li}_to_n{nf}l{lf}"
    results = []
    
    print_subheader("Calibration")
    
    # Transition class for calibration: dipole (|Dl|=1) vs non-dipole (|Dl|!=1)
    delta_l = abs(lf - li)
    t_class = "dipole" if delta_l == 1 else "non_dipole"
    logger.debug(f"Transition |Dl|={delta_l} mapped to class '{t_class}'")
    
    alpha = 1.0
    dE_thr = atom_entry.ip_ev  # Default threshold from atom library
    epsilon_exc_au = -0.5 * (Z**2) / (nf**2)  # Rough estimate
    tong_model = None
    res_pilot = None
    
    if do_calibrate:
        # Use shared helper function for pilot calibration
        alpha, tong_model, res_pilot = run_pilot_calibration(
            spec=spec,
            core_params=core_params,
            params=params,
            use_pol=use_pol,
            threshold_eV=dE_thr,
            epsilon_exc_au=epsilon_exc_au,
            transition_class=t_class,
            n_theta=n_theta
        )
        
        # Update threshold from pilot result if available
        if res_pilot:
            dE_thr = res_pilot.E_excitation_eV
            E_init_eV = -atom_entry.ip_ev
            E_final_eV = E_init_eV + dE_thr
            epsilon_exc_au = ev_to_au(E_final_eV)
            # Recreate TongModel with correct threshold
            tong_model = TongModel(dE_thr, epsilon_exc_au, transition_class=t_class)
            if res_pilot.sigma_total_cm2 > 0:
                alpha = tong_model.calibrate_alpha(1000.0, res_pilot.sigma_total_cm2)
            print_info(f"Threshold: {dE_thr:.2f} eV, α = {alpha:.3f}")
        else:
            print_info("Using default α = 1.0")
    else:
        print_info("Calibration DISABLED (output.calibrate=false)")
        print_info("Using α = 1.0, no Tong model curve will be generated")

    # --- Smart Grid Adjustment ---
    # If user's start is below threshold, regenerate grid with threshold as new start
    logger.debug("Threshold dE_thr = %.3f eV", dE_thr)
    if dE_thr > 0:
        start_epsilon = 0.5  # Start 0.5 eV above threshold
        new_start = dE_thr + start_epsilon
        
        # Check if any points are below threshold
        if np.any(energies < dE_thr):
            # For log grid: regenerate with new start preserving end and density
            if grid_type == 'log' and grid_params:
                end_eV = grid_params['end']
                density = grid_params['density']
                energies = generate_flexible_energy_grid(new_start, end_eV, density)
                print_info(f"Threshold: {dE_thr:.2f} eV, α = {alpha:.3f}")
                print_info(f"Grid adjusted: {len(energies)} points from {new_start:.1f} eV")
            else:
                # For other grid types: filter and add start point
                energies = energies[energies > dE_thr]
                if len(energies) == 0 or energies[0] > new_start + 0.1:
                    energies = np.insert(energies, 0, new_start)
                energies = np.unique(np.round(energies, 3))
                print_info(f"Grid adjusted: {len(energies)} points from {energies[0]:.1f} eV")
            
    # --- Grid Strategy Handling (v2.6+) ---
    # 
    # Three modes:
    # - MANUAL: Use fixed r_max/n_points from params (user-defined)
    # - GLOBAL: Calculate optimal grid once for min energy in scan (recommended)
    # - LOCAL:  Recalculate grid for each energy point (most accurate, slower)
    #
    strategy = params['grid'].get('strategy', 'global').lower()
    base_r_max = params['grid']['r_max']
    base_n_points = params['grid']['n_points']
    scale_factor = params['grid'].get('r_max_scale_factor', 2.5)
    n_points_max = params['grid'].get('n_points_max', 15000)
    min_pts_per_wl = params['grid'].get('min_points_per_wavelength', 15)
    
    # Ionic charge for Coulomb asymptotic validity requirement
    z_ion = core_params.Zc - 1.0  # For single-electron: He+ has Zc=2, z_ion=1
    
    E_min_scan = float(np.min(energies))
    E_max_scan = float(np.max(energies))
    
    # Use worst-case scan energy to size grid for effective runtime L usage.
    L_max_effective = estimate_effective_projectile_lmax(E_max_scan, L_max_proj)
    
    print_subheader("Grid Configuration")
    
    if strategy == 'manual':
        # MANUAL: Use exactly what user specified
        r_max_calc = base_r_max
        n_points_calc = base_n_points
        print_info(f"Strategy: MANUAL (fixed parameters)")
        print_info(f"  r_max = {r_max_calc:.1f} a.u., n_points = {n_points_calc}")
        
    elif strategy == 'local':
        # LOCAL: Will recalculate per energy, but start with E_min for initial prep
        r_max_calc, n_points_calc = calculate_optimal_grid_params(
            E_min_scan, L_max_effective, base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
        )
        print_info(f"Strategy: LOCAL (per-energy adaptive)")
        print_info(f"  Initial grid (E_min={E_min_scan:.1f} eV): r_max = {r_max_calc:.1f}, n_points = {n_points_calc}")
        print_info(f"  Grid will be recalculated for each energy point.")
                   
    else:  # 'global' (default)
        # GLOBAL: Calculate optimal grid for lowest energy, use for all
        k_min = k_from_E_eV(E_min_scan - dE_thr) if E_min_scan > dE_thr else k_from_E_eV(0.5)
        r_max_calc, n_points_calc = calculate_optimal_grid_params(
            E_min_scan, L_max_effective, base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
        )
        print_info(f"Strategy: GLOBAL (single adaptive calculation)")
        print_info(f"  E_min = {E_min_scan:.1f} eV, k_min = {k_min:.2f} a.u.")
        print_info(f"  r_max = {r_max_calc:.1f} a.u., n_points = {n_points_calc}")
    
    # --- Pre-calculate static target properties (Optimization) ---
    print("\n[Optimization] Pre-calculating static target properties...")
    from driver import prepare_target, compute_excitation_cs_precalc
    
    prep = prepare_target(
        chan=spec,
        core_params=core_params,
        use_polarization=use_pol,
        r_max=r_max_calc,
        n_points=n_points_calc
    )
    print_success("Target prepared")


    run_meta = {
        "model": "static+polarization" if use_pol else "static",
        "use_polarization": use_pol,
        "grid": {
            "strategy": strategy.upper(),
            "r_min": float(prep.grid.r[0]),
            "r_max": float(prep.grid.r[-1]),
            "n_points": len(prep.grid.r),
            "base_r_max": base_r_max,
            "base_n_points": base_n_points,
        },
        "numerics": {
            "L_max_integrals": spec.L_max_integrals,
            "L_max_projectile_base": spec.L_max_projectile,
            "n_theta": n_theta,
        },
        "calibration": {
            "alpha": alpha,
            "transition_class": t_class,
            "pilot_energy_eV": pilot_E,
            "threshold_eV": dE_thr,
        },
    }
    
    print_subheader(f"Calculation: {len(energies)} points")
    print("  " + "-" * 45)
    print(f"  {'Energy':>10}  │  {'Cross Section':>15}")
    print("  " + "-" * 45)
    
    # Reset scan-level logging for clean output
    reset_scan_logging()
    
    # Validate high-energy points upfront
    E_max = max(energies)
    he_warnings = validate_high_energy(E_max, L_max_proj, r_max_calc, n_points_calc)
    for warn in he_warnings:
        logger.warning(warn)
    
    try:
        for i_E, E in enumerate(energies):
            if E <= 0.01: continue
            try:
                # LOCAL strategy: recalculate grid for each energy
                current_prep = prep
                if strategy == 'local':
                    # Effective L for this energy (kept consistent with runtime logic)
                    L_eff_local = estimate_effective_projectile_lmax(E, L_max_proj)
                    
                    r_local, n_local = calculate_optimal_grid_params(
                        E, L_eff_local, base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
                    )
                    
                    # For first iteration, ALWAYS recalculate to ensure correct size
                    if i_E == 0:
                        current_prep = prepare_target(
                            chan=spec,
                            core_params=core_params,
                            use_polarization=use_pol,
                            r_max=r_local,
                            n_points=n_local
                        )
                        logger.info("Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d (initial)", 
                                   E, r_local, n_local)
                        prep = current_prep
                    else:
                        # Check if grid parameters changed
                        current_r_max = prep.grid.r[-1]
                        current_n_pts = len(prep.grid.r)
                        params_changed = (abs(r_local - current_r_max) > 0.1 or 
                                         abs(n_local - current_n_pts) > 1)
                        
                        if params_changed:
                            current_prep = prepare_target(
                                chan=spec,
                                core_params=core_params,
                                use_polarization=use_pol,
                                r_max=r_local,
                                n_points=n_local
                            )
                            logger.info("Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d", 
                                       E, r_local, n_local)
                            prep = current_prep  # Update reference for next comparison
                
                res = compute_excitation_cs_precalc(E, current_prep, n_theta=n_theta)
                
                # Log partial waves to debug
                if res.partial_waves:
                    sorted_pws = sorted(res.partial_waves.items(), key=lambda x: x[1], reverse=True)[:3]
                    logger.debug("E=%.1f: %s", E, ", ".join(f"{k}={v:.1e}" for k,v in sorted_pws))
                
                # Calibration values (only if calibration is enabled)
                tong_sigma = None
                cal_factor = 1.0
                if do_calibrate and tong_model is not None:
                    try:
                        tong_sigma = tong_model.calculate_sigma_cm2(E)
                        if res.sigma_total_cm2 > 0 and tong_model.is_calibrated:
                            cal_factor = tong_sigma / res.sigma_total_cm2
                    except Exception as cal_err:
                        logger.debug("Calibration calc failed: %s", cal_err)
                        tong_sigma = None
                        cal_factor = 1.0
                
                # Print result with or without calibration factor
                if do_calibrate:
                    print_result(E, res.sigma_total_cm2, extra=f"[C(E)={cal_factor:.3f}]")
                else:
                    print_result(E, res.sigma_total_cm2)
                
                # Calibrate DCS (a.u.) if available
                dcs_raw_au = None
                dcs_cal_au = None
                theta_deg = None
                if res.dcs_au is not None and res.theta_deg is not None:
                    theta_deg = res.theta_deg.tolist() if hasattr(res.theta_deg, "tolist") else list(res.theta_deg)
                    dcs_raw_au = res.dcs_au.tolist() if hasattr(res.dcs_au, "tolist") else list(res.dcs_au)
                    if do_calibrate:
                        dcs_cal_au = (res.dcs_au * cal_factor).tolist()
                    else:
                        dcs_cal_au = None

                runtime_meta = dict(res.metadata or {})
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "sigma_mtong_cm2": tong_sigma,  # null when calibration disabled
                    "calibration_alpha": alpha,
                    "calibration_factor": cal_factor,
                    "theta_deg": theta_deg,
                    "dcs_au_raw": dcs_raw_au,
                    "dcs_au_calibrated": dcs_cal_au,
                    "partial_waves": res.partial_waves,
                    "L_max_integrals_used": runtime_meta.get("L_max_integrals_used"),
                    "L_max_projectile_used": runtime_meta.get("L_max_projectile_used"),
                    "L_dynamic_required": runtime_meta.get("L_dynamic_required"),
                    "maxL_in_result": runtime_meta.get("L_max_projectile_summed"),
                    "n_projectile_partial_waves_summed": runtime_meta.get("n_projectile_partial_waves_summed"),
                    "meta": {
                        **run_meta,
                        "runtime": runtime_meta,
                    }
                })
                
            except Exception as e:
                print_error(f"{E:.2f} eV: {e}")

    except KeyboardInterrupt:
        print()
        print_warning(f"Interrupted - saving {len(results)} points...")
        save_results(filename, {key: results})
        print_success("Partial results saved")
        return

    print("  " + "-" * 45)
    save_results(filename, {key: results})
    print_success(f"Complete: {len(results)} points saved")


def run_scan_ionization(run_name) -> None:
    """
    Interactive workflow for Ionization Cross Section calculation.
    """
    filename = f"results_{run_name}_ion.json"
    
    if not check_file_exists_warning(filename):
        print("Aborting calculation.")
        return

    print("\n=== IONIZATION CALCULATION ===")
    
    # --- Check for existing config files ---
    config_path = prompt_use_config_file("ionization")
    if config_path:
        print_info(f"Using config file: {config_path}")
        run_from_config(config_path, verbose=False)
        return
    
    # --- Continue with interactive mode ---
    atom_entry = select_target()
    Z = atom_entry.Z
    print(f"Selected Target: {atom_entry.name} (Z={Z})")
    
    print("Initial State:")
    ni = get_input_int("  n", atom_entry.default_n)
    li = get_input_int("  l", atom_entry.default_l)
    
    if li >= ni:
        print(f"\n[ERROR] Initial state impossible: l({li}) >= n({ni}).")
        return
    
    print("\n--- Model Selection ---")
    print("1. Static Only (Standard DWBA)")
    print("2. Static + Polarization (SEP)")
    choice = input("Select Model [1-2] (default=1): ").strip()

    use_pol = False

    if choice == '2':
        use_pol = True
        print_info("Model: Static DWBA + Polarization")
        print_warning("Polarization potential is heuristic (not in the article DWBA).")
        logger.warning("Ionization: polarization potential is heuristic (not in the article DWBA).")
    else:
        print_info("Model: Static DWBA")

    energy_result = get_energy_list_interactive()
    if isinstance(energy_result, tuple):
        energies, grid_type, grid_params = energy_result
    else:
        energies = energy_result
        grid_type, grid_params = None, None

    # --- Smart Grid Adjustment (Ionization) ---
    ion_thr = atom_entry.ip_ev
    if np.any(energies <= ion_thr):
        print(f"\n[Smart Grid] Some energies are below/at IP ({ion_thr:.2f} eV). Correcting...")
        start_epsilon = 0.5
        new_start = ion_thr + start_epsilon
        if grid_type == 'log' and grid_params:
            end_eV = grid_params['end']
            density = grid_params['density']
            energies = generate_flexible_energy_grid(new_start, end_eV, density)
            print(f"[Smart Grid] New log grid: {len(energies)} points from {new_start:.2f} eV")
        else:
            energies = energies[energies > ion_thr]
            if len(energies) == 0 or energies[0] > new_start + 0.1:
                energies = np.insert(energies, 0, new_start)
            energies = np.unique(np.round(energies, 3))
            print(f"[Smart Grid] New start: {energies[0]:.2f} eV")
    
    n_idx_i = ni - li
    if n_idx_i < 1:
        print("Error: Invalid n, l combination (n must be > l).")
        return
    
    # Use centralized defaults with improved UI
    params = prompt_use_defaults(categories=['grid', 'ionization', 'oscillatory', 'hardware', 'output'])
    
    l_eject_max = params['ionization']['l_eject_max']
    L_max = params['ionization']['L_max']
    L_max_proj = params['ionization']['L_max_projectile']
    n_energy_steps = params['ionization']['n_energy_steps']
    energy_quadrature = params['ionization'].get('energy_quadrature', 'gauss_legendre')
    
    # Set oscillatory configuration globally (merge oscillatory + hardware for backward compat)
    osc_config = {**params['oscillatory'], **params['hardware']}
    set_oscillatory_config(osc_config)

    spec = IonizationChannelSpec(
        l_i=li,
        n_index_i=n_idx_i,
        N_equiv=1,
        l_eject_max=l_eject_max,
        L_max=L_max,
        L_i_total=li,
        L_max_projectile=L_max_proj,
        energy_quadrature=energy_quadrature
    )
    
    # Use parameters from library
    core_params = atom_entry.core_params
    
    key = f"Ionization_Z{Z}_{atom_entry.name}_n{ni}l{li}"
    # --- Main Loop ---

    # Optional TDCS angles (angle-resolved)
    tdcs_angles = None
    tdcs_choice = input("\nCompute TDCS (angle-resolved)? [y/N]: ").strip().lower()
    if tdcs_choice == 'y':
        print("Enter angle triplets (theta_scatt, theta_eject, phi_eject) in degrees.")
        print("Example: 30,30,180; 45,60,180  (use ';' to separate triplets)")
        raw = input("Angles: ").strip()
        if raw:
            tdcs_angles = []
            for group in [g.strip() for g in raw.split(';') if g.strip()]:
                parts = [p for p in group.replace(',', ' ').split() if p]
                if len(parts) not in (2, 3):
                    print_warning(f"Skipping invalid angle entry: '{group}'")
                    continue
                try:
                    th_scatt = float(parts[0])
                    th_eject = float(parts[1])
                    ph_eject = float(parts[2]) if len(parts) == 3 else 0.0
                    tdcs_angles.append((th_scatt, th_eject, ph_eject))
                except ValueError:
                    print_warning(f"Skipping invalid angle entry: '{group}'")
            if not tdcs_angles:
                tdcs_angles = None
    
    # --- Adaptive r_max and n_points using unified function (v2.7+) ---
    E_min_scan = float(np.min(energies))
    E_max_scan = float(np.max(energies))
    
    # Get grid parameters from params (supports user config)
    base_r_max = params['grid'].get('r_max', 200.0)
    base_n_points = params['grid'].get('n_points', 4000)
    scale_factor = params['grid'].get('r_max_scale_factor', 2.5)
    n_points_max = params['grid'].get('n_points_max', 10000)
    min_pts_per_wl = params['grid'].get('min_points_per_wavelength', 15)
    
    # L_max_effective: estimate maximum L that will be used at runtime.
    L_eff = estimate_effective_projectile_lmax(E_max_scan, L_max_proj)
    k_sampling_ion = estimate_ionization_worst_oscillatory_k(E_max_scan, ion_thr)
    z_ion = core_params.Zc - 1.0

    # Use unified grid calculation with wavelength scaling.
    # For ionization, use worst oscillatory scale k_scatt+k_eject.
    r_max_optimal, n_points_optimal = calculate_optimal_grid_params(
        E_min_scan, L_eff, base_r_max, base_n_points, 
        scale_factor, n_points_max, min_pts_per_wl, z_ion,
        k_sampling_au=k_sampling_ion
    )
    
    print_info(f"Adaptive Grid: E_min={E_min_scan:.1f} eV -> r_max={r_max_optimal:.0f}, n_points={n_points_optimal}")
    
    # Pre-calculate (Optimization)
    print("\n[Optimization] Pre-calculating static target properties...")
    from driver import prepare_target, ExcitationChannelSpec
    
    # Adapter: prepare_target expects ExcitationChannelSpec but Ionization needs IonizationChannelSpec.
    # We can fake the ExcitationChannelSpec just to get grid, V_core, and orb_i.
    # We set l_f to something valid (l_i + 1) just to pass the solver checks, though we won't use orb_f.
    
    tmp_chan = ExcitationChannelSpec(
        l_i=li, l_f=li+1, n_index_i=n_idx_i, n_index_f=n_idx_i+1, # dummy final
        N_equiv=1, L_max_integrals=10, L_target_i=li, L_target_f=li+1
    )
    
    prep = prepare_target(
        chan=tmp_chan,
        core_params=core_params,
        use_polarization=use_pol,
        r_max=r_max_optimal,
        n_points=n_points_optimal
    )
    print("[Optimization] Ready.")


    print(f"\nStarting calculation for {key} ({len(energies)} points)...")
    
    results = []
    
    try:
        for E in energies:
            print(f"E={E:.2f} eV...", end=" ", flush=True)
            try:
                res = compute_ionization_cs(
                    E, spec, 
                    core_params=core_params, # Ignored for grid/V_core generation if _precalc used
                    r_max=r_max_optimal, n_points=n_points_optimal,
                    use_polarization=use_pol,
                    tdcs_angles_deg=tdcs_angles,
                    n_energy_steps=n_energy_steps,
                    _precalc_grid=prep.grid,
                    _precalc_V_core=prep.V_core,
                    _precalc_orb_i=prep.orb_i
                )
                
                if res.partial_waves:
                    sorted_pws = sorted(res.partial_waves.items(), key=lambda x: x[1], reverse=True)[:3]
                    logger.debug("Ionization E=%.1f: %s", E, ", ".join(f"{k}={v:.1e}" for k, v in sorted_pws))
                
                if res.sigma_total_cm2 > 0:
                    print(f"OK. σ={res.sigma_total_cm2:.2e} cm2")
                else:
                    print(f"Below Threshold (IP={res.IP_eV:.2f} eV)")
                
                results.append({
                    "energy_eV": E,
                    "sigma_au": res.sigma_total_au,
                    "sigma_cm2": res.sigma_total_cm2,
                    "IP_eV": res.IP_eV,
                    "sdcs": res.sdcs_data,
                    "tdcs": res.tdcs_data,
                    "partial_waves": res.partial_waves,
                    "meta": {
                        **(res.metadata or {}),
                        "tdcs_angles_deg": tdcs_angles,
                    }
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed: {e}")

    except KeyboardInterrupt:
        print("\n\n[STOP] Calculation interrupted by user (Ctrl+C).")
        print(f"[INFO] Saving {len(results)} data points collected so far...")
        save_results(filename, {key: results})
        print("[INFO] Partial results saved successfully.")
        return

    save_results(filename, {key: results})
    print("Calculation complete.")

# --- Visualization ---

def run_visualization() -> None:
    print("\n=== PLOT GENERATION ===")
    
    # List JSON files from results/ and root (backward compatibility)
    files = find_result_files("results_*.json")
    if not files:
        print("No result files found in results/ directory.")
        return
        
    print("Available Result Files:")
    for idx, f in enumerate(files):
        print(f"{idx+1}. {f}")
        
    print("0. Cancel")
    
    try:
        choice_idx = int(input("Select file number: ")) - 1
    except ValueError:
        return
        
    if choice_idx < 0 or choice_idx >= len(files):
        return
        
    selected_file = str(files[choice_idx])  # Convert Path to string
    
    print("\nSelect Style:")
    print("1. Standard (eV / cm^2)")
    print("2. Atomic (Ha / a0^2)")
    print("3. Article (E/Thr / pi*a0^2)")
    print("4. Mixed (eV / a.u.)")
    
    choice = input("Choice [1]: ").strip()
    
    # Map to plotter style args
    style_map = {
        '1': 'std',
        '2': 'atomic',
        '3': 'article',
        '4': 'ev_au'
    }
    style = style_map.get(choice, 'std')
    
    print(f"Generating plot from '{selected_file}' with style '{style}'...")
    
    # Pass arguments to plotter
    import plotter
    old_argv = sys.argv
    sys.argv = ["plotter.py", style, selected_file]
    try:
        plotter.main()
    except SystemExit:
        pass 
    except Exception as e:
        print(f"Plotter Error: {e}")
    finally:
        sys.argv = old_argv

def run_dcs_visualization() -> None:
    print("\n=== ANGULAR DCS PLOT GENERATION ===")
    
    # List JSON files from results/ and root (backward compatibility)
    files = find_result_files("results_*.json")
    if not files:
        print("No result files found in results/ directory.")
        return
        
    print("Available Result Files:")
    for idx, f in enumerate(files):
        print(f"{idx+1}. {f}")
        
    print("0. Cancel")
    
    try:
        choice_idx = int(input("Select file number: ")) - 1
    except ValueError:
        return
        
    if choice_idx < 0 or choice_idx >= len(files):
        return
        
    selected_file = str(files[choice_idx])  # Convert Path to string
    
    # Load file to show available energies
    try:
        with open(selected_file, 'r') as f:
            data = json.load(f)
        # Get all energies from first key
        first_key = list(data.keys())[0]
        all_energies = [p["energy_eV"] for p in data[first_key] if p.get("theta_deg")]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.debug("Could not parse energy list from results: %s", e)
        all_energies = []
    
    if all_energies:
        print(f"\nAvailable energies: {len(all_energies)}")
        print("  1. Plot ALL energies")
        print("  2. Select specific energies")
        
        energy_choice = input("\n  Select [1]: ").strip()
        
        if energy_choice == '2':
            print(f"\nAvailable: {', '.join([f'{e:.1f}' for e in all_energies])} eV")
            user_input = input("Enter energies (comma separated, e.g. 50, 100, 200): ").strip()
            try:
                selected_energies = [float(x.strip()) for x in user_input.split(',')]
                # Pass selected energies to plotter via environment variable
                os.environ['DCS_ENERGIES'] = ','.join([str(e) for e in selected_energies])
            except ValueError as e:
                logger.debug("Invalid energy input: %s", e)
                print("Invalid input. Using all energies.")
                if 'DCS_ENERGIES' in os.environ:
                    del os.environ['DCS_ENERGIES']
        else:
            if 'DCS_ENERGIES' in os.environ:
                del os.environ['DCS_ENERGIES']
    
    print(f"Generating DCS plots from '{selected_file}'...")
    
    import plotter
    old_argv = sys.argv
    sys.argv = ["plotter.py", "dcs", selected_file]
    try:
        plotter.main()
    except SystemExit:
        pass 
    except Exception as e:
        print(f"Plotter Error: {e}")
    finally:
        sys.argv = old_argv
        if 'DCS_ENERGIES' in os.environ:
            del os.environ['DCS_ENERGIES']

# --- Main Loop ---

def main() -> None:
    print_header("DWBA CALCULATION SUITE")
    
    run_name = input("\n  Simulation Name [default]: ").strip()
    if not run_name:
        run_name = "default"
    
    print_info(f"Run: {run_name}")
    print_info(f"Output: results/results_{run_name}_exc.json / _ion.json")

    while True:
        print_subheader(f"Main Menu  │  Run: {run_name}")
        print("  1. Excitation Cross Sections")
        print("  2. Ionization Cross Sections")
        print("  3. Total Cross Sections Plots")
        print("  4. Angular DCS Plots")
        print("  5. Partial Wave Analysis")
        print("  6. Fit Potential (New Atom)")
        print("  7. Change Run Name")
        print("  q. Quit")
        
        choice = input("\n  Select: ").strip().lower()
        
        if choice == '1':
            run_scan_excitation(run_name)
        elif choice == '2':
            run_scan_ionization(run_name)
        elif choice == '3':
            run_visualization()
        elif choice == '4':
            run_dcs_visualization()
        elif choice == '5':
            # Run Partial Wave Analysis
            try:
                import partial_wave_plotter
                print("\n=== PARTIAL WAVE ANALYSIS ===")
                # It expects a file argument normally, but we can wrap it or call main() if it allows.
                # Inspecting partial_wave_plotter.py suggests it has a main() that asks for file.
                # Let's try calling its main logic.
                partial_wave_plotter.main()
            except Exception as e:
                print(f"Error running plotter: {e}")
        elif choice == '6':
            # Run Potential Fitter
            try:
                import fit_potential
                fit_potential.main()
            except Exception as e:
                print(f"Error running fitter: {e}")
        elif choice == '7':
            new_name = input("Enter new Simulation Name: ").strip()
            if new_name:
                old_name = run_name
                run_name = new_name
                print_success(f"Run Name changed to: {run_name}")
                
                # Offer to rename existing files (Bug #5 fix)
                old_files = list(get_results_dir().glob(f"results_{old_name}_*.json"))
                old_plots = list(get_results_dir().glob(f"*_{old_name}_*.png"))
                all_old_files = old_files + old_plots
                
                if all_old_files:
                    print_info(f"Found {len(all_old_files)} files with old name '{old_name}'.")
                    rename_choice = input("  Rename existing files to new name? [y/N]: ").strip().lower()
                    if rename_choice == 'y':
                        renamed_count = 0
                        for old_path in all_old_files:
                            new_filename = old_path.name.replace(old_name, new_name)
                            new_path = old_path.parent / new_filename
                            try:
                                old_path.rename(new_path)
                                print(f"    {old_path.name} → {new_filename}")
                                renamed_count += 1
                            except Exception as e:
                                print_warning(f"    Failed to rename {old_path.name}: {e}")
                        print_success(f"Renamed {renamed_count}/{len(all_old_files)} files.")
        elif choice == 'q':
            print("Goodbye.")
            break
        else:
            print("Invalid selection.")

# =============================================================================
# BATCH MODE EXECUTION
# =============================================================================


# =============================================================================
# Helper Utilities
# =============================================================================

def estimate_effective_projectile_lmax(
    E_eV: float,
    L_max_projectile_base: int,
    dynamic_scale: float = 8.0,
    dynamic_offset: int = 5,
    chi_f_buffer: int = 15,
) -> int:
    """
    Estimate effective projectile L_max used by continuum caching/propagation.

    Runtime uses a dynamic estimate L_dynamic ~= k*8+5 and final-channel
    wave caching with an additional buffer (+15). This helper centralizes that
    logic so grid sizing and runtime numerics stay consistent.
    """
    k_au = k_from_E_eV(E_eV)
    L_dynamic = int(k_au * dynamic_scale) + dynamic_offset
    return max(L_dynamic + chi_f_buffer, int(L_max_projectile_base) + chi_f_buffer)


def estimate_ionization_worst_oscillatory_k(E_incident_eV: float, threshold_eV: float) -> float:
    """
    Estimate worst oscillatory k-scale for ionization radial kernels.

    For fixed final kinetic sum E_scatt + E_eject = E_inc - IP, the maximum
    k_scatt + k_eject occurs near equal energy sharing.
    """
    E_inc = max(float(E_incident_eV), 0.0)
    E_thr = max(float(threshold_eV), 0.0)
    if E_inc <= E_thr:
        return 0.0
    E_total_final = E_inc - E_thr
    k_inc = float(k_from_E_eV(E_inc))
    k_half = float(k_from_E_eV(0.5 * E_total_final))
    return max(k_inc, 2.0 * k_half)


def resolve_grid_r_max_for_prep(base_r_max: float | str, fallback_auto: float = 200.0) -> float:
    """
    Resolve r_max to a numeric value for target preparation.

    When config uses r_max='auto', target preparation still needs a concrete
    value to build the grid. We use a stable fallback (200 a.u.) for this step.
    """
    if isinstance(base_r_max, str):
        if base_r_max.strip().lower() == "auto":
            return float(fallback_auto)
        raise ValueError(f"Unsupported r_max string '{base_r_max}'. Use numeric value or 'auto'.")
    return float(base_r_max)


def calculate_optimal_grid_params(
    E_eV: float, 
    L_max_proj: int,
    base_r_max: float | str,
    base_n_points: int,
    scale_factor: float = 2.5,
    n_points_max: int = 15000,
    min_points_per_wavelength: int = 15,
    z_ion: float = 0.0,
    k_sampling_au: float | None = None,
) -> tuple[float, int]:
    """
    Calculate optimal radial grid size based on energy and projectile L_max.
    
    Strategy:
    1. Determine minimum r_max needed to contain the classical turning point
       for L_max_proj with a given safety margin.
    2. For ionic targets (z_ion > 0), also ensure Coulomb asymptotic validity.
    3. Enforce the user's base_r_max as a minimum floor.
    4. Scale n_points proportionally to maintain grid density.
    5. (v2.7+) For high energies, ensure sufficient points per wavelength
       to accurately sample oscillations.
    
    Parameters
    ----------
    base_r_max : float | str
        User-configured base radius. Accepts numeric value or "auto".
        For "auto", the floor is determined by physics (`r_needed`) and
        density scaling uses a reference radius of 200 a.u.
    min_points_per_wavelength : int
        Minimum grid points per wavelength at large r. Default 15.
        Set to 0 to disable wavelength-based scaling.
    z_ion : float
        Ionic charge of target (0 for neutral, 1 for He+, etc.).
        Affects r_max requirement for Coulomb asymptotic validity.
    k_sampling_au : float | None
        Optional oscillatory wavenumber override used only for wavelength-based
        n_points scaling (does not affect turning-point r_max logic).
    
    Returns
    -------
    (r_max_optimal, n_points_optimal)
    """
    # Get wave number
    k_au = k_from_E_eV(E_eV)
    
    # Compute physically required r_max from turning point AND Coulomb logic
    r_needed = compute_required_r_max(k_au, L_max_proj, scale_factor, z_ion)
    
    # Handle 'auto' string for base_r_max.
    # If 'auto', we use r_needed as the floor and scale density against a
    # reference radius (200 a.u.) to avoid under-resolving very large grids.
    is_auto_rmax = str(base_r_max).strip().lower() == 'auto'
    base_r_val = 0.0 if is_auto_rmax else float(base_r_max)
    
    # We want at least the base config (e.g. 200 a.u.)
    # but extend if physics demands it (low energy / high L)
    r_max_optimal = max(base_r_val, r_needed)
    
    # --- Wavelength-based n_points scaling for high energies ---
    # For exponential grid: dr(r) ≈ r * ln(r_max/r_min) / n_points
    # 
    # v2.14+: Focus wavelength requirements on BOUND STATE REGION (r < 15 a.u.)
    # Beyond this, bound states have negligible density, and oscillatory
    # quadrature handles phase via Filon/Levin methods (which are phase-aware).
    # Requiring 15+ pts/wavelength at r=150 is impractical (needs ~50000 points).
    
    r_min = 1e-5  # Standard grid r_min
    
    k_wave = max(k_au, float(k_sampling_au) if k_sampling_au is not None else 0.0)
    if k_wave > 0.1 and min_points_per_wavelength > 0:  # Only for non-zero k
        wavelength = 2 * np.pi / k_wave
        log_ratio = np.log(r_max_optimal / r_min)
        
        # v2.14+: Check at r = 15 a.u. (outer edge of bound state region)
        # This is where accurate sampling matters for matrix elements.
        # Beyond r~15, bound states are negligible (<1%) and oscillatory
        # quadrature uses phase-aware Filon integration.
        r_check = 15.0  # Focus on bound state region instead of 0.7*r_max
        dr_needed = wavelength / min_points_per_wavelength
        
        # For exponential grid: dr(r) ≈ r * log_ratio / n
        # => n >= r * log_ratio / dr_needed
        n_wavelength_required = int(r_check * log_ratio / dr_needed) + 1
        
        # Also check at match point region (~50 a.u.) with relaxed requirement
        # (oscillatory quadrature handles this, but still need ~5 pts/wavelength)
        r_match_typical = 50.0
        min_pts_match = 5  # Relaxed for asymptotic region (Filon helps here)
        dr_match = wavelength / min_pts_match
        n_match_required = int(r_match_typical * log_ratio / dr_match) + 1
        
        n_wavelength_required = max(n_wavelength_required, n_match_required)
        
        # Use max of density-based and wavelength-based requirements
        n_points_optimal = max(base_n_points, n_wavelength_required)
    else:
        n_points_optimal = base_n_points
    
    # Check if we are extending r_max beyond base/reference.
    # For 'auto', use 200 a.u. as a practical density reference so n_points
    # still scales with large r_max values instead of staying fixed.
    density_ref_r = 200.0 if is_auto_rmax else base_r_val
    if density_ref_r > 0 and r_max_optimal > density_ref_r:
        # Scale points to keep density constant
        density = base_n_points / density_ref_r
        n_density_req = int(density * r_max_optimal)
        n_points_optimal = max(n_points_optimal, n_density_req)
    
    # Apply strict memory cap
    pre_cap = n_points_optimal
    n_points_optimal = min(n_points_optimal, n_points_max)
    
    # v2.14+: Warn if cap significantly limits required density
    if pre_cap > n_points_max * 1.5 and k_wave > 2.0:
        logger.warning(
            f"Grid cap limits phase sampling: need {pre_cap} pts for E={E_eV:.0f}eV, "
            f"capped at {n_points_max}. Consider increasing n_points_max for high energies."
        )
    
    # Log if we're scaling up significantly
    if n_points_optimal > base_n_points * 1.5:
        logger.debug(
            f"Grid scaled for E={E_eV:.0f}eV: n_points {base_n_points}→{n_points_optimal} "
            f"(k_wave={k_wave:.2f}, λ={2*np.pi/k_wave:.2f})"
        )
        
    return r_max_optimal, n_points_optimal


def run_pilot_calibration(
    spec: ExcitationChannelSpec,
    core_params: CorePotentialParams,
    params: dict,
    use_pol: bool,
    threshold_eV: float,
    epsilon_exc_au: float,
    transition_class: str,
    n_theta: int = 50
) -> tuple[float, 'TongModel', 'DWBAResult | None']:
    """
    Run pilot calibration calculation with adaptive grid.
    
    This function consolidates the pilot calculation logic used for Tong model
    calibration. It computes an optimal grid for the high-energy pilot point
    (typically 1000 eV) and returns the calibration factor α.
    
    Parameters
    ----------
    spec : ExcitationChannelSpec
        Channel specification (will be modified for pilot L_max values).
    core_params : CorePotentialParams
        Core potential parameters for the target atom.
    params : dict
        Full parameters dict containing 'grid' and 'excitation' sections.
    use_pol : bool
        Whether to use polarization potential.
    threshold_eV : float
        Excitation threshold energy in eV (for TongModel).
    epsilon_exc_au : float
        Final state energy in a.u. (for TongModel).
    transition_class : str
        "dipole" or "non_dipole" (for TongModel).
    n_theta : int
        Number of theta points for DCS (default 50 for speed).
        
    Returns
    -------
    tuple[float, TongModel, DWBAResult | None]
        (alpha, tong_model, pilot_result)
        alpha = 1.0 if calibration fails.
    """
    from dataclasses import replace
    from driver import compute_total_excitation_cs
    
    pilot_E = params['excitation'].get('pilot_energy_eV', 1000.0)
    
    # Pilot L_max configuration: "auto" = dynamic scaling, int = explicit value
    pilot_L_proj_cfg = params['excitation'].get('pilot_L_max_projectile', 'auto')
    pilot_L_int_cfg = params['excitation'].get('pilot_L_max_integrals', 'auto')
    
    # Convert pilot energy to wave number: k = sqrt(2*E/E_h) where E_h = 27.2114 eV (1 Hartree)
    # This is needed for both dynamic L_max and grid scaling calculations
    k_pilot = np.sqrt(2 * pilot_E / 27.2114)  # k in a.u. (bohr^-1)
    
    # Calculate dynamic L_max if needed
    if pilot_L_proj_cfg == 'auto' or pilot_L_int_cfg == 'auto':
        r_max_grid = params['grid']['r_max']
        # Handle 'auto' string - use default 200 a.u. for L_max estimation
        if str(r_max_grid).lower() == 'auto':
            r_max_grid = 200.0  # Default for pilot L_max calculation
        # Estimate L_max from classical turning point: L ~ k*r * scale_factor
        # 0.6 = conservative scale factor to ensure turning point is well within grid
        pilot_L_proj_dynamic = int(k_pilot * float(r_max_grid) * 0.6)
        
        if pilot_L_proj_cfg == 'auto':
            # Cap at 150 to prevent excessive computation time (L^2 scaling)
            pilot_L_proj = max(spec.L_max_projectile, min(pilot_L_proj_dynamic, 150))
        else:
            pilot_L_proj = int(pilot_L_proj_cfg)
        
        if pilot_L_int_cfg == 'auto':
            # L_max for multipole expansion: typically L_proj/4, capped at 25 for efficiency.
            # If production config uses L_max_integrals="auto", use a conservative floor.
            spec_lint_base = spec.L_max_integrals
            if isinstance(spec_lint_base, str):
                spec_lint_floor = 8 if spec_lint_base.strip().lower() == "auto" else 15
            else:
                spec_lint_floor = int(spec_lint_base)
            pilot_L_int = max(spec_lint_floor, min(25, pilot_L_proj // 4))
        else:
            pilot_L_int = int(pilot_L_int_cfg)
        
        logger.debug("Pilot dynamic L_max: L_proj=%d, L_int=%d (k=%.2f)", 
                    pilot_L_proj, pilot_L_int, k_pilot)
    else:
        # User specified explicit values
        pilot_L_proj = int(pilot_L_proj_cfg)
        pilot_L_int = int(pilot_L_int_cfg)
        logger.debug("Pilot explicit L_max: L_proj=%d, L_int=%d", pilot_L_proj, pilot_L_int)
    
    # Initialize TongModel
    tong_model = TongModel(threshold_eV, epsilon_exc_au, transition_class=transition_class)
    alpha = 1.0
    res_pilot = None
    try:
        # Calculate z_ion and L_max_effective for proper Coulomb r_max scaling
        z_ion = core_params.Zc - 1.0
        L_max_eff_pilot = estimate_effective_projectile_lmax(pilot_E, pilot_L_proj)
        
        # Calculate adaptive grid for pilot energy
        pilot_r_max, pilot_n_points = calculate_optimal_grid_params(
            pilot_E, 
            L_max_eff_pilot,
            base_r_max=params['grid']['r_max'],
            base_n_points=params['grid']['n_points'],
            scale_factor=params['grid'].get('r_max_scale_factor', 2.5),
            n_points_max=params['grid'].get('n_points_max', 15000),
            min_points_per_wavelength=params['grid'].get('min_points_per_wavelength', 15),
            z_ion=z_ion
        )
        
        # Log pilot calculation start with grid info
        logger.info("Pilot Calibrate | E=%.0f eV: r_max=%.1f a.u., n_points=%d, L_proj=%d", 
                    pilot_E, pilot_r_max, pilot_n_points, pilot_L_proj)
        
        # Create modified spec for pilot L_max values
        pilot_spec = replace(
            spec,
            L_max_integrals=pilot_L_int,
            L_max_projectile=pilot_L_proj
        )
        
        res_pilot = compute_total_excitation_cs(
            pilot_E, pilot_spec, core_params, 
            r_max=pilot_r_max, n_points=pilot_n_points, 
            use_polarization_potential=use_pol,
            n_theta=n_theta,
        )
        
        if res_pilot and res_pilot.sigma_total_cm2 > 0:
            alpha = tong_model.calibrate_alpha(pilot_E, res_pilot.sigma_total_cm2)
            logger.info("Calibration     | α = %.3f (threshold=%.2f eV)", alpha, threshold_eV)
        else:
            logger.warning("Pilot returned zero cross section. Using α=1.0")
            
    except Exception as e:
        logger.warning("Pilot failed: %s. Using α=1.0", e)
    
    return alpha, tong_model, res_pilot



def run_from_config(config_path: str, verbose: bool = False) -> None:
    """
    Run DWBA calculation from a configuration file (batch mode).
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    verbose : bool
        Enable verbose logging.
    """
    from config_loader import load_config, config_to_params_dict
    
    # Load and validate config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    run_name = config.run_name
    print(f"\n{'═'*50}")
    print(f"  BATCH MODE: {run_name}")
    print(f"{'═'*50}")
    print(f"  Config: {config_path}")
    print(f"  Type: {config.calculation_type}")
    print(f"  Target: {config.target.atom}")
    print(f"{'═'*50}\n")
    
    # Set up parameters
    params = config_to_params_dict(config)
    
    # Set oscillatory configuration (merge oscillatory + hardware for backward compat)
    osc_config = {**params['oscillatory'], **params['hardware']}
    set_oscillatory_config(osc_config)
    
    # Get atom and core parameters
    try:
        atom_entry = atom_library.get_atom(config.target.atom)
        core_params = atom_entry.core_params
        # If Z is not in core_params (it usually is Zc), get from entry
        # CorePotentialParams has Zc. AtomEntry has Z. They should match.
    except (ValueError, KeyError):
        if config.target.Z is not None:
            # Custom atom
            core_params = CorePotentialParams(
                Zc=config.target.Z,
                a1=0, a2=0, a3=0, a4=0, a5=0, a6=0
            )
        else:
            print(f"Error: Atom '{config.target.atom}' not found and no custom Z provided.")
            sys.exit(1)
    
    # Get states
    ni = config.states.initial.n
    li = config.states.initial.l
    nf = config.states.final.n
    lf = config.states.final.l
    
    # Build energy grid
    if config.energy.type == "single":
        energy_grid = [config.energy.start_eV]
    elif config.energy.type == "linear":
        step = config.energy.step_eV or 10.0
        energy_grid = np.arange(config.energy.start_eV, config.energy.end_eV + step, step).tolist()
    elif config.energy.type == "log":
        energy_grid = generate_flexible_energy_grid(
            config.energy.start_eV, 
            config.energy.end_eV, 
            config.energy.density
        )
    elif config.energy.type == "list":
        energy_grid = config.energy.values or [config.energy.start_eV]
    else:
        energy_grid = [config.energy.start_eV]
    
    logger.info("Energy Grid     | %d points (%.1f - %.1f eV)", 
               len(energy_grid), energy_grid[0], energy_grid[-1])
    
    # Run calculation
    if config.calculation_type == "excitation":
        from driver import prepare_target, compute_excitation_cs_precalc
        
        # Build spec
        spec = ExcitationChannelSpec(
            l_i=li, l_f=lf,
            n_index_i=ni-li, n_index_f=nf-lf,
            N_equiv=1,  # SAE: one active electron (match interactive path)
            L_max_integrals=params['excitation']['L_max_integrals'],
            L_target_i=li, L_target_f=lf,
            L_max_projectile=params['excitation']['L_max_projectile']
        )
        
        # --- Adaptive Grid Logic (v2.6+) ---
        strategy = params['grid'].get('strategy', 'global').lower()
        base_r_max = params['grid']['r_max']
        base_n_points = params['grid']['n_points']
        scale_factor = params['grid'].get('r_max_scale_factor', 2.5)
        n_points_max = params['grid'].get('n_points_max', 15000)
        min_pts_per_wl = params['grid'].get('min_points_per_wavelength', 15)
        
        # Determine parameters for initial target prep
        E_ref = min(energy_grid)
        E_max = max(energy_grid)
        L_eff = estimate_effective_projectile_lmax(E_max, params['excitation']['L_max_projectile'])
        z_ion = core_params.Zc - 1.0

        if strategy == 'manual':
            r_max_calc, n_points_calc = base_r_max, base_n_points
            logger.info("Grid Strategy   | MANUAL (r_max=%.1f a.u., n_points=%d)", 
                       r_max_calc, n_points_calc)
        elif strategy == 'local':
            r_max_calc, n_points_calc = calculate_optimal_grid_params(
                E_ref, L_eff,
                base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
            )
            logger.info("Grid Strategy   | LOCAL (per-energy adaptive)")
            logger.info("Grid Initial    | E_min=%.1f eV: r_max=%.1f a.u., n_points=%d", 
                       E_ref, r_max_calc, n_points_calc)
        else:  # 'global' (default)
            r_max_calc, n_points_calc = calculate_optimal_grid_params(
                E_ref, L_eff,
                base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
            )
            logger.info("Grid Strategy   | GLOBAL (E_min=%.1f eV: r_max=%.1f a.u., n_points=%d)", 
                       E_ref, r_max_calc, n_points_calc)

        # --- Pre-calculate static target properties ---
        logger.info("Target Prep     | Pre-calculating bound states...")
        use_pol = (config.physics_model == "polarization")
        
        # Initial preparation (serves as the fixed prep for Manual/Global)
        prep = prepare_target(
            chan=spec,
            core_params=core_params,
            use_polarization=use_pol,
            r_max=r_max_calc,
            n_points=n_points_calc
        )
        
        # Get threshold and binding energies from prep (no duplicate calculation)
        threshold_eV = prep.dE_target_eV
        E_f_au = prep.orb_f.energy_au  # For TongModel
        
        logger.info("Threshold       | %.2f eV", threshold_eV)
        
        # Transition class for calibration
        delta_l = abs(lf - li)
        t_class = "dipole" if delta_l == 1 else "non_dipole"
        
        # Filter energies below threshold (regenerate log-grid if needed)
        energies, log_regenerated = build_energy_grid_above_threshold(
            energy_grid,
            threshold_eV,
            energy_type=config.energy.type,
            log_end_eV=config.energy.end_eV,
            log_density=config.energy.density,
            threshold_margin_eV=0.5,
        )
        if not energies:
            print_error("All energies are below threshold!")
            return
        if log_regenerated:
            logger.info(
                "Energy Grid     | LOG regenerated above threshold: %d points (%.1f - %.1f eV)",
                len(energies), energies[0], energies[-1]
            )

        # --- Initialize calibrator ---
        alpha = 1.0
        tong_model = TongModel(threshold_eV, E_f_au, transition_class=t_class)
        
        if config.output.calibrate:
            # Use shared helper function for pilot calibration
            alpha, tong_model, pilot_res = run_pilot_calibration(
                spec=spec,
                core_params=core_params,
                params=params,
                use_pol=use_pol,
                threshold_eV=threshold_eV,
                epsilon_exc_au=E_f_au,
                transition_class=t_class,
                n_theta=params['excitation'].get('pilot_n_theta', 50)
            )

        # --- Build metadata (same as interactive) ---
        n_theta = params['excitation']['n_theta']
        run_meta = {
            "model": "static+polarization" if use_pol else "static",
            "use_polarization": use_pol,
            "grid": {
                "r_min": float(prep.grid.r[0]),
                "r_max": float(prep.grid.r[-1]),
                "n_points": len(prep.grid.r),
            },
            "numerics": {
                "L_max_integrals": spec.L_max_integrals,
                "L_max_projectile_base": spec.L_max_projectile,
                "n_theta": n_theta,
            },
            "calibration": {
                "alpha": alpha,
                "transition_class": t_class,
                "pilot_energy_eV": params['excitation']['pilot_energy_eV'],
                "threshold_eV": threshold_eV,
            },
        }

        # --- Build result key (same format as interactive) ---
        key = f"{config.target.atom}_{ni}{['s','p','d','f','g','h'][li]}-{nf}{['s','p','d','f','g','h'][lf]}"
        filename = f"results_{run_name}_exc.json"

        # --- Main calculation loop (same as interactive) ---
        print_subheader(f"Calculation: {len(energies)} points")
        print("  " + "-" * 45)
        print(f"  {'Energy':>10}  │  {'Cross Section':>15}")
        print("  " + "-" * 45)
        
        # Reset scan-level logging for clean output
        reset_scan_logging()
        
        # Validate high-energy points upfront
        E_max = max(energies)
        he_warnings = validate_high_energy(E_max, params['excitation']['L_max_projectile'], r_max_calc, n_points_calc)
        for warn in he_warnings:
            logger.warning(warn)
        
        results = []
        try:
            for i_e, E in enumerate(energies):
                try:
                    # LOCAL ADAPTIVE: Re-calculate target if needed
                    current_prep = prep
                    if strategy == 'local':
                        # Use same L_eff logic as interactive mode
                        L_eff_local = estimate_effective_projectile_lmax(
                            E, params['excitation']['L_max_projectile']
                        )
                        
                        r_local, n_local = calculate_optimal_grid_params(
                            E, L_eff_local,
                            base_r_max, base_n_points, scale_factor, n_points_max, min_pts_per_wl, z_ion
                        )
                        
                        # For first iteration, ALWAYS recalculate to ensure correct size
                        # (initial prep may have been made with different E_min before filtering)
                        if i_e == 0:
                            current_prep = prepare_target(
                                chan=spec,
                                core_params=core_params,
                                use_polarization=use_pol,
                                r_max=r_local,
                                n_points=n_local
                            )
                            logger.info("Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d (initial)", 
                                       E, r_local, n_local)
                            prep = current_prep
                        else:
                            # Check if parameters changed from current prep
                            current_r_max = prep.grid.r[-1]
                            current_n_pts = len(prep.grid.r)
                            params_changed = (abs(r_local - current_r_max) > 0.1 or 
                                             abs(n_local - current_n_pts) > 1)
                            
                            if params_changed:
                                current_prep = prepare_target(
                                    chan=spec,
                                    core_params=core_params,
                                    use_polarization=use_pol,
                                    r_max=r_local,
                                    n_points=n_local
                                )
                                logger.info("Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d", 
                                           E, r_local, n_local)
                                prep = current_prep  # Update reference for next comparison

                    # === DEBUG: Log prep sizes before calculation ===
                    logger.debug(
                        "BEFORE calc | E=%.2f | prep.grid=%d | orb_i.u=%d, orb_f.u=%d | V_core=%d",
                        E, len(current_prep.grid.r), 
                        len(current_prep.orb_i.u_of_r), len(current_prep.orb_f.u_of_r),
                        len(current_prep.V_core)
                    )
                    
                    res = compute_excitation_cs_precalc(E, current_prep, n_theta=n_theta)
                    
                    # Calibration factor
                    try:
                        tong_sigma = tong_model.calculate_sigma_cm2(E)
                        cal_factor = 1.0
                        if res.sigma_total_cm2 > 0 and tong_model.is_calibrated:
                            cal_factor = tong_sigma / res.sigma_total_cm2
                    except Exception:
                        tong_sigma = 0.0
                        cal_factor = 1.0
                    
                    print_result(E, res.sigma_total_cm2, extra=f"[C(E)={cal_factor:.3f}]")
                    
                    # DCS data (if available)
                    dcs_raw_au = None
                    dcs_cal_au = None
                    theta_deg = None
                    if res.dcs_au is not None and res.theta_deg is not None:
                        theta_deg = res.theta_deg.tolist() if hasattr(res.theta_deg, "tolist") else list(res.theta_deg)
                        dcs_raw_au = res.dcs_au.tolist() if hasattr(res.dcs_au, "tolist") else list(res.dcs_au)
                        dcs_cal_au = (res.dcs_au * cal_factor).tolist()

                    # Result entry (IDENTICAL format to interactive)
                    runtime_meta = dict(res.metadata or {})
                    results.append({
                        "energy_eV": E,
                        "sigma_au": res.sigma_total_au,
                        "sigma_cm2": res.sigma_total_cm2,
                        "sigma_mtong_cm2": tong_sigma,
                        "calibration_alpha": alpha,
                        "calibration_factor": cal_factor,
                        "theta_deg": theta_deg,
                        "dcs_au_raw": dcs_raw_au,
                        "dcs_au_calibrated": dcs_cal_au,
                        "partial_waves": res.partial_waves,
                        "L_max_integrals_used": runtime_meta.get("L_max_integrals_used"),
                        "L_max_projectile_used": runtime_meta.get("L_max_projectile_used"),
                        "L_dynamic_required": runtime_meta.get("L_dynamic_required"),
                        "maxL_in_result": runtime_meta.get("L_max_projectile_summed"),
                        "n_projectile_partial_waves_summed": runtime_meta.get("n_projectile_partial_waves_summed"),
                        "meta": {
                            **run_meta,
                            "runtime": runtime_meta,
                        }
                    })
                    
                except Exception as e:
                    print_error(f"{E:.2f} eV: {e}")

        except KeyboardInterrupt:
            print()
            print_warning(f"Interrupted - saving {len(results)} points...")
            save_results(filename, {key: results})
            print_success("Partial results saved")
            return

        print("  " + "-" * 45)
        save_results(filename, {key: results})
        print_success(f"Complete: {len(results)} points saved")
        
    elif config.calculation_type == "ionization":
        # Build ionization spec
        ion_spec = IonizationChannelSpec(
            l_i=li,
            n_index_i=ni - li,
            N_equiv=1,
            l_eject_max=params['ionization']['l_eject_max'],
            L_max=params['ionization']['L_max'],
            L_i_total=li,
            L_max_projectile=params['ionization']['L_max_projectile'],
            energy_quadrature=params['ionization'].get('energy_quadrature', 'gauss_legendre')
        )
        
        # Get ionization threshold and reusable target preparation
        from driver import prepare_target
        
        # Pre-calculate target properties
        print("  Pre-calculating static target properties...", end=" ", flush=True)
        use_pol = (config.physics_model == "polarization")
        
        # Grid strategy (aligned with interactive ionization/excitation flows)
        strategy = params['grid'].get('strategy', 'global').lower()
        base_r_max = params['grid'].get('r_max', 200.0)
        base_n_points = int(params['grid'].get('n_points', 4000))
        scale_factor = params['grid'].get('r_max_scale_factor', 2.5)
        n_points_max = params['grid'].get('n_points_max', 10000)
        min_pts_per_wl = params['grid'].get('min_points_per_wavelength', 15)
        z_ion = core_params.Zc - 1.0
        
        # Create a dummy excitation spec for prepare_target
        tmp_chan = ExcitationChannelSpec(
            l_i=li, l_f=li+1 if li < 3 else li, 
            n_index_i=ni-li, n_index_f=ni-li+1,
            N_equiv=1, L_max_integrals=10, 
            L_target_i=li, L_target_f=li+1 if li < 3 else li
        )
        
        # First prep for threshold extraction only (supports r_max='auto').
        prep_threshold = prepare_target(
            chan=tmp_chan,
            core_params=core_params,
            use_polarization=use_pol,
            r_max=resolve_grid_r_max_for_prep(base_r_max),
            n_points=base_n_points
        )
        
        # Get threshold from prep
        E_bind_au = prep_threshold.orb_i.energy_au
        threshold_eV = abs(E_bind_au) / ev_to_au(1.0)
        
        print(f"  Ionization potential: {threshold_eV:.2f} eV")
        
        # Filter energies (regenerate log-grid if needed)
        energies, log_regenerated = build_energy_grid_above_threshold(
            energy_grid,
            threshold_eV,
            energy_type=config.energy.type,
            log_end_eV=config.energy.end_eV,
            log_density=config.energy.density,
            threshold_margin_eV=0.5,
        )
        if not energies:
            print_error("All energies are below ionization threshold!")
            return
        if log_regenerated:
            logger.info(
                "Energy Grid     | LOG regenerated above ionization threshold: %d points (%.1f - %.1f eV)",
                len(energies), energies[0], energies[-1]
            )
        
        print(f"  Grid: {len(energies)} points above threshold")
        
        E_min_scan = float(np.min(energies))
        E_max_scan = float(np.max(energies))
        L_eff_scan = estimate_effective_projectile_lmax(
            E_max_scan, params['ionization']['L_max_projectile']
        )
        k_sampling_ion = estimate_ionization_worst_oscillatory_k(E_max_scan, threshold_eV)
        
        if strategy == "manual":
            if isinstance(base_r_max, str):
                print_error("grid.r_max='auto' is not valid with strategy='manual'.")
                return
            r_max_calc = float(base_r_max)
            n_points_calc = base_n_points
            logger.info(
                "Grid Strategy   | MANUAL (r_max=%.1f a.u., n_points=%d)",
                r_max_calc, n_points_calc
            )
        else:
            r_max_calc, n_points_calc = calculate_optimal_grid_params(
                E_min_scan,
                L_eff_scan,
                base_r_max,
                base_n_points,
                scale_factor,
                n_points_max,
                min_pts_per_wl,
                z_ion,
                k_sampling_au=k_sampling_ion,
            )
            if strategy == "local":
                logger.info("Grid Strategy   | LOCAL (per-energy adaptive)")
                logger.info(
                    "Grid Initial    | E_min=%.1f eV: r_max=%.1f a.u., n_points=%d",
                    E_min_scan, r_max_calc, n_points_calc
                )
            else:
                logger.info(
                    "Grid Strategy   | GLOBAL (E_min=%.1f eV: r_max=%.1f a.u., n_points=%d)",
                    E_min_scan, r_max_calc, n_points_calc
                )
        
        # Main prep for calculations (manual/global fixed baseline).
        prep = prepare_target(
            chan=tmp_chan,
            core_params=core_params,
            use_polarization=use_pol,
            r_max=r_max_calc,
            n_points=n_points_calc
        )
        print("done")
        
        # Build result key (same format as interactive)
        key = f"Ionization_Z{core_params.Zc:.0f}_{config.target.atom}_n{ni}l{li}"
        filename = f"results_{run_name}_ion.json"
        
        # Main calculation loop
        print_subheader(f"Calculation: {len(energies)} points")
        print("  " + "-" * 45)
        print(f"  {'Energy':>10}  │  {'Cross Section':>15}")
        print("  " + "-" * 45)
        
        results = []
        try:
            for i_e, E in enumerate(energies):
                try:
                    current_prep = prep
                    r_eval = r_max_calc
                    n_eval = n_points_calc
                    
                    if strategy == "local":
                        L_eff_local = estimate_effective_projectile_lmax(
                            E, params['ionization']['L_max_projectile']
                        )
                        k_sampling_local = estimate_ionization_worst_oscillatory_k(E, threshold_eV)
                        r_local, n_local = calculate_optimal_grid_params(
                            E,
                            L_eff_local,
                            base_r_max,
                            base_n_points,
                            scale_factor,
                            n_points_max,
                            min_pts_per_wl,
                            z_ion,
                            k_sampling_au=k_sampling_local,
                        )
                        if i_e == 0:
                            current_prep = prepare_target(
                                chan=tmp_chan,
                                core_params=core_params,
                                use_polarization=use_pol,
                                r_max=r_local,
                                n_points=n_local
                            )
                            logger.info(
                                "Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d (initial)",
                                E, r_local, n_local
                            )
                            prep = current_prep
                        else:
                            current_r_max = prep.grid.r[-1]
                            current_n_pts = len(prep.grid.r)
                            params_changed = (
                                abs(r_local - current_r_max) > 0.1 or
                                abs(n_local - current_n_pts) > 1
                            )
                            if params_changed:
                                current_prep = prepare_target(
                                    chan=tmp_chan,
                                    core_params=core_params,
                                    use_polarization=use_pol,
                                    r_max=r_local,
                                    n_points=n_local
                                )
                                logger.info(
                                    "Local Adaptive  | E=%.1f eV: r_max=%.1f a.u., n_points=%d",
                                    E, r_local, n_local
                                )
                                prep = current_prep
                        r_eval = r_local
                        n_eval = n_local
                    
                    ion_res = compute_ionization_cs(
                        E, ion_spec, core_params,
                        r_max=r_eval,
                        n_points=n_eval,
                        use_polarization=use_pol,
                        n_energy_steps=params['ionization']['n_energy_steps'],
                        _precalc_grid=current_prep.grid,
                        _precalc_V_core=current_prep.V_core,
                        _precalc_orb_i=current_prep.orb_i
                    )
                    
                    print_result(E, ion_res.sigma_total_cm2)
                    
                    # Result entry (IDENTICAL format to interactive)
                    results.append({
                        "energy_eV": E,
                        "sigma_au": ion_res.sigma_total_au,
                        "sigma_cm2": ion_res.sigma_total_cm2,
                        "IP_eV": ion_res.IP_eV,
                        "sdcs": ion_res.sdcs_data,
                        "tdcs": ion_res.tdcs_data,
                        "partial_waves": ion_res.partial_waves,
                        "meta": ion_res.metadata or {}
                    })
                    
                except Exception as e:
                    print_error(f"{E:.2f} eV: {e}")
                    logger.exception("Calculation failed at E=%.2f eV", E)

        except KeyboardInterrupt:
            print()
            print_warning(f"Interrupted - saving {len(results)} points...")
            save_results(filename, {key: results})
            print_success("Partial results saved")
            return

        print("  " + "-" * 45)
        save_results(filename, {key: results})
        print_success(f"Complete: {len(results)} points saved")
    
    print(f"\n{'═'*50}")
    print(f"  BATCH COMPLETE: {run_name}")
    print(f"{'═'*50}\n")



def parse_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DWBA Calculation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python DW_main.py                       # Interactive mode
  python DW_main.py -c config.yaml        # Batch mode with config file
  python DW_main.py -c config.yaml -v     # Verbose batch mode
  python DW_main.py --generate-config     # Generate template config
        """
    )
    parser.add_argument("-c", "--config", type=str, metavar="FILE",
                        help="Path to YAML config file for batch mode")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate a template configuration file")
    parser.add_argument("--config-type", choices=["excitation", "ionization"],
                        default="excitation",
                        help="Type of template to generate (default: excitation)")
    parser.add_argument("-o", "--output", type=str, default="config_template.yaml",
                        help="Output path for generated config (default: config_template.yaml)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.generate_config:
        from config_loader import generate_template_config
        generate_template_config(args.output, args.config_type)
        print(f"✓ Template configuration saved to: {args.output}")
        sys.exit(0)
    
    if args.config:
        # Batch mode
        run_from_config(args.config, verbose=args.verbose)
    else:
        # Interactive mode
        main()
