# output_utils.py
"""
Output Path Management for DWBA Calculations
=============================================

Centralizes all output file handling (results JSON, plots) into the `results/`
directory to keep the repository organized and clean.

Usage
-----
    from output_utils import get_results_dir, get_output_path, get_json_path, get_plot_path
    
    # Get path for a JSON results file
    json_path = get_json_path("H2s", "exc")  # -> results/results_H2s_exc.json
    
    # Get path for a plot file
    plot_path = get_plot_path("plot_H2s_exc_std.png")  # -> results/plot_H2s_exc_std.png
    
    # General file in results directory
    file_path = get_output_path("custom_file.dat")  # -> results/custom_file.dat

Notes
-----
- The `results/` directory is created automatically if it doesn't exist.
- All functions return `pathlib.Path` objects for modern path handling.
- Functions accept both string and Path inputs for flexibility.
"""

import json
from pathlib import Path
from typing import Union, List, Tuple

from logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Default results directory name
RESULTS_DIR = "results"


def get_results_dir() -> Path:
    """
    Get the results directory path, creating it if it doesn't exist.
    
    Returns
    -------
    Path
        Absolute path to the results directory.
        
    Examples
    --------
    >>> results = get_results_dir()
    >>> print(results)
    results
    """
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(exist_ok=True)
    return results_path


def get_output_path(filename: Union[str, Path]) -> Path:
    """
    Get full path for an output file in the results directory.
    
    Parameters
    ----------
    filename : str or Path
        Name of the file (without directory prefix).
        
    Returns
    -------
    Path
        Full path to the file in the results directory.
        
    Examples
    --------
    >>> path = get_output_path("custom_data.json")
    >>> print(path)
    results/custom_data.json
    """
    return get_results_dir() / Path(filename).name


def get_json_path(run_name: str, calc_type: str) -> Path:
    """
    Get path for a results JSON file.
    
    Parameters
    ----------
    run_name : str
        Name of the calculation run (e.g., "H2s", "He+2p").
    calc_type : str
        Type of calculation: "exc" for excitation, "ion" for ionization.
        
    Returns
    -------
    Path
        Full path to the JSON file.
        
    Examples
    --------
    >>> path = get_json_path("H2s", "exc")
    >>> print(path)
    results/results_H2s_exc.json
    """
    filename = f"results_{run_name}_{calc_type}.json"
    return get_output_path(filename)


def get_plot_path(plot_name: str) -> Path:
    """
    Get path for a plot file.
    
    Parameters
    ----------
    plot_name : str
        Name of the plot file (with extension).
        
    Returns
    -------
    Path
        Full path to the plot file in results directory.
        
    Examples
    --------
    >>> path = get_plot_path("plot_H2s_exc_std.png")
    >>> print(path)
    results/plot_H2s_exc_std.png
    """
    return get_output_path(plot_name)


def find_result_files(pattern: str = "results_*.json") -> List[Path]:
    """
    Find result files matching a pattern in the results directory.
    
    Also checks the root directory for backward compatibility with
    existing files that haven't been migrated.
    
    Parameters
    ----------
    pattern : str
        Glob pattern to match files (default: "results_*.json").
        
    Returns
    -------
    list of Path
        List of matching file paths, sorted by name.
        
    Examples
    --------
    >>> files = find_result_files()
    >>> for f in files:
    ...     print(f.name)
    results_H2s_exc.json
    results_He+2p_exc.json
    """
    results_dir = get_results_dir()
    
    # Find in results/ directory
    files = list(results_dir.glob(pattern))
    
    # Also check root for backward compatibility
    root_files = list(Path(".").glob(pattern))
    
    # Combine and deduplicate by filename
    seen_names = {f.name for f in files}
    for rf in root_files:
        if rf.name not in seen_names:
            files.append(rf)
    
    return sorted(files, key=lambda p: p.name)


def load_results(filename: Union[str, Path]) -> dict:
    """
    Load existing results JSON.

    Checks `results/` first, then falls back to repository root for
    backward compatibility. Returns empty dict on missing/corrupt file.
    """
    base_name = Path(filename).name

    results_path = get_results_dir() / base_name
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    root_path = Path(base_name)
    if root_path.exists():
        try:
            with open(root_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}


def save_results(filename: Union[str, Path], new_data_dict: dict) -> Path:
    """
    Merge and save results JSON in `results/`.

    Existing file content is loaded via `load_results()` and updated by key.
    Returns path to saved file.
    """
    base_name = Path(filename).name
    output_path = get_output_path(base_name)

    current = load_results(base_name)
    current.update(new_data_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)

    logger.info("Results saved to %s", output_path)
    return output_path


def migrate_existing_files(dry_run: bool = True) -> List[Tuple[Path, Path]]:
    """
    Find and optionally migrate existing output files from root to results/.
    
    Parameters
    ----------
    dry_run : bool
        If True, only report what would be moved without actually moving.
        
    Returns
    -------
    list of tuple
        List of (source, destination) path pairs.
        
    Examples
    --------
    >>> # See what would be migrated
    >>> to_migrate = migrate_existing_files(dry_run=True)
    >>> for src, dst in to_migrate:
    ...     print(f"{src} -> {dst}")
    """
    import shutil
    
    results_dir = get_results_dir()
    migrations = []
    
    # Patterns for output files
    patterns = ["results_*.json", "plot_*.png", "fit_*.png"]
    
    for pattern in patterns:
        for src in Path(".").glob(pattern):
            if src.parent == results_dir:
                continue  # Already in results/
            dst = results_dir / src.name
            migrations.append((src, dst))
            
            if not dry_run:
                logger.info("Migrating %s -> %s", src, dst)
                shutil.move(str(src), str(dst))
            else:
                logger.debug("Would migrate: %s -> %s", src, dst)
    
    if migrations:
        if dry_run:
            logger.info("Found %d files to migrate (dry_run=True)", len(migrations))
        else:
            logger.info("Migrated %d files to results/", len(migrations))
    
    return migrations
