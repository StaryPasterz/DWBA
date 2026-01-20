# logging_config.py
"""
Centralized Logging Configuration for DWBA Calculation Suite
=============================================================

This module provides a unified logging configuration for all modules in the
DWBA codebase. It ensures consistent log formatting, levels, and output
across the entire application.

Usage
-----
In each module, import and call get_logger():

    from logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("Starting calculation...")
    logger.debug("Detailed debug info: %s", value)
    logger.warning("Potential issue detected")
    logger.error("Calculation failed: %s", error)

Log Levels
----------
- DEBUG: Detailed diagnostic info (grid sizes, wave numbers, integral values)
- INFO: High-level progress (starting/completing calculations, results)
- WARNING: Non-critical issues (convergence concerns, fallbacks)
- ERROR: Critical failures that may affect results
- CRITICAL: Fatal errors preventing execution

Configuration
-------------
The log level can be controlled via environment variable:
    set DWBA_LOG_LEVEL=DEBUG

or programmatically:
    logging_config.set_log_level(logging.DEBUG)

File output can be enabled:
    logging_config.enable_file_logging("dwba_run.log")
"""

from __future__ import annotations
import logging
import sys
import os
from typing import Optional
from datetime import datetime

# Module-level logger cache
_loggers: dict = {}

# Default configuration
_DEFAULT_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LEVEL = logging.INFO

# Track if handlers have been configured
_handlers_configured = False
_file_handler: Optional[logging.FileHandler] = None


def _configure_root_handler() -> None:
    """
    Configure the root logger handler once for consistent output.
    Called automatically on first get_logger() call.
    """
    global _handlers_configured
    
    if _handlers_configured:
        return
    
    # Read log level from environment or use default
    env_level = os.environ.get("DWBA_LOG_LEVEL", "").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(env_level, _DEFAULT_LEVEL)
    
    # Create root handler with formatting
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Avoid adding duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
    
    _handlers_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module name.
    
    This is the primary entry point for obtaining loggers throughout the
    DWBA codebase. It ensures consistent configuration and caches loggers
    for efficiency.
    
    Parameters
    ----------
    name : str
        Module name, typically __name__ from the calling module.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
        
    Example
    -------
    >>> from logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Calculation started")
    """
    global _loggers
    
    if name not in _loggers:
        _configure_root_handler()
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


def set_log_level(level: int) -> None:
    """
    Set the global log level for all DWBA loggers.
    
    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
        
    Example
    -------
    >>> import logging
    >>> from logging_config import set_log_level
    >>> set_log_level(logging.DEBUG)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)


def enable_file_logging(
    filename: Optional[str] = None,
    level: int = logging.DEBUG
):
    """
    Enable logging to a file in addition to console output.
    
    File logs are typically set to DEBUG level for detailed diagnostics
    while console remains at INFO for cleaner output.
    
    Parameters
    ----------
    filename : str, optional
        Path to log file. If not specified, generates timestamped filename.
    level : int
        Log level for file output (default: DEBUG).
        
    Returns
    -------
    str
        Path to the created log file.
        
    Example
    -------
    >>> from logging_config import enable_file_logging
    >>> log_file = enable_file_logging("debug_run.log")
    >>> print(f"Logging to: {log_file}")
    """
    global _file_handler
    
    # Generate default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dwba_log_{timestamp}.log"
    
    # Remove existing file handler if present
    if _file_handler is not None:
        root_logger = logging.getLogger()
        root_logger.removeHandler(_file_handler)
        _file_handler.close()
    
    # Create new file handler
    _file_handler = logging.FileHandler(filename, encoding='utf-8')
    _file_handler.setLevel(level)
    
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    _file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(_file_handler)
    
    # Ensure root level is low enough
    if root_logger.level > level:
        root_logger.setLevel(level)
    
    return filename


def disable_file_logging() -> None:
    """
    Disable file logging if it was previously enabled.
    """
    global _file_handler
    
    if _file_handler is not None:
        root_logger = logging.getLogger()
        root_logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None


def silence_logger(name: str) -> None:
    """
    Silence a specific logger (set to WARNING level only).
    
    Useful for quieting verbose third-party libraries.
    
    Parameters
    ----------
    name : str
        Logger name to silence.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)


# Convenience function for quick debug mode
def enable_debug_mode() -> None:
    """
    Enable full debug output to console.
    
    Shortcut for development and troubleshooting.
    """
    set_log_level(logging.DEBUG)


# Silence some noisy libraries by default
def _configure_third_party() -> None:
    """Configure logging for third-party libraries."""
    # Silence matplotlib font manager messages
    silence_logger("matplotlib.font_manager")
    silence_logger("matplotlib")
    # Reduce scipy verbosity
    silence_logger("scipy")


# Auto-configure on import
_configure_third_party()
