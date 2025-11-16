# path: src/coupon_causal/utils.py
"""
Utility functions for causal impact analysis.
"""

import logging
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml


def setup_logging(level: str = "INFO", log_format: str | None = None) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("coupon_causal")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: sklearn and other libraries will use np.random


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_artifact(obj: Any, path: str, logger: logging.Logger | None = None) -> None:
    """
    Save Python object to disk using joblib.

    Args:
        obj: Object to save
        path: Output file path
        logger: Optional logger instance
    """
    ensure_dir(Path(path).parent)
    joblib.dump(obj, path)
    if logger:
        logger.info(f"Saved artifact to {path}")


def load_artifact(path: str, logger: logging.Logger | None = None) -> Any:
    """
    Load Python object from disk using joblib.

    Args:
        path: Input file path
        logger: Optional logger instance

    Returns:
        Loaded object
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    obj = joblib.load(path)
    if logger:
        logger.info(f"Loaded artifact from {path}")

    return obj


def compute_standardized_mean_difference(
    x_treated: np.ndarray,
    x_control: np.ndarray,
    weights_treated: np.ndarray | None = None,
    weights_control: np.ndarray | None = None,
) -> float:
    """
    Compute standardized mean difference (SMD) for a single feature.

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

    Args:
        x_treated: Feature values for treated group
        x_control: Feature values for control group
        weights_treated: Optional weights for treated group
        weights_control: Optional weights for control group

    Returns:
        Standardized mean difference
    """
    if weights_treated is None:
        mean_treated = np.mean(x_treated)
        var_treated = np.var(x_treated)
    else:
        mean_treated = np.average(x_treated, weights=weights_treated)
        var_treated = np.average((x_treated - mean_treated) ** 2, weights=weights_treated)

    if weights_control is None:
        mean_control = np.mean(x_control)
        var_control = np.var(x_control)
    else:
        mean_control = np.average(x_control, weights=weights_control)
        var_control = np.average((x_control - mean_control) ** 2, weights=weights_control)

    pooled_std = np.sqrt((var_treated + var_control) / 2)

    if pooled_std < 1e-10:
        return 0.0

    smd = (mean_treated - mean_control) / pooled_std

    return abs(smd)


def bootstrap_ci(
    statistic_fn,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
    **statistic_kwargs,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        statistic_fn: Function that computes the statistic
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (0-1)
        random_state: Random seed
        **statistic_kwargs: Arguments to pass to statistic_fn

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute point estimate
    point_estimate = statistic_fn(**statistic_kwargs)

    # Bootstrap
    bootstrap_estimates = []
    for _ in range(n_iterations):
        # This is a simple bootstrap; the statistic_fn should handle resampling internally
        # or we pass a resampling flag
        estimate = statistic_fn(**statistic_kwargs, _bootstrap=True)
        bootstrap_estimates.append(estimate)

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def format_number(value: float, precision: int = 2, prefix: str = "", suffix: str = "") -> str:
    """
    Format number for display with optional prefix/suffix.

    Args:
        value: Number to format
        precision: Decimal places
        prefix: String to prepend (e.g., "$")
        suffix: String to append (e.g., "%")

    Returns:
        Formatted string
    """
    return f"{prefix}{value:,.{precision}f}{suffix}"


def format_ci(
    point: float,
    lower: float,
    upper: float,
    precision: int = 2,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Format point estimate with confidence interval.

    Args:
        point: Point estimate
        lower: Lower bound of CI
        upper: Upper bound of CI
        precision: Decimal places
        prefix: String to prepend
        suffix: String to append

    Returns:
        Formatted string like "$15.30 [12.10, 18.50]"
    """
    point_str = format_number(point, precision, prefix, suffix)
    lower_str = format_number(lower, precision, prefix, suffix)
    upper_str = format_number(upper, precision, prefix, suffix)

    return f"{point_str} [{lower_str}, {upper_str}]"
