# path: src/coupon_causal/balance.py
"""
Balance diagnostics for causal inference.

Assess covariate balance between treatment and control groups,
before and after propensity score weighting.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import compute_standardized_mean_difference

logger = logging.getLogger(__name__)


def compute_balance_table(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: List[str],
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute balance table with standardized mean differences.

    Args:
        X: Feature matrix (n_samples, n_features)
        T: Treatment indicator
        feature_names: List of feature names
        weights: Optional weights (e.g., IPW weights)

    Returns:
        DataFrame with balance statistics
    """
    balance_stats = []

    for i, feature_name in enumerate(feature_names):
        x = X[:, i]
        x_treated = x[T == 1]
        x_control = x[T == 0]

        if weights is not None:
            w_treated = weights[T == 1]
            w_control = weights[T == 0]
        else:
            w_treated = None
            w_control = None

        # Compute SMD
        smd = compute_standardized_mean_difference(x_treated, x_control, w_treated, w_control)

        # Compute means
        if w_treated is not None:
            mean_treated = np.average(x_treated, weights=w_treated)
            mean_control = np.average(x_control, weights=w_control)
        else:
            mean_treated = np.mean(x_treated)
            mean_control = np.mean(x_control)

        balance_stats.append(
            {
                "feature": feature_name,
                "mean_treated": mean_treated,
                "mean_control": mean_control,
                "diff": mean_treated - mean_control,
                "smd": smd,
            }
        )

    df_balance = pd.DataFrame(balance_stats)
    df_balance = df_balance.sort_values("smd", ascending=False)

    return df_balance


def compare_balance_before_after(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: List[str],
    weights: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compare covariate balance before and after weighting.

    Args:
        X: Feature matrix
        T: Treatment indicator
        feature_names: List of feature names
        weights: IPW weights

    Returns:
        Tuple of (balance_before, balance_after, summary_stats)
    """
    logger.info("Computing balance before and after weighting...")

    # Balance before weighting
    balance_before = compute_balance_table(X, T, feature_names, weights=None)
    balance_before = balance_before.rename(columns={"smd": "smd_before"})

    # Balance after weighting
    balance_after = compute_balance_table(X, T, feature_names, weights=weights)
    balance_after = balance_after.rename(columns={"smd": "smd_after"})

    # Merge
    balance_comparison = pd.merge(
        balance_before[["feature", "smd_before"]],
        balance_after[["feature", "smd_after"]],
        on="feature",
    )

    balance_comparison["smd_reduction"] = (
        balance_comparison["smd_before"] - balance_comparison["smd_after"]
    )

    # Summary statistics
    summary = {
        "mean_smd_before": balance_before["smd_before"].mean(),
        "median_smd_before": balance_before["smd_before"].median(),
        "max_smd_before": balance_before["smd_before"].max(),
        "mean_smd_after": balance_after["smd_after"].mean(),
        "median_smd_after": balance_after["smd_after"].median(),
        "max_smd_after": balance_after["smd_after"].max(),
        "fraction_below_0.1_before": (balance_before["smd_before"] < 0.1).mean(),
        "fraction_below_0.1_after": (balance_after["smd_after"] < 0.1).mean(),
    }

    logger.info("\nBalance Summary:")
    logger.info(f"  Before weighting - Mean SMD: {summary['mean_smd_before']:.3f}")
    logger.info(f"  After weighting  - Mean SMD: {summary['mean_smd_after']:.3f}")
    logger.info(f"  Fraction with SMD < 0.1 (before): {summary['fraction_below_0.1_before']:.1%}")
    logger.info(f"  Fraction with SMD < 0.1 (after):  {summary['fraction_below_0.1_after']:.1%}")

    return balance_before, balance_after, summary


def assess_balance_quality(
    balance_df: pd.DataFrame,
    smd_threshold: float = 0.1,
) -> Dict[str, any]:
    """
    Assess overall balance quality.

    Common rule of thumb: SMD < 0.1 indicates good balance.

    Args:
        balance_df: DataFrame with balance statistics (must have 'smd' column)
        smd_threshold: Threshold for acceptable balance

    Returns:
        Dictionary of balance quality metrics
    """
    smd_col = "smd" if "smd" in balance_df.columns else "smd_after"

    if smd_col not in balance_df.columns:
        raise ValueError(f"Balance DataFrame must have '{smd_col}' column")

    smds = balance_df[smd_col].values

    quality_metrics = {
        "mean_smd": smds.mean(),
        "median_smd": np.median(smds),
        "max_smd": smds.max(),
        "fraction_below_threshold": (smds < smd_threshold).mean(),
        "n_above_threshold": (smds >= smd_threshold).sum(),
        "problematic_features": balance_df[balance_df[smd_col] >= smd_threshold][
            "feature"
        ].tolist(),
    }

    # Overall assessment
    if quality_metrics["fraction_below_threshold"] >= 0.9:
        quality_metrics["assessment"] = "Good"
    elif quality_metrics["fraction_below_threshold"] >= 0.7:
        quality_metrics["assessment"] = "Acceptable"
    else:
        quality_metrics["assessment"] = "Poor"

    logger.info(f"\nBalance Quality Assessment: {quality_metrics['assessment']}")
    if quality_metrics["problematic_features"]:
        logger.warning(
            f"Features with poor balance (SMD >= {smd_threshold}): "
            f"{quality_metrics['problematic_features'][:5]}"
        )

    return quality_metrics


def compute_variance_ratio(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: List[str],
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute variance ratios between treatment and control groups.

    Variance ratio = Var(X|T=1) / Var(X|T=0)

    Good balance: variance ratio close to 1 (typically between 0.5 and 2.0)

    Args:
        X: Feature matrix
        T: Treatment indicator
        feature_names: List of feature names
        weights: Optional weights

    Returns:
        DataFrame with variance ratios
    """
    variance_ratios = []

    for i, feature_name in enumerate(feature_names):
        x = X[:, i]
        x_treated = x[T == 1]
        x_control = x[T == 0]

        if weights is not None:
            w_treated = weights[T == 1]
            w_control = weights[T == 0]

            var_treated = np.average(
                (x_treated - np.average(x_treated, weights=w_treated)) ** 2, weights=w_treated
            )
            var_control = np.average(
                (x_control - np.average(x_control, weights=w_control)) ** 2, weights=w_control
            )
        else:
            var_treated = np.var(x_treated)
            var_control = np.var(x_control)

        if var_control > 0:
            var_ratio = var_treated / var_control
        else:
            var_ratio = np.nan

        variance_ratios.append(
            {
                "feature": feature_name,
                "var_treated": var_treated,
                "var_control": var_control,
                "variance_ratio": var_ratio,
            }
        )

    df_var_ratio = pd.DataFrame(variance_ratios)

    return df_var_ratio


def check_positivity(
    propensity_scores: np.ndarray,
    T: np.ndarray,
    lower_threshold: float = 0.01,
    upper_threshold: float = 0.99,
) -> Dict[str, any]:
    """
    Check positivity assumption (overlap in propensity scores).

    Positivity requires that all units have some probability of receiving
    either treatment or control: 0 < P(T=1|X) < 1

    Args:
        propensity_scores: Propensity scores
        T: Treatment indicator
        lower_threshold: Lower bound for acceptable propensity scores
        upper_threshold: Upper bound for acceptable propensity scores

    Returns:
        Dictionary of positivity diagnostics
    """
    p_treated = propensity_scores[T == 1]
    p_control = propensity_scores[T == 0]

    # Check violations
    n_treated_low = (p_treated < lower_threshold).sum()
    n_treated_high = (p_treated > upper_threshold).sum()
    n_control_low = (p_control < lower_threshold).sum()
    n_control_high = (p_control > upper_threshold).sum()

    total_violations = n_treated_low + n_treated_high + n_control_low + n_control_high

    diagnostics = {
        "n_treated_low_propensity": n_treated_low,
        "n_treated_high_propensity": n_treated_high,
        "n_control_low_propensity": n_control_low,
        "n_control_high_propensity": n_control_high,
        "total_violations": total_violations,
        "violation_rate": total_violations / len(propensity_scores),
        "propensity_range_treated": (p_treated.min(), p_treated.max()),
        "propensity_range_control": (p_control.min(), p_control.max()),
    }

    if diagnostics["violation_rate"] > 0.05:
        logger.warning(
            f"Positivity violations detected: {total_violations} units "
            f"({diagnostics['violation_rate']:.1%})"
        )
    else:
        logger.info(f"Positivity check passed: {diagnostics['violation_rate']:.1%} violations")

    return diagnostics
