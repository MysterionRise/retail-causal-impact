# path: src/coupon_causal/viz.py
"""
Visualization utilities for causal impact analysis.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set default style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def plot_treatment_distribution(
    T: np.ndarray,
    Y: np.ndarray,
    save_path: str | None = None,
) -> Figure:
    """
    Plot treatment distribution and outcome by treatment group.

    Args:
        T: Treatment indicator
        Y: Outcome vector
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Treatment distribution
    treatment_counts = np.bincount(T.astype(int))
    axes[0].bar([0, 1], treatment_counts, color=["steelblue", "coral"])
    axes[0].set_xlabel("Treatment", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Treatment Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Control", "Treated"])

    # Add counts on bars
    for i, count in enumerate(treatment_counts):
        axes[0].text(i, count + max(treatment_counts) * 0.02, f"{count:,}",
                    ha="center", va="bottom", fontsize=10)

    # Outcome distribution by treatment
    axes[1].hist(Y[T == 0], bins=30, alpha=0.6, label="Control", color="steelblue")
    axes[1].hist(Y[T == 1], bins=30, alpha=0.6, label="Treated", color="coral")
    axes[1].set_xlabel("Outcome", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Outcome Distribution by Treatment", fontsize=14, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_propensity_distribution(
    propensity_scores: np.ndarray,
    T: np.ndarray,
    save_path: str | None = None,
) -> Figure:
    """
    Plot propensity score distributions for treated and control groups.

    Args:
        propensity_scores: Propensity scores
        T: Treatment indicator
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overlapping histograms
    axes[0].hist(
        propensity_scores[T == 0], bins=30, alpha=0.6, label="Control", color="steelblue"
    )
    axes[0].hist(
        propensity_scores[T == 1], bins=30, alpha=0.6, label="Treated", color="coral"
    )
    axes[0].set_xlabel("Propensity Score", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Propensity Score Distribution", fontsize=14, fontweight="bold")
    axes[0].legend()

    # Density plot with common support region
    axes[1].hist(
        propensity_scores[T == 0],
        bins=50,
        alpha=0.5,
        density=True,
        label="Control",
        color="steelblue",
    )
    axes[1].hist(
        propensity_scores[T == 1],
        bins=50,
        alpha=0.5,
        density=True,
        label="Treated",
        color="coral",
    )

    # Mark common support region
    min_common = max(propensity_scores[T == 0].min(), propensity_scores[T == 1].min())
    max_common = min(propensity_scores[T == 0].max(), propensity_scores[T == 1].max())
    axes[1].axvline(min_common, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axvline(max_common, color="red", linestyle="--", linewidth=1, alpha=0.7)

    axes[1].set_xlabel("Propensity Score", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Propensity Score Overlap", fontsize=14, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_love_plot(
    balance_before: pd.DataFrame,
    balance_after: pd.DataFrame,
    top_k: int = 15,
    smd_threshold: float = 0.1,
    save_path: str | None = None,
) -> Figure:
    """
    Create Love plot showing balance before and after weighting.

    Args:
        balance_before: Balance table before weighting
        balance_after: Balance table after weighting
        top_k: Number of top features to show
        smd_threshold: Threshold line for acceptable balance
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    # Merge tables
    # Note: balance_before and balance_after already have smd_before and smd_after columns
    balance_df = pd.merge(
        balance_before[["feature", "smd_before"]],
        balance_after[["feature", "smd_after"]],
        on="feature",
    )

    # Sort by SMD before (worst first)
    balance_df = balance_df.sort_values("smd_before", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))

    y_pos = np.arange(len(balance_df))

    # Plot points
    ax.scatter(
        balance_df["smd_before"],
        y_pos,
        s=100,
        color="coral",
        label="Before weighting",
        zorder=3,
        marker="o",
    )
    ax.scatter(
        balance_df["smd_after"],
        y_pos,
        s=100,
        color="steelblue",
        label="After weighting",
        zorder=3,
        marker="s",
    )

    # Connect with lines
    for i in range(len(balance_df)):
        ax.plot(
            [balance_df.iloc[i]["smd_before"], balance_df.iloc[i]["smd_after"]],
            [i, i],
            color="gray",
            alpha=0.3,
            linewidth=1,
            zorder=1,
        )

    # Threshold line
    ax.axvline(smd_threshold, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"SMD = {smd_threshold}")
    ax.axvline(-smd_threshold, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df["feature"])
    ax.set_xlabel("Standardized Mean Difference (SMD)", fontsize=12)
    ax.set_title(
        "Covariate Balance Before and After Weighting (Love Plot)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_ate_comparison(
    ate_results: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> Figure:
    """
    Plot comparison of ATE estimates from different methods.

    Args:
        ate_results: Dictionary of ATE results from estimate_all_ate_methods
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    methods = []
    estimates = []
    ci_lowers = []
    ci_uppers = []

    for method_name, results in ate_results.items():
        methods.append(method_name.replace("_", " ").title())
        estimates.append(results["ate"])
        ci_lowers.append(results["ci_lower"])
        ci_uppers.append(results["ci_upper"])

    # Convert to arrays
    estimates = np.array(estimates)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Compute error bars
    errors_lower = estimates - ci_lowers
    errors_upper = ci_uppers - estimates

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(methods))

    # Plot error bars
    ax.errorbar(
        estimates,
        y_pos,
        xerr=[errors_lower, errors_upper],
        fmt="o",
        markersize=10,
        capsize=5,
        capthick=2,
        linewidth=2,
        color="steelblue",
    )

    # Zero line
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Average Treatment Effect (ATE)", fontsize=12)
    ax.set_title("ATE Estimates with 95% Confidence Intervals", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_cate_distribution(
    cate_scores: np.ndarray,
    method_name: str = "CATE",
    save_path: str | None = None,
) -> Figure:
    """
    Plot CATE distribution.

    Args:
        cate_scores: CATE predictions
        method_name: Name of CATE method
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(cate_scores, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(
        cate_scores.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: ${cate_scores.mean():.2f}"
    )
    axes[0].axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("Predicted CATE", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"{method_name} Distribution", fontsize=14, fontweight="bold")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(cate_scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("Predicted CATE", fontsize=12)
    axes[1].set_title(f"{method_name} Summary", fontsize=14, fontweight="bold")
    axes[1].set_xticklabels([method_name])

    # Add stats
    stats_text = (
        f"Mean: ${cate_scores.mean():.2f}\n"
        f"Median: ${np.median(cate_scores):.2f}\n"
        f"Std: ${cate_scores.std():.2f}\n"
        f"Range: [${cate_scores.min():.2f}, ${cate_scores.max():.2f}]"
    )
    axes[1].text(
        0.05, 0.95, stats_text,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_qini_curve(
    fractions: np.ndarray,
    qini_values: np.ndarray,
    random_baseline: np.ndarray,
    save_path: str | None = None,
) -> Figure:
    """
    Plot Qini curve for uplift evaluation.

    Args:
        fractions: Fractions of population targeted
        qini_values: Qini curve values
        random_baseline: Random targeting baseline
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Qini curve
    ax.plot(
        fractions * 100,
        qini_values,
        linewidth=3,
        color="steelblue",
        label="Learned Policy",
    )

    # Plot random baseline
    ax.plot(
        fractions * 100,
        random_baseline,
        linewidth=2,
        color="gray",
        linestyle="--",
        label="Random Policy",
    )

    # Fill area between
    ax.fill_between(
        fractions * 100,
        qini_values,
        random_baseline,
        alpha=0.2,
        color="green",
        label="Uplift Gain",
    )

    ax.set_xlabel("% of Population Targeted", fontsize=12)
    ax.set_ylabel("Cumulative Uplift", fontsize=12)
    ax.set_title("Qini Curve - Uplift Model Performance", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    save_path: str | None = None,
) -> Figure:
    """
    Plot feature importance for CATE model.

    Args:
        feature_importances: Feature importance values
        feature_names: Feature names
        top_k: Number of top features to show
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values("importance", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))

    y_pos = np.arange(len(importance_df))

    ax.barh(y_pos, importance_df["importance"], color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df["feature"])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_k} Features for CATE Prediction", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_segment_uplift(
    segment_stats: pd.DataFrame,
    segment_col: str = "segment",
    uplift_col: str = "avg_predicted_uplift",
    save_path: str | None = None,
) -> Figure:
    """
    Plot average uplift by customer segment.

    Args:
        segment_stats: DataFrame with segment statistics
        segment_col: Name of segment column
        uplift_col: Name of uplift column
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    segments = segment_stats[segment_col].values
    uplifts = segment_stats[uplift_col].values

    colors = ["coral" if u > 0 else "steelblue" for u in uplifts]

    y_pos = np.arange(len(segments))

    ax.barh(y_pos, uplifts, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(segments)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Average Predicted Uplift ($)", fontsize=12)
    ax.set_title("Predicted Uplift by Customer Segment", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add values on bars
    for i, uplift in enumerate(uplifts):
        ax.text(
            uplift + max(abs(uplifts)) * 0.02,
            i,
            f"${uplift:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
) -> Figure:
    """
    Plot calibration curve for propensity model.

    Args:
        y_true: True treatment labels
        y_pred: Predicted propensity scores
        n_bins: Number of calibration bins
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, markersize=8, color="steelblue", label="Model")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("True Probability", fontsize=12)
    ax.set_title("Propensity Model Calibration", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig
