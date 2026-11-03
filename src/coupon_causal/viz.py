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

    # Compute error bars (matplotlib rejects negative lengths, which happen if a
    # bootstrap interval does not contain its point estimate; clip and warn)
    errors_lower = estimates - ci_lowers
    errors_upper = ci_uppers - estimates
    if np.nanmin(errors_lower) < 0 or np.nanmin(errors_upper) < 0:
        logger.warning("Some confidence intervals exclude their point estimate; clipping error bars")
        errors_lower = np.clip(errors_lower, 0, None)
        errors_upper = np.clip(errors_upper, 0, None)

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


def plot_targeting_comparison(
    comparison: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    random_baseline: np.ndarray,
    save_path: str | None = None,
) -> Figure:
    """
    Predictive-ML targeting vs causal uplift targeting.

    Left: cumulative doubly-robust gain as a function of the share of
    customers targeted, one line per strategy. Right: incremental spend per
    targeted customer at the comparison budget (doubly-robust estimate, plus
    the true value when the data is synthetic).

    Args:
        comparison: Output of ``policy.evaluate_targeting_strategies``
        curves: Mapping strategy -> (fractions, cumulative_gain)
        random_baseline: Random-targeting baseline for the curves
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    palette = {"uplift_model": "steelblue", "response_model": "coral", "random": "gray"}
    labels = {
        "uplift_model": "Causal uplift model (target by CATE)",
        "response_model": "Predictive response model (target by spend)",
        "random": "Random targeting",
        "oracle (true CATE)": "Oracle (true CATE)",
    }

    for name, (fractions, gain) in curves.items():
        ax1.plot(
            fractions * 100,
            gain,
            linewidth=3,
            color=palette.get(name, None),
            label=labels.get(name, name),
        )
    fractions = next(iter(curves.values()))[0]
    ax1.plot(fractions * 100, random_baseline, "--", color="gray", linewidth=2, label=labels["random"])
    ax1.set_xlabel("% of customers targeted", fontsize=12)
    ax1.set_ylabel("Cumulative incremental spend (doubly-robust, $)", fontsize=12)
    ax1.set_title("Who to target first?", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    budget = comparison["budget_fraction"].iloc[0]
    order = comparison.sort_values("dr_uplift_per_customer", ascending=True)
    y = np.arange(len(order))
    has_truth = "true_uplift_per_customer" in order.columns
    height = 0.38 if has_truth else 0.6
    ax2.barh(
        y + (height / 2 if has_truth else 0),
        order["dr_uplift_per_customer"],
        height=height,
        color=[palette.get(s, "seagreen") for s in order["strategy"]],
        label="Doubly-robust estimate",
    )
    if has_truth:
        ax2.barh(
            y - height / 2,
            order["true_uplift_per_customer"],
            height=height,
            color="black",
            alpha=0.25,
            label="True (synthetic ground truth)",
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([labels.get(s, s) for s in order["strategy"]])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel(f"Incremental spend per targeted customer at {budget:.0%} budget ($)", fontsize=12)
    ax2.set_title("Value of the targeted group", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    return fig


def plot_cate_validation(
    calibration: pd.DataFrame,
    cate_pred: np.ndarray | None = None,
    true_cate: np.ndarray | None = None,
    method_name: str = "CATE",
    save_path: str | None = None,
) -> Figure:
    """
    Held-out validation of CATE predictions.

    Left: calibration by predicted-effect decile (predicted vs doubly-robust
    estimate, plus truth when available). Right (synthetic only): predicted
    vs true per-customer effect.

    Args:
        calibration: Output of ``policy.cate_calibration_by_decile``
        cate_pred: Held-out predictions (for the scatter)
        true_cate: True held-out effects (for the scatter)
        method_name: Label for the CATE method
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    has_truth = true_cate is not None and cate_pred is not None
    fig, axes = plt.subplots(1, 2 if has_truth else 1, figsize=(15 if has_truth else 8, 6))
    ax1 = axes[0] if has_truth else axes

    x = calibration["decile"]
    ax1.plot(x, calibration["predicted_cate"], "o-", color="steelblue", linewidth=2, markersize=8, label=f"Predicted ({method_name})")
    ax1.errorbar(
        x,
        calibration["dr_estimate"],
        yerr=1.96 * calibration["dr_se"],
        fmt="s",
        color="coral",
        capsize=4,
        markersize=7,
        label="Doubly-robust estimate ± 95% CI",
    )
    if "true_cate" in calibration.columns:
        ax1.plot(x, calibration["true_cate"], "^--", color="black", alpha=0.6, markersize=7, label="True effect")
    ax1.set_xlabel("Decile of predicted effect (1 = lowest)", fontsize=12)
    ax1.set_ylabel("Treatment effect ($)", fontsize=12)
    ax1.set_title("Does predicted uplift track real uplift? (held-out)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if has_truth:
        ax2 = axes[1]
        ax2.scatter(true_cate, cate_pred, alpha=0.25, s=8, color="steelblue")
        lo, hi = float(np.min(true_cate)), float(np.max(true_cate))
        ax2.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")
        corr = np.corrcoef(true_cate, cate_pred)[0, 1]
        ax2.text(
            0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax2.transAxes, fontsize=12,
            verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        ax2.set_xlabel("True treatment effect ($)", fontsize=12)
        ax2.set_ylabel("Predicted treatment effect ($)", fontsize=12)
        ax2.set_title("Predicted vs true effect (held-out)", fontsize=14, fontweight="bold")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    return fig


def plot_refutation_summary(
    summary: pd.DataFrame,
    ate_estimate: float,
    save_path: str | None = None,
) -> Figure:
    """
    One row per refutation check: the re-estimated effect against what it
    should be (zero for placebo tests, the original estimate for stability
    tests). Green = passed, red = failed.

    Args:
        summary: ``summary`` table from ``refute.run_all_refutations``
        ate_estimate: The headline estimate (drawn as a reference line)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure
    """
    df_plot = summary.dropna(subset=["new_effect"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.7 * len(df_plot) + 1.5)))
    y = np.arange(len(df_plot))

    ax.axvline(0, color="black", linewidth=1, label="Zero")
    ax.axvline(ate_estimate, color="steelblue", linestyle="--", linewidth=2, label=f"Headline estimate (${ate_estimate:.2f})")
    for i, row in df_plot.iterrows():
        ax.plot([row["expected"], row["new_effect"]], [i, i], color="gray", alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(df_plot["expected"], y, marker="|", s=300, color="black", zorder=2, label="Expected if assumptions hold")
    ax.scatter(
        df_plot["new_effect"],
        y,
        s=120,
        c=["seagreen" if p else "crimson" for p in df_plot["passed"]],
        edgecolor="black",
        zorder=3,
        label="Re-estimated effect",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  [{s}]" for t, s in zip(df_plot["test"], df_plot["source"])])
    ax.invert_yaxis()
    ax.set_xlabel("Estimated effect ($)", fontsize=12)
    ax.set_title("Refutation checks: does the estimate survive?", fontsize=14, fontweight="bold")
    # Legend below the axes so it never covers a data point
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    return fig
