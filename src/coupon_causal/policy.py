# path: src/coupon_causal/policy.py
"""
Policy learning and evaluation for coupon targeting.

Implements:
- Qini curves
- Area Under Uplift Curve (AUUC)
- Top-k uplift evaluation
- Expected incremental revenue under budget constraints
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_qini_curve(
    cate_scores: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Qini curve for uplift evaluation.

    The Qini curve measures cumulative gain when targeting by predicted uplift.
    It plots cumulative treatment effect vs. fraction of population targeted.

    Qini(p) = [Y_t(p) - Y_c(p)] - p * [Y_t(1) - Y_c(1)]

    where:
    - p = fraction of population treated (in descending order of predicted uplift)
    - Y_t(p) = cumulative outcome for treated in top p%
    - Y_c(p) = cumulative outcome for control in top p%

    Args:
        cate_scores: Predicted CATE (uplift) scores
        T: Treatment indicator
        Y: Observed outcomes
        n_bins: Number of bins for the curve

    Returns:
        Tuple of (fractions, qini_values, random_baseline)
    """
    n = len(cate_scores)

    # Sort by predicted uplift (descending)
    sorted_idx = np.argsort(-cate_scores)
    T_sorted = T[sorted_idx]
    Y_sorted = Y[sorted_idx]

    # Compute Qini at different fractions
    fractions = np.linspace(0, 1, n_bins + 1)
    qini_values = np.zeros(len(fractions))
    random_baseline = np.zeros(len(fractions))

    for i, frac in enumerate(fractions):
        if frac == 0:
            qini_values[i] = 0
            random_baseline[i] = 0
            continue

        # Top frac% of population
        n_top = int(frac * n)
        T_top = T_sorted[:n_top]
        Y_top = Y_sorted[:n_top]

        # Separate treated and control
        Y_treated_top = Y_top[T_top == 1]
        Y_control_top = Y_top[T_top == 0]

        # Cumulative outcomes
        cum_treated = Y_treated_top.sum() if len(Y_treated_top) > 0 else 0
        cum_control = Y_control_top.sum() if len(Y_control_top) > 0 else 0

        # Qini value (unnormalized)
        n_treated_top = (T_top == 1).sum()
        n_control_top = (T_top == 0).sum()

        if n_treated_top > 0 and n_control_top > 0:
            qini_values[i] = cum_treated / n_treated_top - cum_control / n_control_top
            qini_values[i] *= n_top  # Scale by number of units
        else:
            qini_values[i] = 0

        # Random baseline (expected under random targeting)
        random_baseline[i] = frac * (Y[T == 1].mean() - Y[T == 0].mean()) * n

    return fractions, qini_values, random_baseline


def compute_auuc(
    cate_scores: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute Area Under Uplift Curve (AUUC).

    AUUC measures the overall quality of uplift predictions.
    Higher is better.

    Args:
        cate_scores: Predicted CATE scores
        T: Treatment indicator
        Y: Observed outcomes
        normalize: Whether to normalize by random baseline

    Returns:
        AUUC value
    """
    fractions, qini_values, random_baseline = compute_qini_curve(cate_scores, T, Y, n_bins=100)

    # Compute area under curve using trapezoidal rule
    auuc = np.trapz(qini_values, fractions)

    if normalize:
        # Compute area under random baseline
        auuc_random = np.trapz(random_baseline, fractions)
        # Normalize: AUUC_norm = (AUUC - AUUC_random) / AUUC_random
        if auuc_random != 0:
            auuc_normalized = (auuc - auuc_random) / abs(auuc_random)
            return auuc_normalized
        else:
            return 0.0

    return auuc


def evaluate_policy_uplift(
    cate_scores: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    budget_fractions: Optional[List[float]] = None,
    cost_per_treatment: float = 1.0,
) -> pd.DataFrame:
    """
    Evaluate targeting policy under different budget constraints.

    Args:
        cate_scores: Predicted CATE scores
        T: Treatment indicator
        Y: Observed outcomes
        budget_fractions: List of budget fractions (0-1) to evaluate
        cost_per_treatment: Cost per coupon/treatment

    Returns:
        DataFrame with policy evaluation metrics
    """
    if budget_fractions is None:
        budget_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = []

    # Sort by predicted uplift
    sorted_idx = np.argsort(-cate_scores)

    for budget_frac in budget_fractions:
        # Select top budget_frac% to treat
        n_to_treat = int(budget_frac * len(cate_scores))
        selected_idx = sorted_idx[:n_to_treat]

        # Predicted uplift in selected group
        predicted_uplift = cate_scores[selected_idx].sum()

        # Observed outcomes for selected group
        T_selected = T[selected_idx]
        Y_selected = Y[selected_idx]

        # Estimated incremental revenue
        # E[Y|treated, selected] - E[Y|control, selected]
        y_treated_selected = Y_selected[T_selected == 1]
        y_control_selected = Y_selected[T_selected == 0]

        if len(y_treated_selected) > 0 and len(y_control_selected) > 0:
            observed_uplift_per_unit = y_treated_selected.mean() - y_control_selected.mean()
            observed_total_uplift = observed_uplift_per_unit * n_to_treat
        else:
            observed_uplift_per_unit = 0.0
            observed_total_uplift = 0.0

        # Cost and ROI
        total_cost = n_to_treat * cost_per_treatment
        roi = (observed_total_uplift - total_cost) / total_cost if total_cost > 0 else 0.0

        results.append(
            {
                "budget_fraction": budget_frac,
                "n_treated": n_to_treat,
                "predicted_total_uplift": predicted_uplift,
                "predicted_avg_uplift": predicted_uplift / n_to_treat if n_to_treat > 0 else 0,
                "observed_uplift_per_unit": observed_uplift_per_unit,
                "observed_total_uplift": observed_total_uplift,
                "total_cost": total_cost,
                "net_benefit": observed_total_uplift - total_cost,
                "roi": roi,
            }
        )

    df_results = pd.DataFrame(results)

    logger.info("Policy evaluation results:")
    logger.info(f"\n{df_results.to_string(index=False)}")

    return df_results


def compare_policies(
    cate_scores: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    original_policy: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compare learned targeting policy vs. baseline policies.

    Policies compared:
    1. Learned policy (target by predicted CATE)
    2. Original policy (actual treatment assignment, if provided)
    3. Random policy

    Args:
        cate_scores: Predicted CATE scores
        T: Treatment indicator (actual assignments)
        Y: Observed outcomes
        original_policy: Original policy scores (if different from T)

    Returns:
        Dictionary of AUUC values for each policy
    """
    results = {}

    # Learned policy
    auuc_learned = compute_auuc(cate_scores, T, Y, normalize=False)
    results["learned_policy"] = auuc_learned
    logger.info(f"Learned policy AUUC: {auuc_learned:.2f}")

    # Original policy (if provided)
    if original_policy is not None:
        auuc_original = compute_auuc(original_policy, T, Y, normalize=False)
        results["original_policy"] = auuc_original
        logger.info(f"Original policy AUUC: {auuc_original:.2f}")
        logger.info(f"Improvement: {((auuc_learned - auuc_original) / auuc_original * 100):.1f}%")

    # Random policy
    random_scores = np.random.randn(len(cate_scores))
    auuc_random = compute_auuc(random_scores, T, Y, normalize=False)
    results["random_policy"] = auuc_random
    logger.info(f"Random policy AUUC: {auuc_random:.2f}")

    return results


def optimal_policy_threshold(
    cate_scores: np.ndarray,
    cost_per_treatment: float = 1.0,
    treatment_capacity: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Determine optimal policy threshold for treatment assignment.

    Simple rule: treat if predicted CATE > cost_per_treatment

    Args:
        cate_scores: Predicted CATE scores
        cost_per_treatment: Cost per treatment unit
        treatment_capacity: Maximum number of units to treat (if constrained)

    Returns:
        Tuple of (optimal_threshold, treatment_indicator)
    """
    if treatment_capacity is not None:
        # Capacity-constrained: treat top-k
        if treatment_capacity >= len(cate_scores):
            threshold = cate_scores.min() - 1
        else:
            threshold = np.partition(cate_scores, -treatment_capacity)[-treatment_capacity]
    else:
        # Cost-based: treat if uplift > cost
        threshold = cost_per_treatment

    # Treatment assignment
    treatment_indicator = (cate_scores >= threshold).astype(int)

    logger.info(f"Optimal threshold: {threshold:.2f}")
    logger.info(f"Fraction treated: {treatment_indicator.mean():.1%}")
    logger.info(f"Expected avg uplift (treated): ${cate_scores[treatment_indicator == 1].mean():.2f}")

    return threshold, treatment_indicator


def segment_uplift_analysis(
    cate_scores: np.ndarray,
    df: pd.DataFrame,
    segment_column: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Analyze predicted uplift by customer segments.

    Args:
        cate_scores: Predicted CATE scores
        df: DataFrame with segment information
        segment_column: Column name for segmentation
        top_k: Number of top segments to return

    Returns:
        DataFrame with uplift by segment
    """
    df_analysis = df.copy()
    df_analysis["predicted_cate"] = cate_scores

    # Group by segment
    segment_stats = (
        df_analysis.groupby(segment_column)["predicted_cate"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )

    segment_stats.columns = [
        "segment",
        "count",
        "avg_predicted_uplift",
        "std_predicted_uplift",
        "min_predicted_uplift",
        "max_predicted_uplift",
    ]

    # Sort by average uplift
    segment_stats = segment_stats.sort_values("avg_predicted_uplift", ascending=False)

    logger.info(f"\nTop {top_k} segments by predicted uplift:")
    logger.info(f"\n{segment_stats.head(top_k).to_string(index=False)}")

    return segment_stats.head(top_k)


def create_targeting_recommendations(
    cate_scores: np.ndarray,
    df: pd.DataFrame,
    threshold: float,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create targeting recommendations for deployment.

    Args:
        cate_scores: Predicted CATE scores
        df: DataFrame with customer information
        threshold: Treatment threshold
        output_path: Optional path to save recommendations

    Returns:
        DataFrame with targeting recommendations
    """
    df_recommendations = df.copy()
    df_recommendations["predicted_uplift"] = cate_scores
    df_recommendations["recommend_treatment"] = (cate_scores >= threshold).astype(int)

    # Sort by predicted uplift
    df_recommendations = df_recommendations.sort_values("predicted_uplift", ascending=False)

    # Keep only relevant columns
    keep_cols = [
        "customer_id",
        "predicted_uplift",
        "recommend_treatment",
    ]
    # Add additional useful columns if they exist
    for col in ["customer_segment", "loyalty_score", "monetary_value"]:
        if col in df_recommendations.columns:
            keep_cols.append(col)

    df_recommendations = df_recommendations[keep_cols]

    if output_path:
        df_recommendations.to_csv(output_path, index=False)
        logger.info(f"Targeting recommendations saved to {output_path}")

    logger.info(f"\nTargeting recommendations summary:")
    logger.info(f"Total customers: {len(df_recommendations):,}")
    logger.info(
        f"Recommended for treatment: {df_recommendations['recommend_treatment'].sum():,} "
        f"({df_recommendations['recommend_treatment'].mean():.1%})"
    )
    logger.info(
        f"Avg predicted uplift (treated): "
        f"${df_recommendations[df_recommendations['recommend_treatment']==1]['predicted_uplift'].mean():.2f}"
    )

    return df_recommendations
