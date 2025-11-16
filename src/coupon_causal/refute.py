# path: src/coupon_causal/refute.py
"""
Refutation and sensitivity analysis using DoWhy.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_causal_graph() -> str:
    """
    Create causal graph (DAG) for coupon campaign.

    Returns:
        DoWhy-compatible graph string
    """
    graph = """
    digraph {
        // Confounders affect both treatment and outcome
        loyalty_score -> treatment;
        loyalty_score -> outcome;
        recency_days -> treatment;
        recency_days -> outcome;
        monetary_value -> treatment;
        monetary_value -> outcome;
        customer_segment -> treatment;
        customer_segment -> outcome;

        // Other features
        frequency_purchases -> treatment;
        frequency_purchases -> outcome;
        price_sensitivity -> outcome;
        avg_basket_size -> outcome;

        // Treatment causes outcome
        treatment -> outcome;
    }
    """
    return graph


def run_dowhy_analysis(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    common_causes: list[str] | None = None,
) -> dict:
    """
    Run DoWhy causal analysis with refutations.

    Args:
        df: DataFrame with treatment, outcome, and covariates
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        common_causes: List of common cause (confounder) names

    Returns:
        Dictionary with causal model and refutation results
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        logger.error("DoWhy not installed. Install with: pip install dowhy")
        return {}

    logger.info("Running DoWhy causal analysis...")

    if common_causes is None:
        # Default confounders for coupon campaign
        common_causes = [
            "loyalty_score",
            "recency_days",
            "monetary_value",
            "frequency_purchases",
        ]

        # Only include features that exist in the DataFrame
        common_causes = [col for col in common_causes if col in df.columns]

    # Create causal model
    model = CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=common_causes,
        graph=create_causal_graph(),
    )

    logger.info("Causal model created")

    # Identify causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    logger.info(f"Identified estimand:\n{identified_estimand}")

    # Estimate causal effect
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
    )

    logger.info(f"Causal estimate: {estimate.value:.2f}")

    results = {
        "model": model,
        "identified_estimand": identified_estimand,
        "estimate": estimate,
        "refutations": {},
    }

    # Run refutations
    logger.info("\nRunning refutation tests...")

    # 1. Random common cause refuter
    # Adds a random variable as confounder - should not change estimate
    try:
        logger.info("[1/3] Random common cause refuter...")
        refute_random = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=50,
        )
        results["refutations"]["random_common_cause"] = refute_random
        logger.info(f"Random common cause: {refute_random}")
    except Exception as e:
        logger.warning(f"Random common cause refutation failed: {e}")

    # 2. Placebo treatment refuter
    # Replaces treatment with random - should get estimate close to 0
    try:
        logger.info("[2/3] Placebo treatment refuter...")
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            num_simulations=50,
        )
        results["refutations"]["placebo_treatment"] = refute_placebo
        logger.info(f"Placebo treatment: {refute_placebo}")
    except Exception as e:
        logger.warning(f"Placebo treatment refutation failed: {e}")

    # 3. Data subset refuter
    # Estimates on subsets of data - should be stable
    try:
        logger.info("[3/3] Data subset refuter...")
        refute_subset = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=20,
        )
        results["refutations"]["data_subset"] = refute_subset
        logger.info(f"Data subset: {refute_subset}")
    except Exception as e:
        logger.warning(f"Data subset refutation failed: {e}")

    logger.info("\nDoWhy analysis complete")

    return results


def sensitivity_analysis_omitted_confounder(
    ate_estimate: float,
    T: np.ndarray,
    Y: np.ndarray,
    rho_TU_range: np.ndarray = np.linspace(-0.5, 0.5, 11),
    rho_YU_range: np.ndarray = np.linspace(-0.5, 0.5, 11),
) -> pd.DataFrame:
    """
    Sensitivity analysis for omitted confounder.

    Simulates the effect of an unobserved confounder U with different
    correlation strengths with T and Y.

    Args:
        ate_estimate: Observed ATE estimate
        T: Treatment indicator
        Y: Outcome vector
        rho_TU_range: Range of correlations between T and U
        rho_YU_range: Range of correlations between Y and U

    Returns:
        DataFrame with sensitivity results
    """
    logger.info("Running sensitivity analysis for omitted confounder...")

    results = []

    var_T = np.var(T)
    var_Y = np.var(Y)

    for rho_TU in rho_TU_range:
        for rho_YU in rho_YU_range:
            # Bias due to omitted confounder
            # Approximate bias: rho_TU * rho_YU * (SD_Y / SD_T)
            bias = rho_TU * rho_YU * np.sqrt(var_Y / var_T) if var_T > 0 else 0

            # Adjusted estimate
            ate_adjusted = ate_estimate - bias

            results.append(
                {
                    "rho_TU": rho_TU,
                    "rho_YU": rho_YU,
                    "bias": bias,
                    "ate_adjusted": ate_adjusted,
                }
            )

    df_sensitivity = pd.DataFrame(results)

    logger.info(f"Sensitivity analysis complete: {len(df_sensitivity)} scenarios")

    return df_sensitivity


def placebo_outcome_test(
    X: np.ndarray,
    T: np.ndarray,
    pre_treatment_outcome: np.ndarray,
    propensity_scores: np.ndarray,
) -> dict[str, float]:
    """
    Placebo test using pre-treatment outcome.

    If our causal model is correct, treatment should have NO effect on
    pre-treatment outcomes.

    Args:
        X: Feature matrix
        T: Treatment indicator
        pre_treatment_outcome: Pre-treatment outcome (should not be affected by T)
        propensity_scores: Propensity scores

    Returns:
        Dictionary with placebo test results
    """
    from .ate import estimate_ate_ipw

    logger.info("Running placebo outcome test...")

    # Estimate "effect" of treatment on pre-treatment outcome
    # This should be close to zero
    placebo_ate = estimate_ate_ipw(pre_treatment_outcome, T, propensity_scores)

    # Bootstrap CI for placebo effect
    from .ate import bootstrap_ate_ci

    _, ci_lower, ci_upper = bootstrap_ate_ci(
        estimate_ate_ipw,
        None,
        pre_treatment_outcome,
        T,
        n_iterations=200,
        propensity_scores=propensity_scores,
    )

    # Check if CI contains zero
    contains_zero = ci_lower <= 0 <= ci_upper

    results = {
        "placebo_ate": placebo_ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "contains_zero": contains_zero,
        "test_passed": contains_zero,
    }

    if contains_zero:
        logger.info(
            f"Placebo test PASSED: Effect on pre-treatment outcome is "
            f"{placebo_ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], includes 0"
        )
    else:
        logger.warning(
            f"Placebo test FAILED: Effect on pre-treatment outcome is "
            f"{placebo_ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}], does NOT include 0"
        )

    return results


def leave_one_out_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    feature_names: list[str],
    ate_baseline: float,
) -> pd.DataFrame:
    """
    Leave-one-out sensitivity: re-estimate ATE after dropping each covariate.

    If ATE changes substantially when a covariate is dropped, it suggests
    that covariate is an important confounder.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        feature_names: List of feature names
        ate_baseline: Baseline ATE estimate with all features

    Returns:
        DataFrame with leave-one-out results
    """
    from .ate import estimate_ate_ipw
    from .propensity import PropensityModel, compute_ipw_weights

    logger.info("Running leave-one-out analysis...")

    results = []

    for i, feature_name in enumerate(feature_names):
        # Create feature matrix without feature i
        X_loo = np.delete(X, i, axis=1)

        # Re-fit propensity model
        prop_model_loo = PropensityModel(model_type="logistic")
        prop_model_loo.fit(X_loo, T)

        # Re-estimate ATE
        compute_ipw_weights(T, prop_model_loo.propensity_scores_, stabilize=True)
        ate_loo = estimate_ate_ipw(Y, T, prop_model_loo.propensity_scores_)

        # Compute change
        ate_change = ate_loo - ate_baseline
        pct_change = (ate_change / ate_baseline * 100) if ate_baseline != 0 else 0

        results.append(
            {
                "feature_dropped": feature_name,
                "ate_without_feature": ate_loo,
                "ate_change": ate_change,
                "pct_change": pct_change,
            }
        )

    df_loo = pd.DataFrame(results).sort_values("pct_change", key=abs, ascending=False)

    logger.info("\nFeatures with largest impact on ATE (top 5):")
    logger.info(f"\n{df_loo.head().to_string(index=False)}")

    return df_loo
