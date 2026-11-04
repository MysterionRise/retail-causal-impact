# path: src/coupon_causal/refute.py
"""
Refutation and sensitivity analysis.

Two complementary layers:

1. **DoWhy** (explicit causal graph): identify the adjustment set from the DAG,
   estimate the effect by regression adjustment, and run DoWhy's refuters
   (placebo treatment, random common cause, data subset).

   DoWhy's ``propensity_score_weighting`` estimator is *not* used for the
   refuters: on this data its placebo-treatment check returns a large
   non-zero "effect" (around -9 when the truth is 0), so it would report a
   spurious failure. Regression adjustment behaves as expected.

2. **In-house checks against the pipeline's own AIPW estimator**: permuted
   (placebo) treatment, effect on a pre-treatment outcome, leave-one-covariate-
   out stability, and the Cinelli & Hazlett (2020) robustness value for
   unobserved confounding.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------

# The analyst's assumed DAG for the coupon campaign. Confounders point at both
# treatment and outcome; effect modifiers and other drivers of spend point at
# the outcome only.
DEFAULT_CAUSAL_EDGES: list[tuple[str, str]] = [
    ("loyalty_score", "treatment"),
    ("loyalty_score", "outcome"),
    ("recency_days", "treatment"),
    ("recency_days", "outcome"),
    ("frequency_purchases", "treatment"),
    ("frequency_purchases", "outcome"),
    ("monetary_value", "treatment"),
    ("monetary_value", "outcome"),
    ("customer_segment", "treatment"),
    ("customer_segment", "outcome"),
    ("price_sensitivity", "outcome"),
    ("avg_basket_size", "outcome"),
    ("pre_trend", "outcome"),
    ("treatment", "outcome"),
]


def build_causal_graph(
    columns: list[str] | None = None,
    edges: list[tuple[str, str]] | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> str:
    """
    Build a DoWhy-compatible DOT graph string.

    Args:
        columns: Columns available in the data. Edges touching a node that is
            not available are dropped, so the same spec works for datasets
            that lack some variables.
        edges: (source, target) pairs. Defaults to ``DEFAULT_CAUSAL_EDGES``.
        treatment_col: Name of the treatment node
        outcome_col: Name of the outcome node

    Returns:
        Single-line DOT string (DoWhy's pydot parser rejects comments and
        multi-line layouts).
    """
    edges = edges if edges is not None else DEFAULT_CAUSAL_EDGES
    if columns is not None:
        allowed = set(columns) | {treatment_col, outcome_col}
        edges = [(a, b) for a, b in edges if a in allowed and b in allowed]
    body = " ".join(f"{a} -> {b};" for a, b in edges)
    return f"digraph {{ {body} }}"


def create_causal_graph() -> str:
    """Backward-compatible alias for the default graph."""
    return build_causal_graph()


# ---------------------------------------------------------------------------
# DoWhy
# ---------------------------------------------------------------------------


def _significance(expected: float, simulated: np.ndarray) -> float:
    """Two-sided normal p-value of ``expected`` under the simulated distribution."""
    simulated = np.asarray(simulated, dtype=float)
    simulated = simulated[np.isfinite(simulated)]
    if len(simulated) < 2 or simulated.std(ddof=1) == 0:
        return float("nan")
    z = (expected - simulated.mean()) / simulated.std(ddof=1)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def run_dowhy_analysis(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    edges: list[tuple[str, str]] | None = None,
    n_simulations: int = 20,
    random_state: int = 42,
    estimator: str = "backdoor.linear_regression",
) -> dict:
    """
    Identify, estimate and refute with DoWhy from an explicit causal graph.

    Args:
        df: DataFrame with treatment, outcome and covariates (raw columns)
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        edges: Causal edges; defaults to ``DEFAULT_CAUSAL_EDGES`` filtered to
            the columns present in ``df``
        n_simulations: Simulations per refuter
        random_state: Seed passed to DoWhy's refuters
        estimator: DoWhy estimator name (see module docstring for why the
            default is regression adjustment)

    Returns:
        Dict with ``graph`` (DOT), ``backdoor_variables``, ``estimand`` (text),
        ``estimate``, ``method`` and ``refutations`` (DataFrame).
    """
    try:
        from dowhy import CausalModel
    except ImportError:
        logger.error("DoWhy not installed. Install with: pip install dowhy")
        return {}

    logger.info("Running DoWhy identification and refutation...")

    graph = build_causal_graph(list(df.columns), edges, treatment_col, outcome_col)
    model = CausalModel(data=df, treatment=treatment_col, outcome=outcome_col, graph=graph)

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    backdoor = list(identified.get_backdoor_variables())
    logger.info(f"Backdoor adjustment set: {backdoor}")

    estimate = model.estimate_effect(identified, method_name=estimator)
    logger.info(f"DoWhy {estimator} estimate: {estimate.value:.2f}")

    specs = [
        # (label, method_name, kwargs, expected value)
        ("Placebo treatment (permuted)", "placebo_treatment_refuter", {}, 0.0),
        ("Random common cause added", "random_common_cause", {}, estimate.value),
        ("Random 80% data subset", "data_subset_refuter", {"subset_fraction": 0.8}, estimate.value),
    ]

    rows = []
    for label, method, kwargs, expected in specs:
        try:
            ref = model.refute_estimate(
                identified,
                estimate,
                method_name=method,
                num_simulations=n_simulations,
                random_seed=random_state,
                **kwargs,
            )
            result = ref.refutation_result if isinstance(ref.refutation_result, dict) else {}
            p_value = result.get("p_value", float("nan"))
            rows.append(
                {
                    "test": label,
                    "source": "DoWhy",
                    "expected": expected,
                    "original_effect": float(ref.estimated_effect),
                    "new_effect": float(ref.new_effect),
                    "p_value": float(p_value) if p_value is not None else float("nan"),
                    "passed": bool(p_value is not None and not np.isnan(p_value) and p_value > 0.05),
                }
            )
            logger.info(f"  {label}: new effect {ref.new_effect:.2f} (p={p_value:.3f})")
        except Exception as e:  # DoWhy refuters occasionally fail on edge cases
            logger.warning(f"  {label} failed: {e}")
            rows.append(
                {
                    "test": label,
                    "source": "DoWhy",
                    "expected": expected,
                    "original_effect": float(estimate.value),
                    "new_effect": float("nan"),
                    "p_value": float("nan"),
                    "passed": False,
                }
            )

    return {
        "graph": graph,
        "backdoor_variables": backdoor,
        "estimand": str(identified),
        "estimate": float(estimate.value),
        "method": estimator,
        "refutations": pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# In-house checks against the pipeline's AIPW estimator
# ---------------------------------------------------------------------------


def placebo_treatment_test(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    ate_estimate: float,
    n_permutations: int = 20,
    random_state: int = 42,
    propensity_config: dict | None = None,
) -> dict:
    """
    Placebo treatment: permute T, refit propensity and AIPW, expect ~0.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        ate_estimate: The real AIPW estimate (for the significance comparison)
        n_permutations: Number of random permutations
        random_state: Seed
        propensity_config: Propensity config section (clip bounds, model params)

    Returns:
        Dict with placebo mean/std, individual effects, p-value that the
        placebo distribution is centred on zero, and p-value of the real
        estimate under the placebo distribution.
    """
    from .ate import _refit_propensity, estimate_ate_aipw

    logger.info(f"Placebo treatment test: {n_permutations} permutations...")
    rng = np.random.default_rng(random_state)
    package_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
    previous_level = package_logger.level
    package_logger.setLevel(max(previous_level, logging.WARNING))

    effects = []
    try:
        for _ in range(n_permutations):
            T_perm = rng.permutation(T)
            e_perm = _refit_propensity(X, T_perm, "logistic", propensity_config, random_state)
            effect, _ = estimate_ate_aipw(X, Y, T_perm, e_perm, model_type="linear")
            effects.append(effect)
    finally:
        package_logger.setLevel(previous_level)

    effects = np.array(effects)
    p_zero = _significance(0.0, effects)
    p_real = _significance(ate_estimate, effects)

    result = {
        "placebo_mean": float(effects.mean()),
        "placebo_std": float(effects.std(ddof=1)) if len(effects) > 1 else float("nan"),
        "effects": effects,
        "p_value_zero": p_zero,
        "p_value_real_effect": p_real,
        "passed": bool(p_zero > 0.05),
    }
    logger.info(
        f"  placebo effect {result['placebo_mean']:.2f} ± {result['placebo_std']:.2f} "
        f"(real estimate {ate_estimate:.2f}, p under placebo = {p_real:.3g})"
    )
    return result


def placebo_outcome_test(
    X: np.ndarray,
    T: np.ndarray,
    pre_treatment_outcome: np.ndarray,
    propensity_scores: np.ndarray,
    n_iterations: int = 100,
    random_state: int = 42,
    propensity_config: dict | None = None,
) -> dict[str, float]:
    """
    Placebo outcome: the treatment should have no "effect" on a variable
    measured before the campaign.

    Args:
        X: Feature matrix (needed to refit the propensity model per bootstrap draw)
        T: Treatment indicator
        pre_treatment_outcome: A pre-treatment variable (e.g. pre-campaign trend)
        propensity_scores: Propensity scores used for the point estimate
        n_iterations: Bootstrap draws for the interval
        random_state: Seed
        propensity_config: Propensity config section

    Returns:
        Dict with placebo effect, CI and pass flag (CI contains zero)
    """
    from .ate import bootstrap_ate_ci, estimate_ate_ipw

    logger.info("Placebo outcome test (pre-treatment variable)...")
    placebo_ate, ci_lower, ci_upper = bootstrap_ate_ci(
        estimate_ate_ipw,
        X,
        pre_treatment_outcome,
        T,
        n_iterations=n_iterations,
        random_state=random_state,
        propensity_model_type="logistic",
        propensity_config=propensity_config,
        propensity_scores=propensity_scores,
    )
    contains_zero = bool(np.isfinite(ci_lower) and ci_lower <= 0 <= ci_upper)

    results = {
        "placebo_ate": float(placebo_ate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "contains_zero": contains_zero,
        "passed": contains_zero,
    }
    status = "PASSED" if contains_zero else "FAILED"
    logger.info(
        f"  {status}: effect on pre-treatment outcome {placebo_ate:.2f} "
        f"[{ci_lower:.2f}, {ci_upper:.2f}]"
    )
    return results


def leave_one_out_analysis(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    feature_names: list[str],
    ate_baseline: float,
    propensity_config: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Leave-one-covariate-out: re-estimate the (AIPW) ATE after dropping each
    covariate. Large swings flag covariates that carry most of the confounding.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        feature_names: Column names of X
        ate_baseline: ATE with all covariates
        propensity_config: Propensity config section
        random_state: Seed

    Returns:
        DataFrame sorted by absolute percentage change
    """
    from .ate import _refit_propensity, estimate_ate_aipw

    logger.info("Leave-one-covariate-out analysis...")
    package_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
    previous_level = package_logger.level
    package_logger.setLevel(max(previous_level, logging.WARNING))

    rows = []
    try:
        for i, feature_name in enumerate(feature_names):
            X_loo = np.delete(X, i, axis=1)
            e_loo = _refit_propensity(X_loo, T, "logistic", propensity_config, random_state)
            ate_loo, _ = estimate_ate_aipw(X_loo, Y, T, e_loo, model_type="linear")
            change = ate_loo - ate_baseline
            rows.append(
                {
                    "feature_dropped": feature_name,
                    "ate_without_feature": ate_loo,
                    "ate_change": change,
                    "pct_change": (change / ate_baseline * 100) if ate_baseline != 0 else 0.0,
                }
            )
    finally:
        package_logger.setLevel(previous_level)

    df_loo = pd.DataFrame(rows).sort_values("pct_change", key=abs, ascending=False)
    logger.info("  Largest shifts when a covariate is dropped:")
    logger.info(f"\n{df_loo.head().to_string(index=False)}")
    return df_loo


def robustness_value(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Cinelli & Hazlett (2020) sensitivity statistics for unobserved confounding.

    Fits OLS ``Y ~ T + X`` and reports:

    - ``partial_r2_treatment``: share of residual outcome variance explained by T
    - ``robustness_value``: RV(q=1). An unobserved confounder would need to
      explain at least this share of the residual variance of *both* treatment
      and outcome to bring the estimate to zero.
    - ``robustness_value_alpha``: RV(q=1, alpha). Same, but to make the
      estimate no longer statistically distinguishable from zero.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        alpha: Significance level for the second robustness value

    Returns:
        Dict of statistics
    """
    import statsmodels.api as sm

    design = sm.add_constant(np.column_stack([T, X]))
    fit = sm.OLS(Y, design).fit()
    t_stat = float(fit.tvalues[1])
    dof = float(fit.df_resid)

    partial_r2 = t_stat**2 / (t_stat**2 + dof)
    f = abs(t_stat) / np.sqrt(dof)  # partial Cohen's f of the treatment

    def _rv(f_q: float) -> float:
        if f_q <= 0:
            return 0.0
        return float(0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2))

    t_alpha = stats.t.ppf(1 - alpha / 2, dof - 1)
    f_alpha = f - t_alpha / np.sqrt(dof - 1)

    result = {
        "ols_estimate": float(fit.params[1]),
        "ols_se": float(fit.bse[1]),
        "t_statistic": t_stat,
        "partial_r2_treatment": float(partial_r2),
        "robustness_value": _rv(f),
        "robustness_value_alpha": _rv(f_alpha),
        "alpha": alpha,
    }
    logger.info(
        f"Robustness value: a confounder would need partial R² ≥ {result['robustness_value']:.1%} "
        f"with both treatment and outcome to explain the effect away "
        f"(≥ {result['robustness_value_alpha']:.1%} to make it non-significant)"
    )
    return result


def run_all_refutations(
    df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    feature_names: list[str],
    propensity_scores: np.ndarray,
    ate_estimate: float,
    config: dict,
) -> dict:
    """
    Run every refutation/sensitivity check and collect a summary table.

    Args:
        df: Raw DataFrame (for DoWhy, which works on named columns)
        X, Y, T: Engineered feature matrix, outcome and treatment
        feature_names: Column names of X
        propensity_scores: Propensity scores used by the pipeline
        ate_estimate: The headline AIPW estimate being stress-tested
        config: Full configuration dictionary

    Returns:
        Dict with ``dowhy``, ``placebo_treatment``, ``placebo_outcome``,
        ``leave_one_out``, ``robustness`` and a tidy ``summary`` DataFrame.
    """
    ref_cfg = config.get("refutation", {})
    propensity_config = config.get("propensity", {})
    random_state = config.get("random_state", 42)
    n_sim = ref_cfg.get("n_simulations", 20)

    results: dict = {}
    summary_rows: list[dict] = []

    # 1. DoWhy: identification + refuters on regression adjustment
    dowhy_cols = [c for c in df.columns if not c.startswith("true_") and c != "customer_id"]
    try:
        dowhy_res = run_dowhy_analysis(
            df[dowhy_cols], n_simulations=n_sim, random_state=random_state
        )
    except Exception as e:
        logger.warning(f"DoWhy analysis failed: {e}")
        dowhy_res = {}
    results["dowhy"] = dowhy_res
    if dowhy_res:
        summary_rows.extend(dowhy_res["refutations"].to_dict("records"))

    # 2. Placebo treatment against our AIPW estimator
    placebo = placebo_treatment_test(
        X,
        Y,
        T,
        ate_estimate,
        n_permutations=ref_cfg.get("n_placebo_permutations", 20),
        random_state=random_state,
        propensity_config=propensity_config,
    )
    results["placebo_treatment"] = placebo
    summary_rows.append(
        {
            "test": "Placebo treatment (permuted), AIPW",
            "source": "pipeline",
            "expected": 0.0,
            "original_effect": ate_estimate,
            "new_effect": placebo["placebo_mean"],
            "p_value": placebo["p_value_zero"],
            "passed": placebo["passed"],
        }
    )

    # 3. Placebo outcome: pre-treatment variable
    placebo_col = ref_cfg.get("placebo_outcome_column", "pre_trend")
    if placebo_col in df.columns:
        placebo_out = placebo_outcome_test(
            X,
            T,
            df[placebo_col].values.astype(float),
            propensity_scores,
            n_iterations=ref_cfg.get("placebo_outcome_bootstrap", 100),
            random_state=random_state,
            propensity_config=propensity_config,
        )
        results["placebo_outcome"] = placebo_out
        summary_rows.append(
            {
                "test": f"Effect on pre-treatment variable ({placebo_col}), IPW",
                "source": "pipeline",
                "expected": 0.0,
                "original_effect": ate_estimate,
                "new_effect": placebo_out["placebo_ate"],
                "p_value": float("nan"),
                "passed": placebo_out["passed"],
            }
        )
    else:
        results["placebo_outcome"] = None

    # 4. Leave-one-covariate-out
    loo = leave_one_out_analysis(
        X, Y, T, feature_names, ate_estimate, propensity_config, random_state
    )
    results["leave_one_out"] = loo
    worst = loo.iloc[0]
    summary_rows.append(
        {
            "test": f"Drop most influential covariate ({worst['feature_dropped']})",
            "source": "pipeline",
            "expected": ate_estimate,
            "original_effect": ate_estimate,
            "new_effect": float(worst["ate_without_feature"]),
            "p_value": float("nan"),
            "passed": bool(abs(worst["pct_change"]) < ref_cfg.get("loo_max_pct_change", 25)),
        }
    )

    # 5. Robustness value
    rv = robustness_value(X, Y, T)
    results["robustness"] = rv

    summary = pd.DataFrame(summary_rows)
    results["summary"] = summary

    n_pass = int(summary["passed"].sum())
    logger.info(f"\nRefutation summary: {n_pass}/{len(summary)} checks passed")
    logger.info(f"\n{summary.to_string(index=False)}")
    return results


def identify_effect(
    df: pd.DataFrame,
    edges: list[tuple[str, str]] | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    estimator: str = "backdoor.linear_regression",
) -> dict:
    """
    Identify the adjustment set from a causal graph and estimate the effect
    by regression adjustment, without running refuters (fast: < 1 s).

    Used interactively ("what if region also drives who gets a coupon?") to
    show how an edit to the assumed graph changes the backdoor set and the
    estimate.

    Args:
        df: DataFrame with treatment, outcome and covariates
        edges: Causal edges; defaults to ``DEFAULT_CAUSAL_EDGES`` filtered to ``df``
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        estimator: DoWhy estimator name

    Returns:
        Dict with ``graph``, ``edges``, ``backdoor_variables``, ``identified``
        (bool) and ``estimate`` (NaN when not identified).
    """
    from dowhy import CausalModel

    edges = edges if edges is not None else DEFAULT_CAUSAL_EDGES
    allowed = set(df.columns) | {treatment_col, outcome_col}
    edges = [(a, b) for a, b in edges if a in allowed and b in allowed]
    graph = build_causal_graph(list(df.columns), edges, treatment_col, outcome_col)

    model = CausalModel(data=df, treatment=treatment_col, outcome=outcome_col, graph=graph)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    backdoor = list(identified.get_backdoor_variables() or [])

    result = {
        "graph": graph,
        "edges": edges,
        "backdoor_variables": backdoor,
        "identified": identified.estimands.get("backdoor") is not None,
        "estimate": float("nan"),
    }
    if result["identified"]:
        estimate = model.estimate_effect(identified, method_name=estimator)
        result["estimate"] = float(estimate.value)
    return result
