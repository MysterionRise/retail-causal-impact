# path: tests/test_refute.py
"""
Tests for refutation and sensitivity checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from coupon_causal import ate, data, features, propensity, refute

FEATURE_CONFIG = {
    "categorical_features": ["customer_segment", "region", "preferred_category"],
    "continuous_features": [
        "recency_days",
        "frequency_purchases",
        "monetary_value",
        "avg_basket_size",
        "loyalty_score",
        "price_sensitivity",
        "pre_trend",
    ],
}


@pytest.fixture(scope="module")
def fitted():
    """Synthetic data with features, propensity and AIPW point estimate."""
    df, ground_truth = data.generate_synthetic_coupon_data(n_samples=3000, random_state=7)
    X, T, Y, fe = features.prepare_features(df, FEATURE_CONFIG, fit=True)
    prop = propensity.PropensityModel(model_type="logistic").fit(X, T)
    ate_est, _ = ate.estimate_ate_aipw(X, Y, T, prop.propensity_scores_, model_type="linear")
    return df, ground_truth, X, T, Y, fe.get_feature_names(), prop.propensity_scores_, ate_est


def test_build_causal_graph_drops_missing_nodes():
    graph = refute.build_causal_graph(columns=["loyalty_score", "treatment", "outcome"])
    assert graph.startswith("digraph {")
    assert "loyalty_score -> treatment;" in graph
    assert "recency_days" not in graph
    assert "treatment -> outcome;" in graph
    assert "\n" not in graph  # DoWhy's pydot parser needs a single line


def test_robustness_value_properties(fitted):
    _, ground_truth, X, T, Y, _, _, _ = fitted
    rv = refute.robustness_value(X, Y, T)

    assert 0.0 < rv["robustness_value"] < 1.0
    assert 0.0 <= rv["robustness_value_alpha"] <= rv["robustness_value"]
    assert 0.0 < rv["partial_r2_treatment"] < 1.0
    # Regression adjustment on this (linear-ish) DGP should be close to the truth
    assert abs(rv["ols_estimate"] - ground_truth["true_ate"]) / ground_truth["true_ate"] < 0.3


def test_placebo_treatment_is_centred_on_zero(fitted):
    _, _, X, T, Y, _, _, ate_est = fitted
    res = refute.placebo_treatment_test(X, Y, T, ate_est, n_permutations=10, random_state=1)

    assert len(res["effects"]) == 10
    assert abs(res["placebo_mean"]) < 1.0, "Permuted treatment should have ~zero effect"
    assert res["passed"]
    # The real estimate should be far outside the placebo distribution
    assert res["p_value_real_effect"] < 0.01


def test_placebo_outcome_on_pre_treatment_variable(fitted):
    df, _, X, T, _, _, e, _ = fitted
    res = refute.placebo_outcome_test(X, T, df["pre_trend"].values, e, n_iterations=30)

    assert np.isfinite(res["ci_lower"]) and np.isfinite(res["ci_upper"])
    assert res["ci_lower"] <= res["placebo_ate"] <= res["ci_upper"]
    assert res["passed"], "Coupons cannot affect a pre-campaign variable"


def test_leave_one_out_flags_loyalty(fitted):
    _, _, X, T, Y, names, _, ate_est = fitted
    loo = refute.leave_one_out_analysis(X, Y, T, names, ate_est)

    assert set(loo.columns) >= {"feature_dropped", "ate_without_feature", "ate_change", "pct_change"}
    assert len(loo) == len(names)
    # Loyalty is the strongest confounder in the generator
    assert loo.iloc[0]["feature_dropped"] == "loyalty_score"


def test_run_dowhy_analysis_identifies_and_refutes(fitted):
    df, ground_truth, *_ = fitted
    cols = [c for c in df.columns if not c.startswith("true_") and c != "customer_id"]
    res = refute.run_dowhy_analysis(df[cols], n_simulations=5, random_state=0)

    assert "loyalty_score" in res["backdoor_variables"]
    assert "price_sensitivity" not in res["backdoor_variables"]  # outcome-only cause
    assert abs(res["estimate"] - ground_truth["true_ate"]) / ground_truth["true_ate"] < 0.3

    table = res["refutations"]
    assert list(table["test"]) == [
        "Placebo treatment (permuted)",
        "Random common cause added",
        "Random 80% data subset",
    ]
    placebo = table.iloc[0]
    assert abs(placebo["new_effect"]) < 2.0
    assert table["passed"].all()


def test_run_all_refutations_summary(fitted):
    df, _, X, T, Y, names, e, ate_est = fitted
    config = {
        "random_state": 0,
        "propensity": {"clip_bounds": [0.01, 0.99]},
        "refutation": {
            "n_simulations": 5,
            "n_placebo_permutations": 5,
            "placebo_outcome_bootstrap": 20,
        },
    }
    res = refute.run_all_refutations(df, X, Y, T, names, e, ate_est, config)

    summary = res["summary"]
    assert len(summary) == 6
    assert summary["passed"].all()
    assert set(summary["source"]) == {"DoWhy", "pipeline"}
    assert "robustness_value" in res["robustness"]
    assert res["dowhy"]["graph"].startswith("digraph")
