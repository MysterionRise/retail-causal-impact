# path: tests/test_policy.py
"""
Tests for policy learning and uplift evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from coupon_causal import cate, data, features, policy, propensity, utils


@pytest.fixture
def policy_data():
    """Generate data for policy testing."""
    config = {
        "random_state": 42,
        "synthetic": {
            "n_samples": 3000,
            "treatment_rate": 0.3,
            "true_ate": 15.0,
        },
        "features": {
            "categorical_features": ["customer_segment"],
            "continuous_features": [
                "loyalty_score",
                "price_sensitivity",
                "monetary_value",
            ],
        },
        "cate": {"x_learner": {"n_estimators": 50, "max_depth": 6}},
    }

    utils.set_random_seed(config["random_state"])

    df, ground_truth = data.generate_synthetic_coupon_data(
        n_samples=config["synthetic"]["n_samples"],
        treatment_rate=config["synthetic"]["treatment_rate"],
        true_ate=config["synthetic"]["true_ate"],
        random_state=config["random_state"],
    )

    # Get CATE scores
    X, T, Y, _ = features.prepare_features(df, config["features"], fit=True)

    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    cate_scores, _ = cate.fit_x_learner(X, Y, T, prop_model.propensity_scores_, config)

    return T, Y, cate_scores, ground_truth


def test_qini_curve_monotonic(policy_data):
    """Test that Qini curve is monotonically increasing."""
    T, Y, cate_scores, _ = policy_data

    fractions, qini_values, _ = policy.compute_qini_curve(cate_scores, T, Y, n_bins=10)

    # Qini should be monotonically increasing (or at least non-decreasing)
    # Allow small violations due to noise
    diffs = np.diff(qini_values)
    proportion_increasing = (diffs >= -1e-6).mean()

    assert proportion_increasing > 0.8, "Qini curve should be mostly increasing"


def test_auuc_better_than_random(policy_data):
    """Test that learned policy AUUC is better than random."""
    T, Y, cate_scores, _ = policy_data

    # AUUC for learned policy
    auuc_learned = policy.compute_auuc(cate_scores, T, Y, normalize=False)

    # AUUC for random policy
    np.random.seed(42)
    random_scores = np.random.randn(len(cate_scores))
    auuc_random = policy.compute_auuc(random_scores, T, Y, normalize=False)

    # Learned should beat random
    assert auuc_learned > auuc_random, (
        f"Learned policy AUUC ({auuc_learned:.2f}) should beat "
        f"random ({auuc_random:.2f})"
    )


def test_policy_uplift_evaluation(policy_data):
    """Test policy uplift evaluation at different budgets."""
    T, Y, cate_scores, _ = policy_data

    policy_results = policy.evaluate_policy_uplift(
        cate_scores, T, Y, budget_fractions=[0.1, 0.2, 0.3], cost_per_treatment=1.0
    )

    # Should have results for each budget
    assert len(policy_results) == 3

    # Number treated should increase with budget
    assert (
        policy_results.loc[0, "n_treated"]
        < policy_results.loc[1, "n_treated"]
        < policy_results.loc[2, "n_treated"]
    )

    # Predicted uplift should be positive (on average)
    assert policy_results["predicted_avg_uplift"].mean() > 0


def test_optimal_threshold(policy_data):
    """Test optimal policy threshold selection."""
    T, Y, cate_scores, _ = policy_data

    # Test with capacity constraint
    capacity = len(cate_scores) // 5  # Target 20%
    threshold, treatment_indicator = policy.optimal_policy_threshold(
        cate_scores, cost_per_treatment=1.0, treatment_capacity=capacity
    )

    # Should select exactly capacity customers
    assert treatment_indicator.sum() == capacity

    # Selected customers should have highest CATE scores
    selected_cate = cate_scores[treatment_indicator == 1]
    not_selected_cate = cate_scores[treatment_indicator == 0]

    assert selected_cate.min() >= not_selected_cate.max() - 1e-6, (
        "Selected customers should have higher CATE than non-selected"
    )


def test_segment_uplift_analysis(policy_data):
    """Test segment uplift analysis."""
    T, Y, cate_scores, _ = policy_data

    # Create a simple dataframe with segments
    import pandas as pd

    df = pd.DataFrame(
        {
            "customer_segment": np.random.choice(
                ["high", "mid", "low"], size=len(cate_scores)
            )
        }
    )

    segment_stats = policy.segment_uplift_analysis(
        cate_scores, df, "customer_segment", top_k=3
    )

    # Should have stats for each segment
    assert len(segment_stats) <= 3

    # Should have required columns
    assert "segment" in segment_stats.columns
    assert "avg_predicted_uplift" in segment_stats.columns


def test_targeting_recommendations(policy_data):
    """Test targeting recommendations generation."""
    T, Y, cate_scores, _ = policy_data

    # Create simple dataframe
    import pandas as pd

    df = pd.DataFrame({"customer_id": np.arange(len(cate_scores))})

    threshold = np.median(cate_scores)

    recommendations = policy.create_targeting_recommendations(
        cate_scores, df, threshold, output_path=None
    )

    # Should have required columns
    assert "customer_id" in recommendations.columns
    assert "predicted_uplift" in recommendations.columns
    assert "recommend_treatment" in recommendations.columns

    # Recommendations should be binary
    assert set(recommendations["recommend_treatment"].unique()).issubset({0, 1})

    # High CATE customers should be recommended
    high_cate_recommended = recommendations[
        recommendations["predicted_uplift"] > threshold
    ]["recommend_treatment"].mean()

    assert high_cate_recommended > 0.8, "Most high-CATE customers should be recommended"


def test_qini_curve_bounds(policy_data):
    """Test that Qini curve values are in reasonable bounds."""
    T, Y, cate_scores, _ = policy_data

    fractions, qini_values, random_baseline = policy.compute_qini_curve(
        cate_scores, T, Y, n_bins=20
    )

    # Fractions should be between 0 and 1
    assert (fractions >= 0).all() and (fractions <= 1).all()

    # Qini values should be finite
    assert np.isfinite(qini_values).all()

    # Random baseline should be finite
    assert np.isfinite(random_baseline).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Doubly-robust policy evaluation and targeting comparison
# ---------------------------------------------------------------------------


@pytest.fixture
def dr_data():
    """Synthetic data with propensity, outcome models and DR pseudo-outcomes."""
    from coupon_causal import ate

    df, ground_truth = data.generate_synthetic_coupon_data(n_samples=4000, random_state=3)
    feature_config = {
        "categorical_features": ["customer_segment"],
        "continuous_features": ["loyalty_score", "price_sensitivity", "monetary_value", "recency_days"],
    }
    X, T, Y, _ = features.prepare_features(df, feature_config, fit=True)
    e = propensity.PropensityModel(model_type="logistic").fit(X, T).propensity_scores_
    _, (m1, m0) = ate.estimate_ate_aipw(X, Y, T, e, model_type="linear")
    dr = policy.compute_dr_scores(Y, T, e, m0.predict(X), m1.predict(X))
    return X, Y, T, e, dr, df["true_treatment_effect"].values, ground_truth


def test_dr_scores_average_to_ate(dr_data):
    _, _, _, _, dr, _, ground_truth = dr_data
    assert abs(dr.mean() - ground_truth["true_ate"]) / ground_truth["true_ate"] < 0.2


def test_dr_uplift_curve_endpoints(dr_data):
    _, _, _, _, dr, true_cate, _ = dr_data
    fractions, gain, baseline = policy.compute_dr_uplift_curve(true_cate, dr, n_bins=10)

    assert len(fractions) == len(gain) == len(baseline) == 11
    assert gain[0] == 0 and baseline[0] == 0
    assert np.isclose(gain[-1], dr.sum()) and np.isclose(baseline[-1], dr.sum())
    # Targeting by the true effect should beat random at the halfway point
    assert gain[5] > baseline[5]


def test_targeting_strategies_ranking(dr_data):
    X, Y, T, _, dr, true_cate, _ = dr_data
    response = policy.fit_response_model(X, Y, T, random_state=0, n_estimators=30)
    table = policy.evaluate_targeting_strategies(
        {"uplift_model": true_cate + np.random.default_rng(0).normal(0, 2, len(true_cate)),
         "response_model": response.predict(X)},
        dr,
        budget_fraction=0.2,
        true_cate=true_cate,
    )

    assert set(table["strategy"]) == {"uplift_model", "response_model", "random", "oracle (true CATE)"}
    assert (table["n_targeted"] == int(0.2 * len(dr))).all()
    val = table.set_index("strategy")["true_uplift_per_customer"]
    assert val["oracle (true CATE)"] >= val["uplift_model"] > val["random"]
    assert val["uplift_model"] > val["response_model"], "Causal targeting must beat spend prediction"


def test_calibration_by_decile(dr_data):
    _, _, _, _, dr, true_cate, _ = dr_data
    noisy_pred = true_cate + np.random.default_rng(1).normal(0, 2, len(true_cate))
    cal = policy.cate_calibration_by_decile(noisy_pred, dr, true_cate, n_bins=10)

    assert list(cal["decile"]) == list(range(1, 11))
    assert cal["n"].sum() == len(dr)
    assert cal["predicted_cate"].is_monotonic_increasing
    assert cal["true_cate"].iloc[-1] > cal["true_cate"].iloc[0]


def test_evaluate_policy_uplift_dr_columns(dr_data):
    _, Y, T, _, dr, true_cate, _ = dr_data
    table = policy.evaluate_policy_uplift(true_cate, T, Y, budget_fractions=[0.1, 0.3], dr_scores=dr)

    assert {"dr_uplift_per_unit", "dr_total_uplift", "dr_net_benefit", "dr_roi"} <= set(table.columns)
    assert np.isclose(table["dr_total_uplift"].iloc[0], table["dr_uplift_per_unit"].iloc[0] * table["n_treated"].iloc[0])
