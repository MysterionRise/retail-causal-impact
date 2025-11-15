# path: tests/test_balance.py
"""
Tests for covariate balance improvement after weighting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from coupon_causal import balance, data, features, propensity, utils


@pytest.fixture
def confounded_data():
    """Generate confounded synthetic data."""
    config = {
        "random_state": 42,
        "synthetic": {
            "n_samples": 3000,
            "treatment_rate": 0.3,
            "confounding_strength": 3.0,  # Strong confounding
        },
        "features": {
            "categorical_features": ["customer_segment"],
            "continuous_features": [
                "loyalty_score",
                "recency_days",
                "monetary_value",
                "frequency_purchases",
            ],
        },
        "propensity": {
            "clip_bounds": [0.01, 0.99],
            "trim_percentiles": [1, 99],
        },
    }

    utils.set_random_seed(config["random_state"])

    df, _ = data.generate_synthetic_coupon_data(
        n_samples=config["synthetic"]["n_samples"],
        treatment_rate=config["synthetic"]["treatment_rate"],
        confounding_strength=config["synthetic"]["confounding_strength"],
        random_state=config["random_state"],
    )

    return df, config


def test_balance_improves_after_weighting(confounded_data):
    """Test that IPW weighting improves covariate balance."""
    df, config = confounded_data

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Compute IPW weights
    ipw_weights = propensity.compute_ipw_weights(
        T, prop_model.propensity_scores_, stabilize=True, trim_percentiles=(1, 99)
    )

    # Check balance before and after
    balance_before, balance_after, summary = balance.compare_balance_before_after(
        X, T, feature_engineer.get_feature_names(), ipw_weights
    )

    # Balance should improve (median SMD should decrease)
    assert summary["median_smd_after"] < summary["median_smd_before"], (
        "Median SMD should decrease after weighting"
    )

    # Mean SMD should also improve
    assert summary["mean_smd_after"] < summary["mean_smd_before"], (
        "Mean SMD should decrease after weighting"
    )


def test_smd_computation(confounded_data):
    """Test SMD computation is correct."""
    from coupon_causal.utils import compute_standardized_mean_difference

    # Simple test case
    x_treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_control = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    smd = compute_standardized_mean_difference(x_treated, x_control)

    # SMD = (mean_diff) / pooled_std
    # mean_diff = 3.0 - 4.0 = -1.0
    # var_treated = var([1,2,3,4,5]) = 2.0
    # var_control = var([2,3,4,5,6]) = 2.0
    # pooled_std = sqrt((2.0 + 2.0) / 2) = sqrt(2.0) ≈ 1.414
    # SMD = abs(-1.0 / 1.414) ≈ 0.707

    assert 0.6 < smd < 0.8, f"SMD should be around 0.707, got {smd}"


def test_variance_ratio_computation(confounded_data):
    """Test variance ratio computation."""
    df, config = confounded_data

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )

    # Compute variance ratios
    var_ratios = balance.compute_variance_ratio(X, T, feature_engineer.get_feature_names())

    # Variance ratios should be positive
    assert (var_ratios["variance_ratio"] > 0).all(), "Variance ratios should be positive"

    # For balanced data, variance ratios should be around 1
    # With confounding, they may deviate
    # Categorical features can have very different variances (esp. one-hot encoded)
    # Check that most (>80%) are in reasonable range (0.05 to 20)
    reasonable_range = var_ratios["variance_ratio"].between(0.05, 20)
    assert reasonable_range.mean() > 0.8, (
        f"Most variance ratios should be in reasonable range. "
        f"Got {reasonable_range.mean():.1%} in range [0.05, 20]"
    )


def test_positivity_check(confounded_data):
    """Test positivity violation detection."""
    df, config = confounded_data

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Check positivity
    positivity_diag = balance.check_positivity(
        prop_model.propensity_scores_, T, lower_threshold=0.01, upper_threshold=0.99
    )

    # Violation rate should be low (< 10%)
    assert (
        positivity_diag["violation_rate"] < 0.1
    ), "Positivity violation rate should be < 10%"


def test_balance_quality_assessment(confounded_data):
    """Test balance quality assessment."""
    df, config = confounded_data

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )

    # Fit propensity and compute weights
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    ipw_weights = propensity.compute_ipw_weights(
        T, prop_model.propensity_scores_, stabilize=True
    )

    # Compute balance after weighting
    balance_after = balance.compute_balance_table(
        X, T, feature_engineer.get_feature_names(), weights=ipw_weights
    )

    # Assess quality
    quality = balance.assess_balance_quality(balance_after, smd_threshold=0.1)

    # Should have quality metrics
    assert "mean_smd" in quality
    assert "assessment" in quality
    assert quality["assessment"] in ["Good", "Acceptable", "Poor"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
