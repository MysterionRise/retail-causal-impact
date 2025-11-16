# path: tests/test_synth_recovery.py
"""
Tests for recovering true ATE and CATE on synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from coupon_causal import ate, cate, data, features, propensity, utils


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    config = {
        "random_state": 42,
        "synthetic": {
            "n_samples": 5000,
            "treatment_rate": 0.3,
            "noise_level": 0.1,
            "true_ate": 15.0,
            "confounding_strength": 2.0,
        },
        "features": {
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
        },
        "propensity": {
            "clip_bounds": [0.01, 0.99],
            "trim_percentiles": [1, 99],
            "models": {
                "logistic": {"max_iter": 1000, "C": 1.0},
                "gbm": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "min_samples_leaf": 50,
                },
            },
        },
        "ate": {
            "bootstrap_iterations": 100,
            "bootstrap_confidence_level": 0.95,
            "ipw_stabilize": True,
        },
        "cate": {
            "cross_fitting_folds": 3,
            "x_learner": {"n_estimators": 50, "max_depth": 6},
        },
    }

    utils.set_random_seed(config["random_state"])

    df, ground_truth = data.generate_synthetic_coupon_data(
        n_samples=config["synthetic"]["n_samples"],
        treatment_rate=config["synthetic"]["treatment_rate"],
        noise_level=config["synthetic"]["noise_level"],
        true_ate=config["synthetic"]["true_ate"],
        confounding_strength=config["synthetic"]["confounding_strength"],
        random_state=config["random_state"],
    )

    return df, ground_truth, config


def test_ate_recovery_aipw(synthetic_data):
    """Test that AIPW recovers true ATE within acceptable error."""
    df, ground_truth, config = synthetic_data

    # Prepare features
    X, T, Y, _ = features.prepare_features(df, config["features"], fit=True)

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Estimate ATE using AIPW
    ate_estimate, _ = ate.estimate_ate_aipw(
        X, Y, T, prop_model.propensity_scores_, model_type="linear"
    )

    true_ate = ground_truth["true_ate"]

    # Check recovery within 20% relative error
    relative_error = abs(ate_estimate - true_ate) / true_ate
    assert relative_error < 0.2, (
        f"AIPW ATE ({ate_estimate:.2f}) deviates from true ATE ({true_ate:.2f}) "
        f"by {relative_error:.1%}"
    )


def test_naive_ate_is_biased(synthetic_data):
    """Test that naive ATE is biased due to confounding."""
    df, ground_truth, config = synthetic_data

    T = df["treatment"].values
    Y = df["outcome"].values

    # Naive estimate
    naive_ate = ate.estimate_ate_naive(Y, T)
    true_ate = ground_truth["true_ate"]

    # Naive should be biased (different from true)
    # With confounding_strength=2.0, bias should be substantial
    assert abs(naive_ate - true_ate) > 2.0, (
        "Naive ATE should be substantially biased with confounding"
    )


def test_cate_heterogeneity(synthetic_data):
    """Test that CATE captures treatment effect heterogeneity."""
    df, ground_truth, config = synthetic_data

    # Prepare features
    X, T, Y, _ = features.prepare_features(df, config["features"], fit=True)

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Estimate CATE using X-Learner
    cate_pred, _ = cate.fit_x_learner(X, Y, T, prop_model.propensity_scores_, config)

    # Check that CATE has meaningful variance
    cate_std = cate_pred.std()
    true_cate_std = ground_truth["cate_std"]

    # CATE std should be within reasonable range of true std
    assert cate_std > 0, "CATE should have positive variance"
    assert (
        abs(cate_std - true_cate_std) / true_cate_std < 0.5
    ), "CATE std should be within 50% of true std"


def test_cate_mean_equals_ate(synthetic_data):
    """Test that mean CATE equals ATE."""
    df, ground_truth, config = synthetic_data

    # Prepare features
    X, T, Y, _ = features.prepare_features(df, config["features"], fit=True)

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Estimate CATE
    cate_pred, _ = cate.fit_x_learner(X, Y, T, prop_model.propensity_scores_, config)

    # Mean CATE should approximate ATE
    mean_cate = cate_pred.mean()
    true_ate = ground_truth["true_ate"]

    relative_error = abs(mean_cate - true_ate) / true_ate
    assert relative_error < 0.3, (
        f"Mean CATE ({mean_cate:.2f}) should approximate ATE ({true_ate:.2f})"
    )


def test_propensity_model_calibration(synthetic_data):
    """Test that propensity model is reasonably calibrated."""
    df, ground_truth, config = synthetic_data

    # Prepare features
    X, T, Y, _ = features.prepare_features(df, config["features"], fit=True)

    # Fit propensity model
    prop_model = propensity.PropensityModel(model_type="logistic")
    prop_model.fit(X, T)

    # Compute diagnostics
    diagnostics = prop_model.compute_diagnostics(T)

    # AUC should be > 0.6 (better than random)
    assert diagnostics["auc"] > 0.6, "Propensity model should have AUC > 0.6"

    # Brier score should be reasonable (< 0.25 for binary classification)
    assert diagnostics["brier_score"] < 0.25, "Propensity model should have Brier < 0.25"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
