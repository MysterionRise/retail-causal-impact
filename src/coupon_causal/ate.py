# path: src/coupon_causal/ate.py
"""
Average Treatment Effect (ATE) estimation methods.
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def estimate_ate_naive(Y: np.ndarray, T: np.ndarray) -> float:
    """
    Naive ATE estimate: simple difference in means.

    This estimator is biased if treatment assignment is confounded.

    Args:
        Y: Outcome vector
        T: Treatment indicator

    Returns:
        Naive ATE estimate
    """
    mean_treated = Y[T == 1].mean()
    mean_control = Y[T == 0].mean()

    ate_naive = mean_treated - mean_control

    logger.info(f"Naive ATE: ${ate_naive:.2f}")
    logger.info(f"  Mean(Y|T=1): ${mean_treated:.2f}")
    logger.info(f"  Mean(Y|T=0): ${mean_control:.2f}")

    return ate_naive


def estimate_ate_ipw(
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Inverse Propensity Weighting (IPW) ATE estimate.

    ATE_IPW = E[Y * T / p(X)] - E[Y * (1-T) / (1-p(X))]

    Args:
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores p(X)
        weights: Optional pre-computed IPW weights

    Returns:
        IPW ATE estimate
    """
    if weights is None:
        # Compute IPW weights
        from .propensity import compute_ipw_weights

        weights = compute_ipw_weights(T, propensity_scores, stabilize=True)

    # Weighted means
    mean_treated_weighted = np.average(Y[T == 1], weights=weights[T == 1])
    mean_control_weighted = np.average(Y[T == 0], weights=weights[T == 0])

    ate_ipw = mean_treated_weighted - mean_control_weighted

    logger.info(f"IPW ATE: ${ate_ipw:.2f}")

    return ate_ipw


def estimate_ate_aipw(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    outcome_models: tuple | None = None,
    model_type: str = "linear",
) -> tuple[float, tuple]:
    """
    Augmented Inverse Propensity Weighting (AIPW) ATE estimate.

    AIPW is doubly-robust: consistent if either propensity model OR outcome model is correct.

    AIPW formula:
    ATE = E[μ1(X) - μ0(X)]
        + E[T * (Y - μ1(X)) / p(X)]
        - E[(1-T) * (Y - μ0(X)) / (1-p(X))]

    where μ1(X) = E[Y|X,T=1] and μ0(X) = E[Y|X,T=0]

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores
        outcome_models: Tuple of (model_treated, model_control) if pre-fitted
        model_type: Type of outcome model ('linear' or 'rf')

    Returns:
        Tuple of (ATE estimate, (model_treated, model_control))
    """
    len(Y)

    # Fit outcome models if not provided
    if outcome_models is None:
        if model_type == "linear":
            model_treated = LinearRegression()
            model_control = LinearRegression()
        elif model_type == "rf":
            model_treated = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            model_control = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Fit on respective groups
        model_treated.fit(X[T == 1], Y[T == 1])
        model_control.fit(X[T == 0], Y[T == 0])

        logger.info(f"Fitted {model_type} outcome models for AIPW")
    else:
        model_treated, model_control = outcome_models

    # Predict potential outcomes
    mu1 = model_treated.predict(X)  # E[Y|X, T=1]
    mu0 = model_control.predict(X)  # E[Y|X, T=0]

    # AIPW components
    # Component 1: Outcome model predictions
    outcome_component = (mu1 - mu0).mean()

    # Component 2: IPW correction for treated
    treated_correction = (T * (Y - mu1) / propensity_scores).mean()

    # Component 3: IPW correction for control
    control_correction = ((1 - T) * (Y - mu0) / (1 - propensity_scores)).mean()

    # Combine
    ate_aipw = outcome_component + treated_correction - control_correction

    logger.info(f"AIPW ATE: ${ate_aipw:.2f}")
    logger.info(f"  Outcome component: ${outcome_component:.2f}")
    logger.info(f"  IPW correction (treated): ${treated_correction:.2f}")
    logger.info(f"  IPW correction (control): ${-control_correction:.2f}")

    return ate_aipw, (model_treated, model_control)


def bootstrap_ate_ci(
    ate_fn,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_iterations: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
    **ate_kwargs,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for ATE estimate.

    Args:
        ate_fn: Function to compute ATE (e.g., estimate_ate_ipw)
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (0-1)
        random_state: Random seed
        **ate_kwargs: Additional arguments for ate_fn

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(Y)

    # Compute point estimate
    if "X" in ate_fn.__code__.co_varnames:
        point_estimate = ate_fn(X, Y, T, **ate_kwargs)
        if isinstance(point_estimate, tuple):
            point_estimate = point_estimate[0]
    else:
        point_estimate = ate_fn(Y, T, **ate_kwargs)

    logger.info(f"Computing bootstrap CI with {n_iterations} iterations...")

    # Bootstrap
    bootstrap_estimates = []
    for i in range(n_iterations):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx] if X is not None else None
        Y_boot = Y[idx]
        T_boot = T[idx]

        # Re-fit propensity model if needed
        boot_kwargs = ate_kwargs.copy()
        if "propensity_scores" in ate_kwargs:
            # For IPW/AIPW, refit propensity model on bootstrap sample
            from .propensity import PropensityModel

            prop_model = PropensityModel(model_type="logistic")
            prop_model.fit(X_boot, T_boot)
            boot_kwargs["propensity_scores"] = prop_model.propensity_scores_

        # Compute ATE on bootstrap sample
        try:
            if "X" in ate_fn.__code__.co_varnames:
                ate_boot = ate_fn(X_boot, Y_boot, T_boot, **boot_kwargs)
                if isinstance(ate_boot, tuple):
                    ate_boot = ate_boot[0]
            else:
                ate_boot = ate_fn(Y_boot, T_boot, **boot_kwargs)

            bootstrap_estimates.append(ate_boot)
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed: {e}")
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    logger.info(
        f"Bootstrap CI ({confidence_level:.0%}): "
        f"${point_estimate:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return point_estimate, lower_bound, upper_bound


def estimate_all_ate_methods(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    config: dict,
) -> dict[str, dict[str, float]]:
    """
    Estimate ATE using all methods and compute confidence intervals.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores
        config: Configuration dictionary

    Returns:
        Dictionary of results for each method
    """
    results = {}

    ate_config = config.get("ate", {})
    n_boot = ate_config.get("bootstrap_iterations", 500)
    conf_level = ate_config.get("bootstrap_confidence_level", 0.95)
    random_state = config.get("random_state", 42)

    logger.info("=" * 60)
    logger.info("AVERAGE TREATMENT EFFECT ESTIMATION")
    logger.info("=" * 60)

    # 1. Naive estimate
    logger.info("\n[1/4] Naive difference in means...")
    ate_naive, ci_lower_naive, ci_upper_naive = bootstrap_ate_ci(
        estimate_ate_naive,
        None,
        Y,
        T,
        n_iterations=n_boot,
        confidence_level=conf_level,
        random_state=random_state,
    )
    results["naive"] = {
        "ate": ate_naive,
        "ci_lower": ci_lower_naive,
        "ci_upper": ci_upper_naive,
    }

    # 2. IPW estimate
    logger.info("\n[2/4] Inverse Propensity Weighting (IPW)...")
    from .propensity import compute_ipw_weights

    compute_ipw_weights(
        T,
        propensity_scores,
        stabilize=ate_config.get("ipw_stabilize", True),
        trim_percentiles=tuple(config.get("propensity", {}).get("trim_percentiles", [1, 99])),
    )

    ate_ipw, ci_lower_ipw, ci_upper_ipw = bootstrap_ate_ci(
        estimate_ate_ipw,
        None,
        Y,
        T,
        n_iterations=n_boot,
        confidence_level=conf_level,
        random_state=random_state,
        propensity_scores=propensity_scores,
    )
    results["ipw"] = {
        "ate": ate_ipw,
        "ci_lower": ci_lower_ipw,
        "ci_upper": ci_upper_ipw,
    }

    # 3. AIPW with linear outcome models
    logger.info("\n[3/4] AIPW with linear outcome models...")
    ate_aipw_linear, ci_lower_aipw_linear, ci_upper_aipw_linear = bootstrap_ate_ci(
        estimate_ate_aipw,
        X,
        Y,
        T,
        n_iterations=n_boot,
        confidence_level=conf_level,
        random_state=random_state,
        propensity_scores=propensity_scores,
        model_type="linear",
    )
    results["aipw_linear"] = {
        "ate": ate_aipw_linear,
        "ci_lower": ci_lower_aipw_linear,
        "ci_upper": ci_upper_aipw_linear,
    }

    # 4. AIPW with flexible outcome models (Random Forest)
    logger.info("\n[4/4] AIPW with Random Forest outcome models...")
    ate_aipw_rf, ci_lower_aipw_rf, ci_upper_aipw_rf = bootstrap_ate_ci(
        estimate_ate_aipw,
        X,
        Y,
        T,
        n_iterations=n_boot,
        confidence_level=conf_level,
        random_state=random_state,
        propensity_scores=propensity_scores,
        model_type="rf",
    )
    results["aipw_rf"] = {
        "ate": ate_aipw_rf,
        "ci_lower": ci_lower_aipw_rf,
        "ci_upper": ci_upper_aipw_rf,
    }

    logger.info("\n" + "=" * 60)
    logger.info("ATE ESTIMATION COMPLETE")
    logger.info("=" * 60)

    return results
