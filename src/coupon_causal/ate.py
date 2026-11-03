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
    rf_params: dict | None = None,
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
        rf_params: Optional RandomForestRegressor hyperparameters for model_type='rf'

    Returns:
        Tuple of (ATE estimate, (model_treated, model_control))
    """
    # Fit outcome models if not provided
    if outcome_models is None:
        if model_type == "linear":
            model_treated = LinearRegression()
            model_control = LinearRegression()
        elif model_type == "rf":
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 1,
                **(rf_params or {}),
            }
            model_treated = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model_control = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
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


def _refit_propensity(
    X: np.ndarray,
    T: np.ndarray,
    model_type: str,
    propensity_config: dict | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Refit the propensity model on a (bootstrap) sample.

    ``model_type`` must match the model that produced the point estimate's
    propensity scores, otherwise the bootstrap distribution is centred on a
    different estimator and the interval can miss the point estimate entirely.
    """
    from .propensity import PropensityModel

    propensity_config = propensity_config or {}
    clip_bounds = tuple(propensity_config.get("clip_bounds", [0.01, 0.99]))
    model_cfgs = propensity_config.get("models", {})

    def _fit(kind: str) -> np.ndarray:
        model = PropensityModel(
            model_type=kind,
            clip_bounds=clip_bounds,
            config={**model_cfgs.get(kind, {}), "random_state": random_state},
        )
        model.fit(X, T)
        return model.propensity_scores_

    if model_type in ("logistic", "gbm"):
        return _fit(model_type)
    if model_type == "ensemble":
        return (_fit("logistic") + _fit("gbm")) / 2
    raise ValueError(f"Unknown propensity model_type: {model_type}")


def bootstrap_ate_ci(
    ate_fn,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_iterations: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
    propensity_model_type: str = "logistic",
    propensity_config: dict | None = None,
    **ate_kwargs,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for an ATE estimate.

    For IPW/AIPW (i.e. when ``propensity_scores`` is passed in ``ate_kwargs``)
    the propensity model is refit on every bootstrap sample so the interval
    reflects propensity-estimation uncertainty too.

    Args:
        ate_fn: Function to compute ATE (e.g., estimate_ate_ipw)
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (0-1)
        random_state: Random seed
        propensity_model_type: Model refit on each bootstrap sample
            ('logistic', 'gbm' or 'ensemble'). Must match the model used for
            the point estimate's ``propensity_scores``.
        propensity_config: Propensity section of the config (clip bounds, model params)
        **ate_kwargs: Additional arguments for ate_fn

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    n = len(Y)
    needs_X = "X" in ate_fn.__code__.co_varnames

    # Compute point estimate
    if needs_X:
        point_estimate = ate_fn(X, Y, T, **ate_kwargs)
        if isinstance(point_estimate, tuple):
            point_estimate = point_estimate[0]
    else:
        point_estimate = ate_fn(Y, T, **ate_kwargs)

    logger.info(f"Computing bootstrap CI with {n_iterations} iterations...")

    refit_propensity = "propensity_scores" in ate_kwargs
    if refit_propensity and (X is None or np.isnan(X).any()):
        logger.warning("Cannot refit propensity in bootstrap: X is missing or contains NaN")
        return point_estimate, np.nan, np.nan

    # The estimators log their result at INFO on every call; silence that
    # inside the loop so a 500-draw bootstrap does not print 500 lines.
    package_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
    previous_level = package_logger.level
    package_logger.setLevel(max(previous_level, logging.WARNING))

    bootstrap_estimates = []
    try:
        for i in range(n_iterations):
            # Resample with replacement
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X[idx] if X is not None else None
            Y_boot = Y[idx]
            T_boot = T[idx]

            if len(np.unique(T_boot)) < 2:
                logger.warning(f"Bootstrap iteration {i} skipped: only one treatment group")
                continue

            try:
                boot_kwargs = ate_kwargs.copy()
                if refit_propensity:
                    boot_kwargs["propensity_scores"] = _refit_propensity(
                        X_boot, T_boot, propensity_model_type, propensity_config, random_state
                    )

                if needs_X:
                    ate_boot = ate_fn(X_boot, Y_boot, T_boot, **boot_kwargs)
                    if isinstance(ate_boot, tuple):
                        ate_boot = ate_boot[0]
                else:
                    ate_boot = ate_fn(Y_boot, T_boot, **boot_kwargs)

                bootstrap_estimates.append(ate_boot)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                continue
    finally:
        package_logger.setLevel(previous_level)

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Check if we have enough successful bootstrap iterations
    min_successful_iterations = max(10, int(n_iterations * 0.1))  # At least 10 or 10% of iterations
    if len(bootstrap_estimates) < min_successful_iterations:
        logger.warning(
            f"Only {len(bootstrap_estimates)}/{n_iterations} bootstrap iterations succeeded. "
            f"Returning point estimate without CI."
        )
        # Return point estimate with NaN bounds to indicate failure
        return point_estimate, np.nan, np.nan

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    if not lower_bound <= point_estimate <= upper_bound:
        logger.warning(
            f"Bootstrap interval [{lower_bound:.2f}, {upper_bound:.2f}] does not contain the "
            f"point estimate {point_estimate:.2f}; the bootstrap is refitting a different "
            f"propensity model ({propensity_model_type}) than the point estimate used."
        )

    logger.info(
        f"Bootstrap CI ({confidence_level:.0%}) from {len(bootstrap_estimates)} iterations: "
        f"${point_estimate:.2f} [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return point_estimate, lower_bound, upper_bound


def estimate_all_ate_methods(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    config: dict,
    propensity_model_type: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Estimate ATE using all methods and compute confidence intervals.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores used for the point estimates
        config: Configuration dictionary
        propensity_model_type: Which propensity model produced ``propensity_scores``
            ('logistic', 'gbm' or 'ensemble'); the bootstrap refits the same one.
            Defaults to ``config['propensity']['ate_model']`` or 'logistic'.

    Returns:
        Dictionary of results for each method
    """
    results = {}

    ate_config = config.get("ate", {})
    propensity_config = config.get("propensity", {})
    n_boot = ate_config.get("bootstrap_iterations", 500)
    conf_level = ate_config.get("bootstrap_confidence_level", 0.95)
    random_state = config.get("random_state", 42)
    outcome_models = ate_config.get("outcome_models", ["linear", "rf"])
    rf_params = ate_config.get("rf_outcome_model", {})
    if propensity_model_type is None:
        propensity_model_type = propensity_config.get("ate_model", "logistic")
    n_methods = 2 + len(outcome_models)
    boot_common = {
        "n_iterations": n_boot,
        "confidence_level": conf_level,
        "random_state": random_state,
        "propensity_model_type": propensity_model_type,
        "propensity_config": propensity_config,
    }

    logger.info("=" * 60)
    logger.info("AVERAGE TREATMENT EFFECT ESTIMATION")
    logger.info("=" * 60)

    # 1. Naive estimate
    logger.info(f"\n[1/{n_methods}] Naive difference in means...")
    ate_naive, ci_lower_naive, ci_upper_naive = bootstrap_ate_ci(
        estimate_ate_naive, None, Y, T, **boot_common
    )
    results["naive"] = {
        "ate": ate_naive,
        "ci_lower": ci_lower_naive,
        "ci_upper": ci_upper_naive,
    }

    # 2. IPW estimate
    logger.info(f"\n[2/{n_methods}] Inverse Propensity Weighting (IPW)...")
    ate_ipw, ci_lower_ipw, ci_upper_ipw = bootstrap_ate_ci(
        estimate_ate_ipw,
        X,  # Pass X so propensity model can be refit in bootstrap
        Y,
        T,
        propensity_scores=propensity_scores,
        **boot_common,
    )
    results["ipw"] = {
        "ate": ate_ipw,
        "ci_lower": ci_lower_ipw,
        "ci_upper": ci_upper_ipw,
    }

    # 3+. AIPW (doubly robust) with each configured outcome model
    labels = {"linear": "linear", "rf": "Random Forest"}
    for i, model_type in enumerate(outcome_models, start=3):
        logger.info(f"\n[{i}/{n_methods}] AIPW with {labels.get(model_type, model_type)} outcome models...")
        ate_aipw, ci_lower_aipw, ci_upper_aipw = bootstrap_ate_ci(
            estimate_ate_aipw,
            X,
            Y,
            T,
            propensity_scores=propensity_scores,
            model_type=model_type,
            rf_params=rf_params,
            **boot_common,
        )
        results[f"aipw_{model_type}"] = {
            "ate": ate_aipw,
            "ci_lower": ci_lower_aipw,
            "ci_upper": ci_upper_aipw,
        }

    logger.info("\n" + "=" * 60)
    logger.info("ATE ESTIMATION COMPLETE")
    logger.info("=" * 60)

    return results
