# path: src/coupon_causal/cate.py
"""
Conditional Average Treatment Effect (CATE) estimation.

Implements multiple methods for estimating heterogeneous treatment effects:
- DRLearner (EconML)
- Orthogonal Random Forest (EconML)
- X-Learner
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class XLearner:
    """
    X-Learner for CATE estimation.

    The X-Learner:
    1. Fits outcome models μ0(X) and μ1(X) separately on control and treated
    2. Imputes treatment effects:
       - For treated: τ1(X) = Y - μ0(X)
       - For control: τ0(X) = μ1(X) - Y
    3. Fits CATE models on imputed effects
    4. Combines using propensity scores: τ(X) = p(X) * τ0(X) + (1-p(X)) * τ1(X)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        """
        Initialize X-Learner.

        Args:
            n_estimators: Number of trees for RF models
            max_depth: Maximum depth of trees
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.mu0_model = None  # E[Y|X, T=0]
        self.mu1_model = None  # E[Y|X, T=1]
        self.tau0_model = None  # E[τ|X, T=0]
        self.tau1_model = None  # E[τ|X, T=1]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        propensity_scores: np.ndarray | None = None,
    ) -> "XLearner":
        """
        Fit X-Learner.

        Args:
            X: Feature matrix
            Y: Outcome vector
            T: Treatment indicator
            propensity_scores: Optional propensity scores for weighting

        Returns:
            Self (fitted)
        """
        logger.info("Fitting X-Learner...")

        # Stage 1: Fit outcome models
        self.mu0_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.mu1_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.mu0_model.fit(X[T == 0], Y[T == 0])
        self.mu1_model.fit(X[T == 1], Y[T == 1])

        # Stage 2: Impute individual treatment effects
        # For treated units: τ1(X) = Y - μ0(X)
        mu0_on_treated = self.mu0_model.predict(X[T == 1])
        tau_treated = Y[T == 1] - mu0_on_treated

        # For control units: τ0(X) = μ1(X) - Y
        mu1_on_control = self.mu1_model.predict(X[T == 0])
        tau_control = mu1_on_control - Y[T == 0]

        # Stage 3: Fit CATE models on imputed effects
        self.tau0_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.tau1_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.tau0_model.fit(X[T == 0], tau_control)
        self.tau1_model.fit(X[T == 1], tau_treated)

        logger.info("X-Learner fitted")

        return self

    def predict(
        self,
        X: np.ndarray,
        propensity_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict CATE.

        Args:
            X: Feature matrix
            propensity_scores: Propensity scores for weighted combination

        Returns:
            CATE predictions
        """
        tau0_pred = self.tau0_model.predict(X)
        tau1_pred = self.tau1_model.predict(X)

        if propensity_scores is not None:
            # Weighted combination
            cate = propensity_scores * tau0_pred + (1 - propensity_scores) * tau1_pred
        else:
            # Simple average
            cate = (tau0_pred + tau1_pred) / 2

        return cate


def fit_dr_learner(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, object]:
    """
    Fit EconML DRLearner for CATE estimation.

    DRLearner uses doubly-robust scoring with cross-fitting to reduce bias.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        config: Configuration dictionary

    Returns:
        Tuple of (CATE predictions, fitted model)
    """
    try:
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    except ImportError:
        logger.error("EconML not installed. Install with: pip install econml")
        raise

    logger.info("Fitting DRLearner for CATE estimation...")

    cate_config = config.get("cate", {})
    dr_config = cate_config.get("dr_learner", {})
    cv_folds = cate_config.get("cross_fitting_folds", 5)

    # Outcome and propensity models for DRLearner
    model_regression_config = dr_config.get("model_regression", {})
    model_regression = GradientBoostingRegressor(
        n_estimators=model_regression_config.get("n_estimators", 100),
        max_depth=model_regression_config.get("max_depth", 6),
        min_samples_leaf=model_regression_config.get("min_samples_leaf", 20),
        random_state=config.get("random_state", 42),
    )

    model_propensity_config = dr_config.get("model_propensity", {})
    model_propensity = GradientBoostingClassifier(
        n_estimators=model_propensity_config.get("n_estimators", 100),
        max_depth=model_propensity_config.get("max_depth", 4),
        random_state=config.get("random_state", 42),
    )

    # Final CATE model (default to Random Forest)
    model_final = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=config.get("random_state", 42),
        n_jobs=-1,
    )

    # Initialize DRLearner
    dr_learner = DRLearner(
        model_regression=model_regression,
        model_propensity=model_propensity,
        model_final=model_final,
        cv=cv_folds,
        random_state=config.get("random_state", 42),
    )

    # Fit with cross-fitting
    dr_learner.fit(Y, T, X=X)

    # Predict CATE
    cate_pred = dr_learner.effect(X)

    logger.info(f"DRLearner CATE - mean: ${cate_pred.mean():.2f}, std: ${cate_pred.std():.2f}")

    return cate_pred, dr_learner


def fit_orthogonal_forest(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, object, np.ndarray]:
    """
    Fit EconML CausalForest (Orthogonal Random Forest) for CATE estimation.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        config: Configuration dictionary

    Returns:
        Tuple of (CATE predictions, fitted model, feature importances)
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    except ImportError:
        logger.error("EconML not installed. Install with: pip install econml")
        raise

    logger.info("Fitting Orthogonal Random Forest for CATE estimation...")

    cate_config = config.get("cate", {})
    orf_config = cate_config.get("orthogonal_forest", {})
    cv_folds = cate_config.get("cross_fitting_folds", 5)

    # Nuisance models
    model_y = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=config.get("random_state", 42),
    )
    model_t = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=config.get("random_state", 42),
    )

    # Causal Forest
    causal_forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=orf_config.get("n_trees", 500),
        min_samples_leaf=orf_config.get("min_leaf_size", 10),
        max_depth=orf_config.get("max_depth", 10),
        cv=cv_folds,
        random_state=config.get("random_state", 42),
        n_jobs=-1,
    )

    # Fit
    causal_forest.fit(Y, T, X=X)

    # Predict CATE
    cate_pred = causal_forest.effect(X)

    # Feature importances
    try:
        # Try to get feature importances from the causal forest
        feature_importances = causal_forest.feature_importances()
    except:
        # Fallback: compute as variance of CATE predictions per feature
        feature_importances = np.zeros(X.shape[1])
        logger.warning("Could not extract feature importances from model")

    logger.info(
        f"Orthogonal Forest CATE - mean: ${cate_pred.mean():.2f}, std: ${cate_pred.std():.2f}"
    )

    return cate_pred, causal_forest, feature_importances


def fit_x_learner(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, XLearner]:
    """
    Fit X-Learner for CATE estimation.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores
        config: Configuration dictionary

    Returns:
        Tuple of (CATE predictions, fitted model)
    """
    logger.info("Fitting X-Learner for CATE estimation...")

    cate_config = config.get("cate", {})
    xl_config = cate_config.get("x_learner", {})

    xl_model = XLearner(
        n_estimators=xl_config.get("n_estimators", 100),
        max_depth=xl_config.get("max_depth", 6),
        random_state=config.get("random_state", 42),
    )

    xl_model.fit(X, Y, T, propensity_scores)
    cate_pred = xl_model.predict(X, propensity_scores)

    logger.info(f"X-Learner CATE - mean: ${cate_pred.mean():.2f}, std: ${cate_pred.std():.2f}")

    return cate_pred, xl_model


def estimate_all_cate_methods(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    propensity_scores: np.ndarray,
    config: dict,
    feature_names: list | None = None,
) -> dict[str, dict]:
    """
    Estimate CATE using all available methods.

    Args:
        X: Feature matrix
        Y: Outcome vector
        T: Treatment indicator
        propensity_scores: Propensity scores
        config: Configuration dictionary
        feature_names: Optional list of feature names

    Returns:
        Dictionary of results for each method
    """
    results = {}

    logger.info("=" * 60)
    logger.info("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) ESTIMATION")
    logger.info("=" * 60)

    # 1. DRLearner
    logger.info("\n[1/3] Fitting DRLearner...")
    try:
        cate_dr, model_dr = fit_dr_learner(X, Y, T, config)
        results["dr_learner"] = {
            "cate": cate_dr,
            "model": model_dr,
            "mean": cate_dr.mean(),
            "std": cate_dr.std(),
            "min": cate_dr.min(),
            "max": cate_dr.max(),
        }
    except Exception as e:
        logger.error(f"DRLearner failed: {e}")
        results["dr_learner"] = None

    # 2. Orthogonal Forest
    logger.info("\n[2/3] Fitting Orthogonal Random Forest...")
    try:
        cate_orf, model_orf, feat_imp_orf = fit_orthogonal_forest(X, Y, T, config)
        results["orthogonal_forest"] = {
            "cate": cate_orf,
            "model": model_orf,
            "feature_importances": feat_imp_orf,
            "mean": cate_orf.mean(),
            "std": cate_orf.std(),
            "min": cate_orf.min(),
            "max": cate_orf.max(),
        }
    except Exception as e:
        logger.error(f"Orthogonal Forest failed: {e}")
        results["orthogonal_forest"] = None

    # 3. X-Learner
    logger.info("\n[3/3] Fitting X-Learner...")
    try:
        cate_xl, model_xl = fit_x_learner(X, Y, T, propensity_scores, config)
        results["x_learner"] = {
            "cate": cate_xl,
            "model": model_xl,
            "mean": cate_xl.mean(),
            "std": cate_xl.std(),
            "min": cate_xl.min(),
            "max": cate_xl.max(),
        }
    except Exception as e:
        logger.error(f"X-Learner failed: {e}")
        results["x_learner"] = None

    logger.info("\n" + "=" * 60)
    logger.info("CATE ESTIMATION COMPLETE")
    logger.info("=" * 60)

    # Summary
    for method_name, method_results in results.items():
        if method_results is not None:
            logger.info(
                f"{method_name}: CATE mean=${method_results['mean']:.2f}, "
                f"std=${method_results['std']:.2f}"
            )

    return results


def compute_cate_intervals(
    cate_model,
    X: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for CATE predictions.

    Args:
        cate_model: Fitted CATE model (EconML model with inference support)
        X: Feature matrix
        alpha: Significance level

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    try:
        # EconML models support .effect_interval()
        lb, ub = cate_model.effect_interval(X, alpha=alpha)
        return lb, ub
    except:
        logger.warning("CATE model does not support confidence intervals")
        return None, None
