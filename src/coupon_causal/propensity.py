# path: src/coupon_causal/propensity.py
"""
Propensity score modeling and diagnostics for causal inference.
"""

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


class PropensityModel:
    """
    Propensity score estimation with multiple model options.

    Supports:
    - Logistic regression (parametric)
    - Gradient boosting (flexible, non-parametric)
    - Calibration diagnostics
    - Propensity score trimming
    """

    def __init__(
        self,
        model_type: str = "logistic",
        clip_bounds: tuple[float, float] = (0.01, 0.99),
        config: dict | None = None,
    ):
        """
        Initialize propensity model.

        Args:
            model_type: Type of model ('logistic' or 'gbm')
            clip_bounds: (min, max) bounds for clipping propensity scores
            config: Model-specific configuration
        """
        self.model_type = model_type
        self.clip_bounds = clip_bounds
        self.config = config or {}

        self.model = None
        self.propensity_scores_ = None
        self.calibration_curve_ = None

    def fit(self, X: np.ndarray, T: np.ndarray) -> "PropensityModel":
        """
        Fit propensity score model.

        Args:
            X: Feature matrix (n_samples, n_features)
            T: Treatment indicator (n_samples,)

        Returns:
            Self (fitted)
        """
        logger.info(f"Fitting {self.model_type} propensity model...")

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=self.config.get("max_iter", 1000),
                C=self.config.get("C", 1.0),
                random_state=self.config.get("random_state", 42),
            )
        elif self.model_type == "gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.get("n_estimators", 100),
                max_depth=self.config.get("max_depth", 5),
                learning_rate=self.config.get("learning_rate", 0.1),
                min_samples_leaf=self.config.get("min_samples_leaf", 50),
                random_state=self.config.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X, T)

        # Compute propensity scores with cross-validation to avoid overfitting
        self.propensity_scores_ = self._compute_propensity_scores(X, T)

        logger.info(f"Propensity model fitted. AUC: {self._compute_auc(T):.3f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict propensity scores.

        Args:
            X: Feature matrix

        Returns:
            Propensity scores (clipped to bounds)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        propensity_raw = self.model.predict_proba(X)[:, 1]
        propensity_clipped = np.clip(propensity_raw, self.clip_bounds[0], self.clip_bounds[1])

        return propensity_clipped

    def _compute_propensity_scores(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute propensity scores with cross-validation to prevent overfitting.

        Args:
            X: Feature matrix
            T: Treatment indicator

        Returns:
            Cross-validated propensity scores
        """
        # Use 5-fold cross-validation to get out-of-fold predictions
        propensity_cv = cross_val_predict(
            self.model, X, T, cv=5, method="predict_proba"
        )[:, 1]

        # Clip to bounds
        propensity_clipped = np.clip(propensity_cv, self.clip_bounds[0], self.clip_bounds[1])

        return propensity_clipped

    def _compute_auc(self, T: np.ndarray) -> float:
        """Compute AUC for propensity model."""
        if self.propensity_scores_ is None:
            return 0.0
        return roc_auc_score(T, self.propensity_scores_)

    def compute_diagnostics(
        self, T: np.ndarray, n_bins: int = 10
    ) -> dict[str, float]:
        """
        Compute propensity model diagnostics.

        Args:
            T: Treatment indicator
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary of diagnostic metrics
        """
        if self.propensity_scores_ is None:
            raise ValueError("Model not fitted yet")

        # AUC and average precision
        auc = roc_auc_score(T, self.propensity_scores_)
        avg_precision = average_precision_score(T, self.propensity_scores_)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            T, self.propensity_scores_, n_bins=n_bins, strategy="uniform"
        )
        self.calibration_curve_ = (prob_true, prob_pred)

        # Compute calibration error (Brier score)
        brier_score = np.mean((T - self.propensity_scores_) ** 2)

        # Overlap: check if there's sufficient overlap in propensity scores
        p_treated = self.propensity_scores_[T == 1]
        p_control = self.propensity_scores_[T == 0]

        overlap_min = max(p_treated.min(), p_control.min())
        overlap_max = min(p_treated.max(), p_control.max())
        overlap_range = overlap_max - overlap_min

        diagnostics = {
            "auc": auc,
            "average_precision": avg_precision,
            "brier_score": brier_score,
            "overlap_range": overlap_range,
            "propensity_mean": self.propensity_scores_.mean(),
            "propensity_std": self.propensity_scores_.std(),
        }

        return diagnostics


def compute_ipw_weights(
    T: np.ndarray,
    propensity_scores: np.ndarray,
    stabilize: bool = True,
    trim_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Compute inverse propensity weights (IPW).

    IPW formula:
    - For treated (T=1): w = 1 / p(X)
    - For control (T=0): w = 1 / (1 - p(X))

    Stabilized IPW:
    - For treated (T=1): w = P(T=1) / p(X)
    - For control (T=0): w = P(T=0) / (1 - p(X))

    Args:
        T: Treatment indicator (n_samples,)
        propensity_scores: Propensity scores (n_samples,)
        stabilize: Whether to use stabilized weights
        trim_percentiles: (lower, upper) percentiles for trimming extreme weights

    Returns:
        IPW weights (n_samples,)
    """
    weights = np.zeros_like(propensity_scores)

    if stabilize:
        # Stabilized weights
        treatment_prob = T.mean()
        weights[T == 1] = treatment_prob / propensity_scores[T == 1]
        weights[T == 0] = (1 - treatment_prob) / (1 - propensity_scores[T == 0])
    else:
        # Unstabilized weights
        weights[T == 1] = 1.0 / propensity_scores[T == 1]
        weights[T == 0] = 1.0 / (1 - propensity_scores[T == 0])

    # Trim extreme weights if requested
    if trim_percentiles is not None:
        lower, upper = trim_percentiles
        lower_bound = np.percentile(weights, lower)
        upper_bound = np.percentile(weights, upper)
        weights = np.clip(weights, lower_bound, upper_bound)

        logger.info(f"Trimmed weights to [{lower_bound:.3f}, {upper_bound:.3f}]")

    logger.info(f"IPW weights - mean: {weights.mean():.3f}, std: {weights.std():.3f}")
    logger.info(f"IPW weights - range: [{weights.min():.3f}, {weights.max():.3f}]")

    return weights


def fit_propensity_models(
    X: np.ndarray,
    T: np.ndarray,
    config: dict,
) -> tuple[PropensityModel, PropensityModel, np.ndarray]:
    """
    Fit multiple propensity models and return ensemble.

    Args:
        X: Feature matrix
        T: Treatment indicator
        config: Configuration dictionary

    Returns:
        Tuple of (logistic_model, gbm_model, ensemble_propensity_scores)
    """
    propensity_config = config.get("propensity", {})
    clip_bounds = tuple(propensity_config.get("clip_bounds", [0.01, 0.99]))

    # Fit logistic regression model
    logger.info("Fitting logistic propensity model...")
    logistic_model = PropensityModel(
        model_type="logistic",
        clip_bounds=clip_bounds,
        config={**propensity_config.get("models", {}).get("logistic", {}),
                "random_state": config.get("random_state", 42)},
    )
    logistic_model.fit(X, T)

    # Fit GBM model
    logger.info("Fitting GBM propensity model...")
    gbm_model = PropensityModel(
        model_type="gbm",
        clip_bounds=clip_bounds,
        config={**propensity_config.get("models", {}).get("gbm", {}),
                "random_state": config.get("random_state", 42)},
    )
    gbm_model.fit(X, T)

    # Ensemble: average of both models
    ensemble_scores = (
        logistic_model.propensity_scores_ + gbm_model.propensity_scores_
    ) / 2

    logger.info("Propensity model ensemble created")

    return logistic_model, gbm_model, ensemble_scores


def assess_overlap(
    propensity_scores: np.ndarray,
    T: np.ndarray,
    threshold: float = 0.1,
) -> dict[str, float]:
    """
    Assess overlap in propensity score distributions.

    Good overlap is critical for valid causal inference.
    Lack of overlap means some treatment groups have no comparable controls.

    Args:
        propensity_scores: Propensity scores
        T: Treatment indicator
        threshold: Threshold for defining poor overlap

    Returns:
        Dictionary of overlap metrics
    """
    p_treated = propensity_scores[T == 1]
    p_control = propensity_scores[T == 0]

    # Common support range
    common_min = max(p_treated.min(), p_control.min())
    common_max = min(p_treated.max(), p_control.max())
    common_range = common_max - common_min

    # Proportion of units in common support
    in_support_treated = np.mean((p_treated >= common_min) & (p_treated <= common_max))
    in_support_control = np.mean((p_control >= common_min) & (p_control <= common_max))

    # Proportion with poor overlap (extreme propensity scores)
    poor_overlap_treated = np.mean(p_treated > (1 - threshold))
    poor_overlap_control = np.mean(p_control < threshold)

    overlap_metrics = {
        "common_support_min": common_min,
        "common_support_max": common_max,
        "common_support_range": common_range,
        "treated_in_support": in_support_treated,
        "control_in_support": in_support_control,
        "treated_poor_overlap": poor_overlap_treated,
        "control_poor_overlap": poor_overlap_control,
    }

    logger.info(f"Overlap assessment - common support: [{common_min:.3f}, {common_max:.3f}]")
    logger.info(f"Treated in support: {in_support_treated:.1%}")
    logger.info(f"Control in support: {in_support_control:.1%}")

    return overlap_metrics
