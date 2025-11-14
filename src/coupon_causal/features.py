# path: src/coupon_causal/features.py
"""
Feature engineering for causal impact analysis.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for causal analysis.

    Handles:
    - Categorical encoding
    - Continuous feature scaling
    - Feature selection and validation
    - Prevention of data leakage
    """

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        scale_features: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            categorical_features: List of categorical feature names
            continuous_features: List of continuous feature names
            scale_features: Whether to scale continuous features
        """
        self.categorical_features = categorical_features or []
        self.continuous_features = continuous_features or []
        self.scale_features = scale_features

        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit feature transformers on training data.

        Args:
            df: Training DataFrame

        Returns:
            Self (fitted)
        """
        logger.info("Fitting feature transformers...")

        # Fit scaler for continuous features
        if self.continuous_features and self.scale_features:
            self.scaler = StandardScaler()
            continuous_data = df[self.continuous_features].values
            self.scaler.fit(continuous_data)
            logger.info(f"Fitted scaler on {len(self.continuous_features)} continuous features")

        # Fit encoder for categorical features
        if self.categorical_features:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            categorical_data = df[self.categorical_features].values
            self.encoder.fit(categorical_data)
            logger.info(
                f"Fitted encoder on {len(self.categorical_features)} categorical features"
            )

        # Store feature names
        self._build_feature_names()

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features.

        Args:
            df: Input DataFrame

        Returns:
            Transformed feature matrix (2D numpy array)
        """
        transformed_parts = []

        # Transform continuous features
        if self.continuous_features:
            continuous_data = df[self.continuous_features].values
            if self.scale_features and self.scaler is not None:
                continuous_transformed = self.scaler.transform(continuous_data)
            else:
                continuous_transformed = continuous_data
            transformed_parts.append(continuous_transformed)

        # Transform categorical features
        if self.categorical_features and self.encoder is not None:
            categorical_data = df[self.categorical_features].values
            categorical_transformed = self.encoder.transform(categorical_data)
            transformed_parts.append(categorical_transformed)

        # Concatenate all features
        if transformed_parts:
            X = np.hstack(transformed_parts)
        else:
            X = np.array([]).reshape(len(df), 0)

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit transformers and transform features in one step.

        Args:
            df: Input DataFrame

        Returns:
            Transformed feature matrix
        """
        return self.fit(df).transform(df)

    def _build_feature_names(self) -> None:
        """Build list of output feature names after transformation."""
        feature_names = []

        # Add continuous feature names
        if self.continuous_features:
            feature_names.extend(self.continuous_features)

        # Add categorical feature names
        if self.categorical_features and self.encoder is not None:
            # Get feature names from encoder
            cat_feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)

        self.feature_names_out_ = feature_names

    def get_feature_names(self) -> list[str]:
        """
        Get names of output features.

        Returns:
            List of feature names
        """
        if self.feature_names_out_ is None:
            raise ValueError("Feature engineer not fitted yet")
        return self.feature_names_out_


def prepare_features(
    df: pd.DataFrame,
    feature_config: dict,
    fit: bool = True,
    feature_engineer: FeatureEngineer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FeatureEngineer]:
    """
    Prepare features for causal estimation.

    Args:
        df: Input DataFrame with treatment and outcome
        feature_config: Configuration with categorical and continuous feature lists
        fit: Whether to fit the feature engineer (True for train, False for test)
        feature_engineer: Pre-fitted feature engineer (for test set)

    Returns:
        Tuple of (X, T, Y, feature_engineer)
        - X: Feature matrix (n_samples, n_features)
        - T: Treatment vector (n_samples,)
        - Y: Outcome vector (n_samples,)
        - feature_engineer: Fitted FeatureEngineer instance
    """
    logger.info("Preparing features for causal analysis...")

    # Extract treatment and outcome
    if "treatment" not in df.columns:
        raise ValueError("DataFrame must contain 'treatment' column")
    if "outcome" not in df.columns:
        raise ValueError("DataFrame must contain 'outcome' column")

    T = df["treatment"].values
    Y = df["outcome"].values

    # Initialize or use provided feature engineer
    if feature_engineer is None:
        feature_engineer = FeatureEngineer(
            categorical_features=feature_config.get("categorical_features", []),
            continuous_features=feature_config.get("continuous_features", []),
            scale_features=True,
        )

    # Transform features
    if fit:
        X = feature_engineer.fit_transform(df)
        logger.info(f"Fitted and transformed features: {X.shape}")
    else:
        X = feature_engineer.transform(df)
        logger.info(f"Transformed features: {X.shape}")

    # Validate shapes
    assert X.shape[0] == len(T) == len(Y), "Feature, treatment, and outcome lengths must match"

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Treatment distribution: {np.bincount(T.astype(int))}")
    logger.info(f"Outcome range: [{Y.min():.2f}, {Y.max():.2f}]")

    return X, T, Y, feature_engineer


def check_feature_leakage(df: pd.DataFrame, feature_names: list[str]) -> list[str]:
    """
    Check for potential data leakage in features.

    Args:
        df: Input DataFrame
        feature_names: List of feature names to check

    Returns:
        List of potentially leaky features
    """
    leakage_patterns = [
        "outcome",
        "post_",
        "after_",
        "response",
        "purchase_post",
        "spend_post",
        "revenue_post",
    ]

    leaky_features = []
    for feature in feature_names:
        feature_lower = feature.lower()
        if any(pattern in feature_lower for pattern in leakage_patterns):
            leaky_features.append(feature)

    if leaky_features:
        logger.warning(f"Potential leakage detected in features: {leaky_features}")

    return leaky_features


def create_interaction_features(
    df: pd.DataFrame,
    interaction_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Create interaction features for heterogeneity analysis.

    Args:
        df: Input DataFrame
        interaction_pairs: List of (feature1, feature2) tuples to interact

    Returns:
        DataFrame with additional interaction features
    """
    df_with_interactions = df.copy()

    if interaction_pairs is None:
        # Default interactions for retail coupon analysis
        interaction_pairs = [
            ("loyalty_score", "price_sensitivity"),
            ("recency_days", "frequency_purchases"),
            ("monetary_value", "avg_basket_size"),
        ]

    for feat1, feat2 in interaction_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_with_interactions[interaction_name] = df[feat1] * df[feat2]
            logger.info(f"Created interaction feature: {interaction_name}")

    return df_with_interactions


def get_feature_importance_names(
    feature_engineer: FeatureEngineer,
    importance_values: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Get top-k feature importances with names.

    Args:
        feature_engineer: Fitted FeatureEngineer instance
        importance_values: Array of importance values
        top_k: Number of top features to return

    Returns:
        DataFrame with feature names and importances, sorted by importance
    """
    feature_names = feature_engineer.get_feature_names()

    if len(importance_values) != len(feature_names):
        raise ValueError(
            f"Importance values ({len(importance_values)}) and feature names "
            f"({len(feature_names)}) length mismatch"
        )

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_k)
