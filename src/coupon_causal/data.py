# path: src/coupon_causal/data.py
"""
Data loading and generation for coupon causal impact analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_coupon_public(file_path: str) -> pd.DataFrame:
    """
    Load public coupon campaign dataset.

    Expected columns:
        - customer_id: Unique customer identifier
        - treatment: Binary indicator (1 = received coupon, 0 = control)
        - outcome: Target variable (spend or purchase indicator)
        - covariates: Pre-treatment features

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with coupon campaign data
    """
    logger.info(f"Loading public dataset from {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Public dataset not found at {file_path}. "
            "Please download a coupon dataset and place it in data/raw/, "
            "or use the synthetic data option with --data_source synth"
        )

    df = pd.read_csv(file_path)

    # Basic validation
    required_cols = ["treatment", "outcome"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. "
            "Please ensure your dataset has 'treatment' and 'outcome' columns."
        )

    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


def generate_synthetic_coupon_data(
    n_samples: int = 10000,
    treatment_rate: float = 0.3,
    noise_level: float = 0.1,
    true_ate: float = 15.0,
    confounding_strength: float = 2.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate synthetic coupon campaign data with realistic confounding.

    The data generation process:
    1. Generate customer features (loyalty, recency, frequency, etc.)
    2. Create confounded treatment assignment via propensity score
    3. Generate potential outcomes Y(0) and treatment effect τ(X)
    4. Observe outcome Y = Y(0) + T * τ(X) + noise

    Args:
        n_samples: Number of customers to generate
        treatment_rate: Target proportion receiving treatment
        noise_level: Standard deviation of outcome noise
        true_ate: True average treatment effect (in dollars)
        confounding_strength: Strength of confounding relationship
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame, ground_truth_dict)
        - DataFrame with features, treatment, and observed outcome
        - Dict with true ATE and CATE parameters for validation
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic data: n={n_samples:,}, ATE={true_ate}")

    # Generate customer features
    # Continuous features
    recency_days = np.random.exponential(scale=30, size=n_samples).clip(1, 365)
    frequency_purchases = np.random.poisson(lam=12, size=n_samples).clip(1, 100)
    monetary_value = np.random.gamma(shape=2, scale=50, size=n_samples).clip(10, 1000)
    avg_basket_size = np.random.gamma(shape=3, scale=20, size=n_samples).clip(5, 200)

    # Loyalty score (0-100, derived feature)
    loyalty_score = (
        50
        + 0.3 * (100 - recency_days / 3.65)  # Recency effect
        + 0.4 * np.minimum(frequency_purchases * 2, 40)  # Frequency effect
        + 0.3 * np.minimum(monetary_value / 10, 10)  # Monetary effect
        + np.random.normal(0, 10, n_samples)  # Noise
    ).clip(0, 100)

    # Price sensitivity (higher = more price-sensitive)
    price_sensitivity = np.random.beta(a=2, b=5, size=n_samples) * 10

    # Pre-campaign trend (spending growth rate)
    pre_trend = np.random.normal(0, 5, n_samples)

    # Categorical features
    segments = np.random.choice(
        ["high_value", "mid_value", "low_value", "new_customer"],
        size=n_samples,
        p=[0.2, 0.35, 0.30, 0.15],
    )

    regions = np.random.choice(
        ["north", "south", "east", "west"],
        size=n_samples,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    categories = np.random.choice(
        ["grocery", "electronics", "apparel", "home", "other"],
        size=n_samples,
        p=[0.3, 0.2, 0.2, 0.15, 0.15],
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n_samples),
            "recency_days": recency_days,
            "frequency_purchases": frequency_purchases,
            "monetary_value": monetary_value,
            "avg_basket_size": avg_basket_size,
            "loyalty_score": loyalty_score,
            "price_sensitivity": price_sensitivity,
            "pre_trend": pre_trend,
            "customer_segment": segments,
            "region": regions,
            "preferred_category": categories,
        }
    )

    # Generate confounded treatment assignment
    # Propensity score depends on loyalty, recency, and segment
    # Companies tend to target loyal, recent, high-value customers
    logit_propensity = (
        -1.0  # Baseline log-odds
        + confounding_strength * (loyalty_score - 50) / 50  # Loyalty effect
        - confounding_strength * 0.5 * (recency_days - 180) / 180  # Recency effect
        + confounding_strength * 0.3 * (frequency_purchases - 12) / 12  # Frequency
        + np.where(segments == "high_value", 1.0, 0.0)  # Target high-value
        - np.where(segments == "new_customer", 0.8, 0.0)  # Avoid new customers
    )

    propensity_score = 1 / (1 + np.exp(-logit_propensity))

    # Adjust to match target treatment rate
    treatment_threshold = np.percentile(propensity_score, (1 - treatment_rate) * 100)
    treatment = (propensity_score >= treatment_threshold).astype(int)

    # Store true propensity for validation
    df["true_propensity"] = propensity_score

    # Generate potential outcomes
    # Y(0): Base spending depends on loyalty, monetary value, and basket size
    base_spending = (
        20  # Baseline
        + 0.5 * loyalty_score  # Loyalty drives spending
        + 0.3 * monetary_value / 10  # Past monetary behavior
        + 0.2 * avg_basket_size  # Basket size
        + 0.1 * frequency_purchases  # Purchase frequency
        + 1.0 * pre_trend  # Pre-trend
        + np.where(segments == "high_value", 30, 0)
        + np.where(segments == "mid_value", 10, 0)
        - np.where(segments == "new_customer", 15, 0)
    )

    # Treatment effect τ(X): Heterogeneous by price sensitivity and loyalty
    # High price-sensitive customers respond more to coupons
    # But effect diminishes for already-loyal customers
    treatment_effect = (
        true_ate  # Base ATE
        + 5 * (price_sensitivity - 5) / 5  # Price-sensitive customers respond more
        - 0.2 * (loyalty_score - 50) / 50 * 10  # Loyal customers respond less
        + np.where(segments == "low_value", 8, 0)  # Bigger lift for low-value
        + np.where(segments == "new_customer", 12, 0)  # Biggest lift for new
        + np.random.normal(0, 3, n_samples)  # Individual variation
    )

    # Observed outcome: Y = Y(0) + T * τ(X) + ε
    outcome = (
        base_spending
        + treatment * treatment_effect
        + np.random.normal(0, noise_level * base_spending.mean(), n_samples)
    )

    # Ensure non-negative spending
    outcome = outcome.clip(0)

    df["treatment"] = treatment
    df["outcome"] = outcome

    # Store ground truth for validation (not available in real data!)
    df["true_treatment_effect"] = treatment_effect
    df["true_base_outcome"] = base_spending

    # Compute true ATE and key metrics
    true_ate_actual = treatment_effect.mean()
    true_att = treatment_effect[treatment == 1].mean()  # ATT = E[τ|T=1]
    true_atc = treatment_effect[treatment == 0].mean()  # ATC = E[τ|T=0]

    ground_truth = {
        "true_ate": true_ate_actual,
        "true_att": true_att,
        "true_atc": true_atc,
        "treatment_rate": treatment.mean(),
        "mean_outcome_treated": outcome[treatment == 1].mean(),
        "mean_outcome_control": outcome[treatment == 0].mean(),
        "cate_std": treatment_effect.std(),
    }

    logger.info(f"Generated synthetic data - True ATE: ${true_ate_actual:.2f}")
    logger.info(f"Treatment rate: {treatment.mean():.1%}")
    logger.info(f"CATE std: ${treatment_effect.std():.2f}")

    return df, ground_truth


def clean_data(df: pd.DataFrame, drop_leakage_cols: bool = True) -> pd.DataFrame:
    """
    Clean and validate coupon campaign data.

    Args:
        df: Input DataFrame
        drop_leakage_cols: Whether to drop columns that might leak information

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")

    df_clean = df.copy()

    # Check for missing values
    missing_summary = df_clean.isnull().sum()
    if missing_summary.any():
        logger.warning(f"Missing values detected:\n{missing_summary[missing_summary > 0]}")

    # Drop columns that leak information about the outcome
    # (e.g., post-treatment variables)
    if drop_leakage_cols:
        leakage_patterns = ["post_", "after_", "response_"]
        leakage_cols = [
            col for col in df_clean.columns if any(pattern in col for pattern in leakage_patterns)
        ]

        # Exclude ground truth columns (they're for validation only)
        ground_truth_cols = ["true_treatment_effect", "true_base_outcome", "true_propensity"]
        leakage_cols = [col for col in leakage_cols if col not in ground_truth_cols]

        if leakage_cols:
            logger.warning(f"Dropping potential leakage columns: {leakage_cols}")
            df_clean = df_clean.drop(columns=leakage_cols)

    # Validate treatment and outcome
    if "treatment" in df_clean.columns:
        if not set(df_clean["treatment"].unique()).issubset({0, 1}):
            logger.warning("Treatment column contains values other than 0/1, converting...")
            df_clean["treatment"] = (df_clean["treatment"] > 0).astype(int)

    if "outcome" in df_clean.columns:
        # Check for negative outcomes (shouldn't happen for spending)
        if (df_clean["outcome"] < 0).any():
            logger.warning("Negative outcomes detected, clipping to 0")
            df_clean["outcome"] = df_clean["outcome"].clip(lower=0)

    logger.info(f"Cleaned data: {len(df_clean):,} records, {len(df_clean.columns)} columns")

    return df_clean


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data: train={1-test_size:.0%}, test={test_size:.0%}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["treatment"] if "treatment" in df.columns else None,
    )

    logger.info(f"Train set: {len(train_df):,} records")
    logger.info(f"Test set: {len(test_df):,} records")

    return train_df, test_df


def load_synth_coupon(config: Dict, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate and optionally save synthetic coupon data.

    Args:
        config: Configuration dictionary
        output_path: Optional path to save generated data

    Returns:
        DataFrame with synthetic coupon data
    """
    synth_config = config.get("synthetic", {})

    df, ground_truth = generate_synthetic_coupon_data(
        n_samples=synth_config.get("n_samples", 10000),
        treatment_rate=synth_config.get("treatment_rate", 0.3),
        noise_level=synth_config.get("noise_level", 0.1),
        true_ate=synth_config.get("true_ate", 15.0),
        confounding_strength=synth_config.get("confounding_strength", 2.0),
        random_state=config.get("random_state", 42),
    )

    if output_path:
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")

        # Save ground truth separately
        ground_truth_path = str(Path(output_path).with_suffix(".ground_truth.joblib"))
        import joblib

        joblib.dump(ground_truth, ground_truth_path)
        logger.info(f"Saved ground truth to {ground_truth_path}")

    return df
