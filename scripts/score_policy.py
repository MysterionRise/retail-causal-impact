# path: scripts/score_policy.py
"""
CLI script for scoring and deploying coupon targeting policy.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from coupon_causal import policy, utils


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Score and deploy coupon targeting policy")

    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to CATE scores parquet file",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.2,
        help="Budget fraction (0-1) for targeting",
    )
    parser.add_argument(
        "--cost_per_coupon",
        type=float,
        default=1.0,
        help="Cost per coupon/treatment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/tables/targeting_recommendations.csv",
        help="Output path for targeting recommendations",
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    logger = utils.setup_logging()
    logger.info("Coupon Targeting Policy Scorer")

    # Load CATE scores
    logger.info(f"Loading CATE scores from {args.scores}")
    df = pd.read_parquet(args.scores)

    if "predicted_cate" not in df.columns:
        raise ValueError("Input file must contain 'predicted_cate' column")

    cate_scores = df["predicted_cate"].values

    # Determine optimal threshold
    logger.info(f"\nComputing optimal policy with budget={args.budget:.1%}")

    treatment_capacity = int(args.budget * len(df))
    threshold, treatment_indicator = policy.optimal_policy_threshold(
        cate_scores,
        cost_per_treatment=args.cost_per_coupon,
        treatment_capacity=treatment_capacity,
    )

    # Create targeting recommendations
    recommendations = policy.create_targeting_recommendations(
        cate_scores,
        df,
        threshold,
        output_path=args.output,
    )

    logger.info(f"\n✓ Targeting recommendations saved to {args.output}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("TARGETING POLICY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total customers: {len(df):,}")
    logger.info(f"Recommended for treatment: {treatment_indicator.sum():,} ({treatment_indicator.mean():.1%})")
    logger.info(f"Predicted average uplift (treated): ${cate_scores[treatment_indicator == 1].mean():.2f}")
    logger.info(f"Predicted total uplift: ${cate_scores[treatment_indicator == 1].sum():,.2f}")
    logger.info(f"Total cost: ${treatment_indicator.sum() * args.cost_per_coupon:,.2f}")
    logger.info(f"Net benefit: ${cate_scores[treatment_indicator == 1].sum() - treatment_indicator.sum() * args.cost_per_coupon:,.2f}")


if __name__ == "__main__":
    main()
