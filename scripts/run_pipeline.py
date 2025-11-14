# path: scripts/run_pipeline.py
"""
Main CLI pipeline for coupon causal impact analysis.

Runs end-to-end:  data → features → estimation → reports
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from coupon_causal import (
    ate,
    balance,
    cate,
    data,
    features,
    policy,
    propensity,
    utils,
    viz,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Coupon Causal Impact Analysis Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["public", "synth"],
        default="synth",
        help="Data source: 'public' for public dataset, 'synth' for synthetic",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file (for public data source)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["data", "estimation", "report", "all"],
        default="all",
        help="Pipeline stage to run",
    )

    return parser.parse_args()


def run_data_stage(config, data_source, input_path=None):
    """Run data loading/generation stage."""
    logger = logging.getLogger("coupon_causal")
    logger.info("=" * 80)
    logger.info("STAGE 1: DATA LOADING")
    logger.info("=" * 80)

    processed_path = Path(config["paths"]["processed_data"])
    utils.ensure_dir(processed_path)

    if data_source == "synth":
        # Generate synthetic data
        df = data.load_synth_coupon(
            config, output_path=str(processed_path / "coupon_data.parquet")
        )
    else:
        # Load public dataset
        if input_path is None:
            input_path = str(Path(config["paths"]["raw_data"]) / "coupons.csv")

        df = data.load_coupon_public(input_path)
        df = data.clean_data(df)
        df.to_parquet(processed_path / "coupon_data.parquet", index=False)

    logger.info(f"Data saved to {processed_path / 'coupon_data.parquet'}")

    return df


def run_estimation_stage(config, df):
    """Run causal estimation stage."""
    logger = logging.getLogger("coupon_causal")
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: CAUSAL ESTIMATION")
    logger.info("=" * 80)

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )

    logger.info(f"\nFeature matrix: {X.shape}")
    logger.info(f"Treatment rate: {T.mean():.1%}")
    logger.info(f"Outcome mean: ${Y.mean():.2f}")

    # Fit propensity models
    logger.info("\n" + "-" * 80)
    logger.info("PROPENSITY SCORE ESTIMATION")
    logger.info("-" * 80)

    logistic_prop, gbm_prop, ensemble_prop = propensity.fit_propensity_models(
        X, T, config
    )

    # Compute IPW weights
    ipw_weights = propensity.compute_ipw_weights(
        T,
        ensemble_prop,
        stabilize=config["ate"]["ipw_stabilize"],
        trim_percentiles=tuple(config["propensity"]["trim_percentiles"]),
    )

    # Estimate ATE
    logger.info("\n" + "-" * 80)
    logger.info("AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
    logger.info("-" * 80)

    ate_results = ate.estimate_all_ate_methods(X, Y, T, ensemble_prop, config)

    # Estimate CATE
    logger.info("\n" + "-" * 80)
    logger.info("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) ESTIMATION")
    logger.info("-" * 80)

    cate_results = cate.estimate_all_cate_methods(
        X, Y, T, ensemble_prop, config, feature_engineer.get_feature_names()
    )

    # Use best CATE method (Orthogonal Forest if available, else DR Learner)
    if cate_results.get("orthogonal_forest") is not None:
        cate_scores = cate_results["orthogonal_forest"]["cate"]
        cate_method = "orthogonal_forest"
    elif cate_results.get("dr_learner") is not None:
        cate_scores = cate_results["dr_learner"]["cate"]
        cate_method = "dr_learner"
    else:
        cate_scores = cate_results["x_learner"]["cate"]
        cate_method = "x_learner"

    logger.info(f"\nUsing {cate_method} for policy evaluation")

    # Balance diagnostics
    logger.info("\n" + "-" * 80)
    logger.info("BALANCE DIAGNOSTICS")
    logger.info("-" * 80)

    balance_before, balance_after, balance_summary = balance.compare_balance_before_after(
        X, T, feature_engineer.get_feature_names(), ipw_weights
    )

    # Policy evaluation
    logger.info("\n" + "-" * 80)
    logger.info("POLICY EVALUATION")
    logger.info("-" * 80)

    policy_results = policy.evaluate_policy_uplift(
        cate_scores, T, Y, budget_fractions=config["policy"]["budget_fractions"]
    )

    auuc = policy.compute_auuc(cate_scores, T, Y, normalize=True)
    logger.info(f"\nArea Under Uplift Curve (AUUC): {auuc:.3f}")

    # Segment analysis
    if "customer_segment" in df.columns:
        segment_stats = policy.segment_uplift_analysis(
            cate_scores, df, "customer_segment", top_k=10
        )
    else:
        segment_stats = None

    # Save results
    models_path = Path(config["paths"]["models"])
    utils.ensure_dir(models_path)

    results = {
        "feature_engineer": feature_engineer,
        "propensity_scores": ensemble_prop,
        "ipw_weights": ipw_weights,
        "ate_results": ate_results,
        "cate_results": cate_results,
        "cate_scores": cate_scores,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "balance_summary": balance_summary,
        "policy_results": policy_results,
        "auuc": auuc,
        "segment_stats": segment_stats,
    }

    utils.save_artifact(results, str(models_path / "estimation_results.joblib"))

    logger.info(f"\nResults saved to {models_path / 'estimation_results.joblib'}")

    return results


def run_report_stage(config, df, results):
    """Generate reports and visualizations."""
    logger = logging.getLogger("coupon_causal")
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: REPORT GENERATION")
    logger.info("=" * 80)

    figures_path = Path(config["paths"]["figures"])
    tables_path = Path(config["paths"]["tables"])
    utils.ensure_dir(figures_path)
    utils.ensure_dir(tables_path)

    # Extract results
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=False, feature_engineer=results["feature_engineer"]
    )

    propensity_scores = results["propensity_scores"]
    cate_scores = results["cate_scores"]
    balance_before = results["balance_before"]
    balance_after = results["balance_after"]
    ate_results = results["ate_results"]

    # Generate visualizations
    logger.info("Generating visualizations...")

    # 1. Treatment distribution
    viz.plot_treatment_distribution(
        T, Y, save_path=str(figures_path / "01_treatment_distribution.png")
    )

    # 2. Propensity distribution
    viz.plot_propensity_distribution(
        propensity_scores, T, save_path=str(figures_path / "02_propensity_distribution.png")
    )

    # 3. Love plot
    viz.plot_love_plot(
        balance_before,
        balance_after,
        top_k=config["balance"]["plot_top_features"],
        save_path=str(figures_path / "03_love_plot.png"),
    )

    # 4. ATE comparison
    viz.plot_ate_comparison(
        ate_results, save_path=str(figures_path / "04_ate_comparison.png")
    )

    # 5. CATE distribution
    viz.plot_cate_distribution(
        cate_scores, method_name="CATE", save_path=str(figures_path / "05_cate_distribution.png")
    )

    # 6. Qini curve
    fractions, qini_values, random_baseline = policy.compute_qini_curve(
        cate_scores, T, Y, n_bins=config["policy"]["qini_bins"]
    )
    viz.plot_qini_curve(
        fractions, qini_values, random_baseline,
        save_path=str(figures_path / "06_qini_curve.png")
    )

    # 7. Segment uplift
    if results["segment_stats"] is not None:
        viz.plot_segment_uplift(
            results["segment_stats"],
            save_path=str(figures_path / "07_segment_uplift.png")
        )

    # Save tables
    logger.info("Saving summary tables...")

    # ATE summary
    ate_summary = pd.DataFrame(
        [
            {
                "method": method,
                "ate": res["ate"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
            }
            for method, res in ate_results.items()
        ]
    )
    ate_summary.to_csv(tables_path / "ate_summary.csv", index=False)

    # Balance summary
    balance_before.to_csv(tables_path / "balance_before.csv", index=False)
    balance_after.to_csv(tables_path / "balance_after.csv", index=False)

    # Policy results
    results["policy_results"].to_csv(tables_path / "policy_results.csv", index=False)

    # CATE scores
    cate_df = df.copy()
    cate_df["predicted_cate"] = cate_scores
    cate_df.to_parquet(tables_path / "cate_scores.parquet", index=False)

    logger.info(f"\nFigures saved to {figures_path}/")
    logger.info(f"Tables saved to {tables_path}/")


def main():
    """Main pipeline execution."""
    args = parse_args()

    # Load config
    config = utils.load_config(args.config)
    utils.set_random_seed(config["random_state"])

    # Setup logging
    logger = utils.setup_logging(
        level=config["logging"]["level"],
        log_format=config["logging"]["format"],
    )

    logger.info("Coupon Causal Impact Analysis Pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Stage: {args.stage}")

    # Run pipeline stages
    if args.stage in ["data", "all"]:
        df = run_data_stage(config, args.data_source, args.input)
    else:
        # Load existing data
        processed_path = Path(config["paths"]["processed_data"]) / "coupon_data.parquet"
        df = pd.read_parquet(processed_path)
        logger.info(f"Loaded data from {processed_path}")

    if args.stage in ["estimation", "all"]:
        results = run_estimation_stage(config, df)
    else:
        # Load existing results
        models_path = Path(config["paths"]["models"]) / "estimation_results.joblib"
        results = utils.load_artifact(str(models_path))
        logger.info(f"Loaded results from {models_path}")

    if args.stage in ["report", "all"]:
        run_report_stage(config, df, results)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
