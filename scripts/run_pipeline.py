# path: scripts/run_pipeline.py
"""
Main CLI pipeline for coupon causal impact analysis.

Runs end-to-end:  data → features → estimation → refutation → reports
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from coupon_causal import (
    ate,
    balance,
    cate,
    data,
    features,
    policy,
    propensity,
    refute,
    utils,
    viz,
)

# Preference order for the CATE model used downstream (policy, segments, app)
CATE_METHOD_PREFERENCE = ["causal_forest", "dr_learner", "x_learner"]


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


def _section(logger, title):
    logger.info("\n" + "-" * 80)
    logger.info(title)
    logger.info("-" * 80)


def run_estimation_stage(config, df):
    """Run causal estimation stage."""
    logger = logging.getLogger("coupon_causal")
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: CAUSAL ESTIMATION")
    logger.info("=" * 80)

    random_state = config.get("random_state", 42)

    # Prepare features
    X, T, Y, feature_engineer = features.prepare_features(
        df, config["features"], fit=True
    )
    feature_names = feature_engineer.get_feature_names()

    logger.info(f"\nFeature matrix: {X.shape}")
    logger.info(f"Treatment rate: {T.mean():.1%}")
    logger.info(f"Outcome mean: ${Y.mean():.2f}")

    # ------------------------------------------------------------------
    # Propensity scores
    # ------------------------------------------------------------------
    _section(logger, "PROPENSITY SCORE ESTIMATION")

    logistic_prop, gbm_prop, ensemble_prop = propensity.fit_propensity_models(
        X, T, config
    )

    # Choose which propensity scores drive IPW/AIPW. The bootstrap refits the
    # same model on every draw, so this must be something cheap to refit
    # (logistic by default). GBM and the ensemble are still fit above and
    # reported as diagnostics.
    propensity_model_type = config["propensity"].get("ate_model", "logistic")
    propensity_scores = {
        "logistic": logistic_prop.propensity_scores_,
        "gbm": gbm_prop.propensity_scores_,
        "ensemble": ensemble_prop,
    }[propensity_model_type]
    logger.info(f"Using {propensity_model_type} propensity scores for weighting and ATE")

    overlap = propensity.assess_overlap(propensity_scores, T)

    # Compute IPW weights
    ipw_weights = propensity.compute_ipw_weights(
        T,
        propensity_scores,
        stabilize=config["ate"]["ipw_stabilize"],
        trim_percentiles=tuple(config["propensity"]["trim_percentiles"]),
    )

    # ------------------------------------------------------------------
    # ATE (full sample)
    # ------------------------------------------------------------------
    _section(logger, "AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")

    ate_results = ate.estimate_all_ate_methods(
        X, Y, T, propensity_scores, config, propensity_model_type=propensity_model_type
    )
    headline_method = "aipw_linear" if "aipw_linear" in ate_results else next(iter(ate_results))
    headline_ate = ate_results[headline_method]["ate"]

    # ------------------------------------------------------------------
    # Balance diagnostics (full sample)
    # ------------------------------------------------------------------
    _section(logger, "BALANCE DIAGNOSTICS")

    balance_before, balance_after, balance_summary = balance.compare_balance_before_after(
        X, T, feature_names, ipw_weights
    )

    # ------------------------------------------------------------------
    # Held-out split for everything predictive (CATE, policy, targeting)
    # ------------------------------------------------------------------
    eval_config = config.get("evaluation", {})
    test_size = eval_config.get("test_size", 0.3)
    idx_train, idx_test = train_test_split(
        np.arange(len(Y)), test_size=test_size, random_state=random_state, stratify=T
    )
    X_tr, Y_tr, T_tr, e_tr = X[idx_train], Y[idx_train], T[idx_train], propensity_scores[idx_train]
    X_te, Y_te, T_te, e_te = X[idx_test], Y[idx_test], T[idx_test], propensity_scores[idx_test]
    true_cate_all = df["true_treatment_effect"].values if "true_treatment_effect" in df.columns else None
    true_cate_te = true_cate_all[idx_test] if true_cate_all is not None else None

    logger.info(f"\nHeld-out split: train={len(idx_train):,}  test={len(idx_test):,}")

    # Doubly-robust pseudo-outcomes on the test set: outcome models fit on train,
    # so ψ on the test rows is out-of-sample.
    rf_params = config.get("ate", {}).get("rf_outcome_model", {})
    _, (model_mu1, model_mu0) = ate.estimate_ate_aipw(
        X_tr, Y_tr, T_tr, e_tr, model_type="rf", rf_params=rf_params
    )
    dr_scores_te = policy.compute_dr_scores(
        Y_te, T_te, e_te, model_mu0.predict(X_te), model_mu1.predict(X_te)
    )
    logger.info(f"Mean DR pseudo-outcome on test set: ${dr_scores_te.mean():.2f}")

    # ------------------------------------------------------------------
    # CATE (fit on train, evaluate on test, score everyone for deployment)
    # ------------------------------------------------------------------
    _section(logger, "CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) ESTIMATION")

    cate_results = cate.estimate_all_cate_methods(
        X_tr, Y_tr, T_tr, e_tr, config, feature_names
    )

    comparison_budget = config["policy"].get("comparison_budget", 0.2)
    k_budget = max(1, int(comparison_budget * len(idx_test)))
    heldout_rows = []
    cate_test_by_method = {}
    cate_all_by_method = {}
    for method_name, method_results in cate_results.items():
        if method_results is None:
            continue
        c_te = cate.predict_cate(method_name, method_results, X_te, e_te)
        c_all = cate.predict_cate(method_name, method_results, X, propensity_scores)
        cate_test_by_method[method_name] = c_te
        cate_all_by_method[method_name] = c_all
        top = np.argsort(-c_te)[:k_budget]
        row = {
            "method": method_name,
            "mean_cate_test": float(c_te.mean()),
            "std_cate_test": float(c_te.std()),
            "auuc_heldout": float(policy.compute_auuc(c_te, T_te, Y_te)),
            f"dr_uplift_top{int(comparison_budget * 100)}": float(dr_scores_te[top].mean()),
        }
        if true_cate_te is not None:
            row["corr_true_cate_heldout"] = float(np.corrcoef(true_cate_te, c_te)[0, 1])
            row[f"true_uplift_top{int(comparison_budget * 100)}"] = float(true_cate_te[top].mean())
        heldout_rows.append(row)
    cate_heldout_metrics = pd.DataFrame(heldout_rows)
    logger.info("\nHeld-out CATE metrics:")
    logger.info(f"\n{cate_heldout_metrics.round(3).to_string(index=False)}")

    cate_method = next(m for m in CATE_METHOD_PREFERENCE if m in cate_test_by_method)
    cate_scores = cate_all_by_method[cate_method]
    cate_scores_test = cate_test_by_method[cate_method]
    logger.info(f"\nUsing {cate_method} for policy evaluation and targeting")

    # Calibration of the chosen model by predicted-effect decile (held-out)
    cate_calibration = policy.cate_calibration_by_decile(
        cate_scores_test, dr_scores_te, true_cate_te, n_bins=10
    )

    # ------------------------------------------------------------------
    # Policy evaluation (held-out)
    # ------------------------------------------------------------------
    _section(logger, "POLICY EVALUATION (HELD-OUT)")

    policy_results = policy.evaluate_policy_uplift(
        cate_scores_test,
        T_te,
        Y_te,
        budget_fractions=config["policy"]["budget_fractions"],
        dr_scores=dr_scores_te,
    )

    auuc = policy.compute_auuc(cate_scores_test, T_te, Y_te, normalize=True)
    logger.info(f"\nArea Under Uplift Curve (AUUC, held-out): {auuc:.3f}")

    # Predictive ML (response model) vs causal uplift targeting
    _section(logger, "TARGETING: PREDICTIVE RESPONSE MODEL vs CAUSAL UPLIFT MODEL")

    response_model = policy.fit_response_model(X_tr, Y_tr, T_tr, random_state=random_state)
    strategy_scores = {
        "uplift_model": cate_scores_test,
        "response_model": response_model.predict(X_te),
    }
    targeting_comparison = policy.evaluate_targeting_strategies(
        strategy_scores,
        dr_scores_te,
        budget_fraction=comparison_budget,
        true_cate=true_cate_te,
        random_state=random_state,
    )
    n_bins = config["policy"]["qini_bins"]
    targeting_curves = {}
    for name, scores in strategy_scores.items():
        fractions, gain, random_baseline = policy.compute_dr_uplift_curve(
            scores, dr_scores_te, n_bins=n_bins
        )
        targeting_curves[name] = (fractions, gain)
    targeting_random_baseline = random_baseline

    # Segment analysis (all customers, deployment scores)
    if "customer_segment" in df.columns:
        segment_stats = policy.segment_uplift_analysis(
            cate_scores, df, "customer_segment", top_k=10
        )
    else:
        segment_stats = None

    # ------------------------------------------------------------------
    # Refutation and sensitivity (full sample, headline AIPW estimate)
    # ------------------------------------------------------------------
    _section(logger, "REFUTATION AND SENSITIVITY CHECKS")

    refutation = refute.run_all_refutations(
        df, X, Y, T, feature_names, propensity_scores, headline_ate, config
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    models_path = Path(config["paths"]["models"])
    utils.ensure_dir(models_path)

    results = {
        "feature_engineer": feature_engineer,
        "propensity_scores": propensity_scores,
        "propensity_model_type": propensity_model_type,
        "overlap": overlap,
        "ipw_weights": ipw_weights,
        "ate_results": ate_results,
        "headline_ate_method": headline_method,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "balance_summary": balance_summary,
        "split": {"train_idx": idx_train, "test_idx": idx_test, "test_size": test_size},
        "cate_results": cate_results,
        "cate_method": cate_method,
        "cate_scores": cate_scores,
        "cate_scores_test": cate_scores_test,
        "cate_heldout_metrics": cate_heldout_metrics,
        "cate_calibration": cate_calibration,
        "dr_scores_test": dr_scores_te,
        "true_cate_test": true_cate_te,
        "policy_results": policy_results,
        "auuc": auuc,
        "targeting_comparison": targeting_comparison,
        "targeting_curves": targeting_curves,
        "targeting_random_baseline": targeting_random_baseline,
        "segment_stats": segment_stats,
        "refutation": refutation,
        "dag": refutation.get("dowhy", {}).get("graph") or refute.build_causal_graph(list(df.columns)),
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
    idx_test = results["split"]["test_idx"]
    cate_scores_test = results["cate_scores_test"]
    cate_method = results["cate_method"]

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

    # 5. CATE distribution (deployment scores, all customers)
    viz.plot_cate_distribution(
        cate_scores, method_name=cate_method, save_path=str(figures_path / "05_cate_distribution.png")
    )

    # 6. Qini curve (held-out)
    fractions, qini_values, random_baseline = policy.compute_qini_curve(
        cate_scores_test, T[idx_test], Y[idx_test], n_bins=config["policy"]["qini_bins"]
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

    # 8. Predictive ML vs causal targeting (held-out)
    viz.plot_targeting_comparison(
        results["targeting_comparison"],
        results["targeting_curves"],
        results["targeting_random_baseline"],
        save_path=str(figures_path / "08_targeting_ml_vs_causal.png"),
    )

    # 9. CATE validation (held-out)
    viz.plot_cate_validation(
        results["cate_calibration"],
        cate_pred=cate_scores_test,
        true_cate=results["true_cate_test"],
        method_name=cate_method,
        save_path=str(figures_path / "09_cate_validation.png"),
    )

    # 10. Refutation summary
    headline_ate = ate_results[results["headline_ate_method"]]["ate"]
    viz.plot_refutation_summary(
        results["refutation"]["summary"],
        headline_ate,
        save_path=str(figures_path / "10_refutation_checks.png"),
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

    # Policy, held-out CATE metrics, targeting comparison, calibration
    results["policy_results"].to_csv(tables_path / "policy_results.csv", index=False)
    results["cate_heldout_metrics"].to_csv(tables_path / "cate_heldout_metrics.csv", index=False)
    results["targeting_comparison"].to_csv(tables_path / "targeting_comparison.csv", index=False)
    results["cate_calibration"].to_csv(tables_path / "cate_calibration_deciles.csv", index=False)

    # Refutation
    refutation = results["refutation"]
    refutation["summary"].to_csv(tables_path / "refutation_summary.csv", index=False)
    refutation["leave_one_out"].to_csv(tables_path / "leave_one_out.csv", index=False)
    pd.DataFrame([refutation["robustness"]]).to_csv(tables_path / "robustness_value.csv", index=False)
    (tables_path / "causal_graph.dot").write_text(results["dag"] + "\n")

    # CATE scores (deployment scores for everyone, plus held-out diagnostics)
    cate_df = df.copy()
    cate_df["predicted_cate"] = cate_scores
    cate_df["is_test"] = False
    cate_df.loc[cate_df.index[idx_test], "is_test"] = True
    cate_df["dr_score"] = np.nan
    cate_df.loc[cate_df.index[idx_test], "dr_score"] = results["dr_scores_test"]
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
    # DoWhy is chatty at INFO (and prints its full estimand on every refuter)
    logging.getLogger("dowhy").setLevel(logging.WARNING)

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
    elif args.stage == "report":
        # Load existing results only if needed for report
        models_path = Path(config["paths"]["models"]) / "estimation_results.joblib"
        results = utils.load_artifact(str(models_path))
        logger.info(f"Loaded results from {models_path}")
    else:
        # No results needed for data-only stage
        results = None

    if args.stage in ["report", "all"]:
        if results is None:
            logger.error("Results not available. Run estimation stage first.")
            sys.exit(1)
        run_report_stage(config, df, results)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
