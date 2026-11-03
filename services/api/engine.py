# path: services/api/engine.py
"""
The causal engine behind the API and the copilot.

Loads the artifacts written by ``make all`` (data, config, estimation results)
and exposes deterministic, JSON-serialisable queries over them: subgroup
effects, what-if budgets, targeting comparisons, graph edits, refutation
results and confounding simulations.

Every number the copilot shows comes from one of these methods. The language
model never computes anything itself.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from coupon_causal import (  # noqa: E402
    ate,
    balance,
    data,
    features,
    policy,
    propensity,
    refute,
    utils,
)

logger = logging.getLogger(__name__)

CATEGORICAL_FILTERS = ("customer_segment", "region", "preferred_category")
NUMERIC_FILTERS = (
    "loyalty_score",
    "recency_days",
    "frequency_purchases",
    "monetary_value",
    "avg_basket_size",
    "price_sensitivity",
    "pre_trend",
)


class ArtifactsMissingError(RuntimeError):
    """Raised when the pipeline outputs the engine needs are not on disk."""


def _clean(value: Any) -> Any:
    """Make numpy / pandas values JSON-friendly (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_clean(r) for r in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return _clean(value.tolist())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    return value


@dataclass
class Engine:
    """In-memory causal engine over the pipeline artifacts."""

    config: dict
    df: pd.DataFrame
    results: dict
    X: np.ndarray = field(repr=False)
    T: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)
    feature_names: list[str]
    propensity_scores: np.ndarray = field(repr=False)
    dr_scores: np.ndarray = field(repr=False)
    response_scores: np.ndarray = field(repr=False)
    baseline_dag: dict

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | str = ROOT, config_path: str | None = None) -> Engine:
        """Load config, data and results from a repo checkout."""
        root = Path(root)
        config = utils.load_config(str(config_path or root / "config" / "default.yaml"))
        data_path = root / config["paths"]["processed_data"] / "coupon_data.parquet"
        results_path = root / config["paths"]["models"] / "estimation_results.joblib"
        missing = [p for p in (data_path, results_path) if not p.exists()]
        if missing:
            raise ArtifactsMissingError(
                "Pipeline artifacts not found: "
                + ", ".join(str(p) for p in missing)
                + ". Run `make all` first."
            )
        df = pd.read_parquet(data_path)
        results = utils.load_artifact(str(results_path))
        return cls.from_results(config, df, results)

    @classmethod
    def from_results(cls, config: dict, df: pd.DataFrame, results: dict) -> Engine:
        """Build the engine from in-memory pipeline outputs."""
        X, T, Y, fe = features.prepare_features(
            df, config["features"], fit=False, feature_engineer=results["feature_engineer"]
        )
        feature_names = fe.get_feature_names()
        e = np.asarray(results["propensity_scores"])
        random_state = config.get("random_state", 42)

        # Doubly-robust pseudo-outcomes for *every* customer via 3-fold
        # cross-fitting of the outcome models (the pipeline only scores the
        # held-out split; subgroup queries want the whole population).
        rf_params = config.get("ate", {}).get("rf_outcome_model", {})
        dr = np.empty(len(Y))
        for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=random_state).split(X):
            _, (m1, m0) = ate.estimate_ate_aipw(
                X[train_idx], Y[train_idx], T[train_idx], e[train_idx], model_type="rf", rf_params=rf_params
            )
            dr[test_idx] = policy.compute_dr_scores(
                Y[test_idx], T[test_idx], e[test_idx], m0.predict(X[test_idx]), m1.predict(X[test_idx])
            )

        # Response-model baseline (predict spend among coupon recipients)
        split = results.get("split")
        train_idx = split["train_idx"] if split else np.arange(len(Y))
        response = policy.fit_response_model(X[train_idx], Y[train_idx], T[train_idx], random_state=random_state)
        response_scores = response.predict(X)

        baseline_dag = refute.identify_effect(cls._dowhy_frame(df))

        return cls(
            config=config,
            df=df,
            results=results,
            X=X,
            T=T,
            Y=Y,
            feature_names=feature_names,
            propensity_scores=e,
            dr_scores=dr,
            response_scores=response_scores,
            baseline_dag=baseline_dag,
        )

    @staticmethod
    def _dowhy_frame(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if not c.startswith("true_") and c != "customer_id"]
        return df[cols]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def has_ground_truth(self) -> bool:
        return "true_treatment_effect" in self.df.columns

    @property
    def true_cate(self) -> np.ndarray | None:
        return self.df["true_treatment_effect"].values if self.has_ground_truth else None

    @property
    def cate_scores(self) -> np.ndarray:
        return np.asarray(self.results["cate_scores"])

    @property
    def headline_ate(self) -> float:
        method = self.results.get("headline_ate_method", "aipw_linear")
        return float(self.results["ate_results"][method]["ate"])

    def filter_options(self) -> dict:
        """Values and ranges the subgroup filters accept."""
        out: dict[str, Any] = {}
        for col in CATEGORICAL_FILTERS:
            if col in self.df.columns:
                out[col] = sorted(self.df[col].unique().tolist())
        for col in NUMERIC_FILTERS:
            if col in self.df.columns:
                out[col] = {"min": float(self.df[col].min()), "max": float(self.df[col].max())}
        return _clean(out)

    def _mask(self, filters: dict | None) -> tuple[np.ndarray, list[str]]:
        mask = np.ones(len(self.df), dtype=bool)
        described: list[str] = []
        for col, spec in (filters or {}).items():
            if col not in self.df.columns:
                raise ValueError(f"Unknown filter column '{col}'. Known: {list(self.filter_options())}")
            series = self.df[col]
            if col in CATEGORICAL_FILTERS:
                values = [spec] if isinstance(spec, str) else list(spec)
                unknown = sorted(set(values) - set(series.unique()))
                if unknown:
                    raise ValueError(f"Unknown values for {col}: {unknown}. Known: {sorted(series.unique())}")
                mask &= series.isin(values).values
                described.append(f"{col} in {values}")
            else:
                if not isinstance(spec, dict):
                    raise ValueError(f"Numeric filter for {col} must be an object with 'min' and/or 'max'")
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None:
                    mask &= (series >= lo).values
                if hi is not None:
                    mask &= (series <= hi).values
                described.append(f"{col} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]")
        return mask, described

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Headline numbers for the dashboard and the copilot's system prompt."""
        r = self.results
        ate_rows = [
            {"method": m, "ate": v["ate"], "ci_lower": v["ci_lower"], "ci_upper": v["ci_upper"]}
            for m, v in r["ate_results"].items()
        ]
        naive = r["ate_results"]["naive"]["ate"]
        headline = self.headline_ate
        refutation = r.get("refutation", {})
        summary_table = refutation.get("summary")
        out = {
            "n_customers": int(len(self.df)),
            "treatment_rate": float(self.T.mean()),
            "mean_outcome": float(self.Y.mean()),
            "true_ate": float(self.true_cate.mean()) if self.has_ground_truth else None,
            "naive_ate": naive,
            "headline_ate": headline,
            "headline_method": r.get("headline_ate_method", "aipw_linear"),
            "naive_overstatement_pct": (naive / headline - 1) * 100 if headline else None,
            "ate_table": ate_rows,
            "cate_method": r.get("cate_method"),
            "balance": r.get("balance_summary"),
            "overlap": r.get("overlap"),
            "auuc_heldout": r.get("auuc"),
            "heldout_metrics": r.get("cate_heldout_metrics"),
            "refutation_checks_passed": int(summary_table["passed"].sum()) if summary_table is not None else None,
            "refutation_checks_total": int(len(summary_table)) if summary_table is not None else None,
            "robustness_value": refutation.get("robustness", {}).get("robustness_value"),
            "backdoor_variables": refutation.get("dowhy", {}).get("backdoor_variables"),
            "propensity_model": r.get("propensity_model_type"),
        }
        return _clean(out)

    def subgroup_effect(self, filters: dict | None = None) -> dict:
        """
        Average treatment effect inside a subgroup defined by raw-column filters.

        Uses doubly-robust pseudo-outcomes, so the estimate is valid even
        though coupon assignment inside the subgroup is still confounded.
        """
        mask, described = self._mask(filters)
        n = int(mask.sum())
        if n < 30:
            raise ValueError(f"Subgroup has only {n} customers (need at least 30): {described}")
        dr = self.dr_scores[mask]
        se = float(dr.std(ddof=1) / np.sqrt(n))
        est = float(dr.mean())
        t_sub, y_sub = self.T[mask], self.Y[mask]
        naive = float(y_sub[t_sub == 1].mean() - y_sub[t_sub == 0].mean()) if (t_sub == 1).any() and (t_sub == 0).any() else None
        out = {
            "filters": described or ["all customers"],
            "n": n,
            "n_treated": int(t_sub.sum()),
            "share_of_population": n / len(self.df),
            "effect_dr": est,
            "se": se,
            "ci_lower": est - 1.96 * se,
            "ci_upper": est + 1.96 * se,
            "naive_difference": naive,
            "mean_predicted_cate": float(self.cate_scores[mask].mean()),
            "population_ate": self.headline_ate,
            "true_effect": float(self.true_cate[mask].mean()) if self.has_ground_truth else None,
            "mean_outcome": float(y_sub.mean()),
        }
        return _clean(out)

    def whatif_budget(self, budget_fraction: float = 0.2, cost_per_coupon: float = 1.0) -> dict:
        """
        Target the top ``budget_fraction`` of customers by predicted uplift at a
        given coupon cost, and report cost, incremental spend and net benefit.
        """
        if not 0 < budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        if cost_per_coupon < 0:
            raise ValueError("cost_per_coupon must be non-negative")
        n = len(self.df)
        k = max(1, int(budget_fraction * n))
        order = np.argsort(-self.cate_scores)
        top = order[:k]
        rng = np.random.default_rng(self.config.get("random_state", 42))
        random_pick = rng.choice(n, size=k, replace=False)
        resp_top = np.argsort(-self.response_scores)[:k]

        def value(idx: np.ndarray) -> dict:
            dr_total = float(self.dr_scores[idx].sum())
            cost = k * cost_per_coupon
            d = {
                "n_targeted": int(len(idx)),
                "predicted_total_uplift": float(self.cate_scores[idx].sum()),
                "dr_total_uplift": dr_total,
                "dr_uplift_per_customer": dr_total / len(idx),
                "cost": cost,
                "net_benefit": dr_total - cost,
                "roi": (dr_total - cost) / cost if cost > 0 else None,
            }
            if self.has_ground_truth:
                d["true_total_uplift"] = float(self.true_cate[idx].sum())
            return d

        # Break-even: first rank where predicted uplift drops below the cost
        sorted_cate = self.cate_scores[order]
        below = np.where(sorted_cate < cost_per_coupon)[0]
        breakeven_fraction = float(below[0] / n) if len(below) else 1.0
        everyone = value(np.arange(n))
        out = {
            "budget_fraction": budget_fraction,
            "cost_per_coupon": cost_per_coupon,
            "uplift_policy": value(top),
            "random_policy": value(random_pick),
            "response_model_policy": value(resp_top),
            "treat_everyone": everyone,
            "breakeven_budget_fraction": breakeven_fraction,
            "cate_threshold": float(sorted_cate[k - 1]),
        }
        return _clean(out)

    def compare_targeting(self, budget_fraction: float = 0.2) -> dict:
        """Predictive response model vs causal uplift model on the held-out split."""
        split = self.results.get("split")
        if split is None:
            idx = np.arange(len(self.df))
        else:
            idx = np.asarray(split["test_idx"])
        strategies = {
            "uplift_model": self.cate_scores[idx],
            "response_model": self.response_scores[idx],
        }
        table = policy.evaluate_targeting_strategies(
            strategies,
            self.dr_scores[idx],
            budget_fraction=budget_fraction,
            true_cate=self.true_cate[idx] if self.has_ground_truth else None,
            random_state=self.config.get("random_state", 42),
        )
        curves = {}
        for name, scores in strategies.items():
            fr, gain, baseline = policy.compute_dr_uplift_curve(scores, self.dr_scores[idx], n_bins=20)
            curves[name] = {"fractions": fr, "gain": gain}
        return _clean(
            {
                "budget_fraction": budget_fraction,
                "n_heldout": int(len(idx)),
                "strategies": table,
                "curves": curves,
                "random_baseline": baseline,
            }
        )

    def dag(self) -> dict:
        """The assumed causal graph and what DoWhy derives from it."""
        return _clean(
            {
                "edges": [list(e) for e in self.baseline_dag["edges"]],
                "backdoor_variables": self.baseline_dag["backdoor_variables"],
                "identified": self.baseline_dag["identified"],
                "dowhy_estimate": self.baseline_dag["estimate"],
                "graph": self.baseline_dag["graph"],
                "nodes": sorted({n for e in self.baseline_dag["edges"] for n in e}),
                "available_variables": [
                    c for c in self.df.columns if c not in ("customer_id",) and not c.startswith("true_")
                ],
            }
        )

    def edit_dag(
        self,
        add_edges: list[list[str]] | None = None,
        remove_edges: list[list[str]] | None = None,
        base_edges: list[list[str]] | None = None,
    ) -> dict:
        """
        Apply edge edits to the assumed graph, re-identify the adjustment set
        with DoWhy and re-estimate the effect with the pipeline's AIPW using
        exactly that adjustment set.
        """
        current = [tuple(e) for e in (base_edges or self.baseline_dag["edges"])]
        allowed = set(self.df.columns) - {"customer_id"} - {c for c in self.df.columns if c.startswith("true_")}
        for pair in (add_edges or []) + (remove_edges or []):
            if len(pair) != 2 or pair[0] not in allowed or pair[1] not in allowed:
                raise ValueError(f"Edge {pair} must be two of: {sorted(allowed)}")
            if pair[1] == "treatment" and pair[0] == "outcome":
                raise ValueError("outcome -> treatment would make the graph cyclic")
        edges = [e for e in current if list(e) not in [list(r) for r in (remove_edges or [])]]
        for a, b in add_edges or []:
            if (a, b) not in edges:
                edges.append((a, b))

        new = refute.identify_effect(self._dowhy_frame(self.df), edges)

        # Re-estimate with AIPW on the backdoor set only
        aipw_estimate = None
        if new["identified"]:
            cols = [c for c in new["backdoor_variables"] if c in self.df.columns]
            cat = [c for c in cols if c in self.config["features"]["categorical_features"]]
            cont = [c for c in cols if c in self.config["features"]["continuous_features"]]
            if cat or cont:
                fe = features.FeatureEngineer(categorical_features=cat, continuous_features=cont)
                X_adj = fe.fit_transform(self.df)
                e_adj = ate._refit_propensity(X_adj, self.T, "logistic", self.config.get("propensity", {}), self.config.get("random_state", 42))
                aipw_estimate, _ = ate.estimate_ate_aipw(X_adj, self.Y, self.T, e_adj, model_type="linear")
            else:
                aipw_estimate = float(self.Y[self.T == 1].mean() - self.Y[self.T == 0].mean())

        base = self.baseline_dag
        out = {
            "edges": [list(e) for e in edges],
            "added": add_edges or [],
            "removed": remove_edges or [],
            "backdoor_variables": new["backdoor_variables"],
            "baseline_backdoor_variables": base["backdoor_variables"],
            "identified": new["identified"],
            "dowhy_estimate": new["estimate"],
            "baseline_dowhy_estimate": base["estimate"],
            "aipw_estimate_with_new_adjustment": aipw_estimate,
            "pipeline_headline_ate": self.headline_ate,
            "true_ate": float(self.true_cate.mean()) if self.has_ground_truth else None,
            "graph": new["graph"],
            "nodes": sorted({n for e in edges for n in e}),
        }
        return _clean(out)

    def simulate_confounding(self, confounding_strength: float = 2.0, n_samples: int = 5000) -> dict:
        """
        Regenerate the synthetic campaign with a different confounding strength
        and compare naive, IPW and AIPW estimates against the truth. Fast enough
        for a slider.
        """
        if not 0 <= confounding_strength <= 8:
            raise ValueError("confounding_strength must be between 0 and 8")
        n_samples = int(min(max(n_samples, 500), 20000))
        synth = self.config["synthetic"]
        df, gt = data.generate_synthetic_coupon_data(
            n_samples=n_samples,
            treatment_rate=synth["treatment_rate"],
            noise_level=synth.get("noise_level", 0.1),
            true_ate=synth["true_ate"],
            confounding_strength=confounding_strength,
            random_state=self.config.get("random_state", 42),
        )
        X, T, Y, fe = features.prepare_features(df, self.config["features"], fit=True)
        e = ate._refit_propensity(X, T, "logistic", self.config.get("propensity", {}), self.config.get("random_state", 42))
        w = propensity.compute_ipw_weights(T, e, stabilize=True, trim_percentiles=(1, 99))
        _, _, bal = balance.compare_balance_before_after(X, T, fe.get_feature_names(), w)
        naive = ate.estimate_ate_naive(Y, T)
        ipw = ate.estimate_ate_ipw(Y, T, e, weights=w)
        aipw, _ = ate.estimate_ate_aipw(X, Y, T, e, model_type="linear")
        return _clean(
            {
                "confounding_strength": confounding_strength,
                "n_samples": n_samples,
                "true_ate": gt["true_ate"],
                "naive_ate": naive,
                "ipw_ate": ipw,
                "aipw_ate": aipw,
                "naive_bias": naive - gt["true_ate"],
                "aipw_bias": aipw - gt["true_ate"],
                "propensity_auc": roc_auc_score(T, e),
                "mean_smd_before": bal["mean_smd_before"],
                "mean_smd_after": bal["mean_smd_after"],
            }
        )

    def refutation(self) -> dict:
        r = self.results.get("refutation", {})
        if not r:
            raise ArtifactsMissingError("Results predate the refutation stage; re-run `make all`.")
        return _clean(
            {
                "summary": r["summary"],
                "robustness": r.get("robustness"),
                "leave_one_out": r.get("leave_one_out"),
                "backdoor_variables": r.get("dowhy", {}).get("backdoor_variables"),
                "dowhy_estimate": r.get("dowhy", {}).get("estimate"),
                "headline_ate": self.headline_ate,
            }
        )

    def calibration(self) -> dict:
        return _clean(
            {
                "cate_method": self.results.get("cate_method"),
                "deciles": self.results.get("cate_calibration"),
                "heldout_metrics": self.results.get("cate_heldout_metrics"),
            }
        )

    def segments(self) -> dict:
        if "customer_segment" not in self.df.columns:
            return {"segments": []}
        g = self.df.assign(predicted_cate=self.cate_scores, dr=self.dr_scores).groupby("customer_segment")
        rows = []
        for seg, grp in g:
            row = {
                "segment": seg,
                "n": int(len(grp)),
                "treatment_rate": float(grp["treatment"].mean()),
                "mean_predicted_cate": float(grp["predicted_cate"].mean()),
                "effect_dr": float(grp["dr"].mean()),
                "mean_outcome": float(grp["outcome"].mean()),
            }
            if self.has_ground_truth:
                row["true_effect"] = float(grp["true_treatment_effect"].mean())
            rows.append(row)
        rows.sort(key=lambda r: -r["mean_predicted_cate"])
        return _clean({"segments": rows})

    def top_customers(self, budget_fraction: float = 0.2, limit: int = 20) -> dict:
        k = max(1, int(budget_fraction * len(self.df)))
        order = np.argsort(-self.cate_scores)[:k]
        cols = [c for c in ("customer_id", "customer_segment", "region", "loyalty_score", "monetary_value") if c in self.df.columns]
        head = self.df.iloc[order[:limit]][cols].copy()
        head["predicted_cate"] = self.cate_scores[order[:limit]]
        return _clean({"budget_fraction": budget_fraction, "n_targeted": k, "customers": head})
