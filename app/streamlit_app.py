# path: app/streamlit_app.py
"""
Streamlit dashboard for Coupon Causal Impact Analysis.

Reads the artifacts written by `make all` (models/estimation_results.joblib)
and walks through the causal story: the confounding bias, the assumptions
made visible, refutation checks, who to target, and validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from coupon_causal import data, policy, utils

# Page config
st.set_page_config(
    page_title="Coupon Causal Impact Dashboard",
    page_icon="🎟️",
    layout="wide",
)

st.title("🎟️ Coupon Causal Impact Analysis")
st.markdown("### Did the coupons work, for whom, and can we trust the answer?")

STRATEGY_LABELS = {
    "uplift_model": "Causal uplift model (target by CATE)",
    "response_model": "Predictive response model (target by spend)",
    "random": "Random targeting",
    "oracle (true CATE)": "Oracle (true CATE)",
}
STRATEGY_COLORS = {
    "uplift_model": "steelblue",
    "response_model": "coral",
    "random": "gray",
    "oracle (true CATE)": "seagreen",
}


@st.cache_data
def load_data(data_source, config_path="config/default.yaml"):
    """Load and cache data."""
    config = utils.load_config(config_path)
    utils.set_random_seed(config["random_state"])

    if data_source == "Synthetic":
        # Pass every generator parameter so the rows line up exactly with the
        # data the pipeline scored (models/estimation_results.joblib).
        synth = config["synthetic"]
        df, ground_truth = data.generate_synthetic_coupon_data(
            n_samples=synth["n_samples"],
            treatment_rate=synth["treatment_rate"],
            noise_level=synth.get("noise_level", 0.1),
            true_ate=synth["true_ate"],
            confounding_strength=synth.get("confounding_strength", 2.0),
            random_state=config["random_state"],
        )
        return df, config, ground_truth
    else:
        # Try to load processed data
        try:
            df = pd.read_parquet("data/processed/coupon_data.parquet")
            return df, config, None
        except (FileNotFoundError, OSError):
            st.error("Public data not found. Please run the pipeline first or use Synthetic data.")
            return None, None, None


@st.cache_resource
def load_results():
    """Load cached estimation results."""
    try:
        results = utils.load_artifact("models/estimation_results.joblib")
        return results
    except FileNotFoundError:
        return None


def missing_results():
    st.warning("No estimation results found. Please run the pipeline first.")
    st.code("make all", language="bash")


def needs_rerun(results, key):
    """True (and shows a hint) when the results file predates a feature."""
    if key in results:
        return False
    st.info("This panel needs results from the current pipeline. Re-run `make all`.")
    return True


# Sidebar
st.sidebar.header("Configuration")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic", "Public"],
    index=0,
)

df, config, ground_truth = load_data(data_source)

if df is None:
    st.stop()

results = load_results()

# Main content
(
    tab_overview,
    tab_ate,
    tab_assumptions,
    tab_policy,
    tab_targeting,
    tab_validation,
    tab_segments,
) = st.tabs(
    [
        "📊 Overview",
        "📈 ATE: naive vs causal",
        "🧭 Assumptions & refutation",
        "🎯 Uplift & policy",
        "🤖 ML vs causal targeting",
        "✅ Validation",
        "👥 Segments",
    ]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        treatment_rate = df["treatment"].mean()
        st.metric("Treatment Rate", f"{treatment_rate:.1%}")

    with col3:
        avg_outcome = df["outcome"].mean()
        st.metric("Avg Outcome", f"${avg_outcome:.2f}")

    with col4:
        if ground_truth is not None:
            st.metric("True ATE", f"${ground_truth['true_ate']:.2f}")
        else:
            st.metric("Data Source", data_source)

    st.markdown(
        """
**The story in one paragraph.** The marketing team sent coupons mostly to loyal, recent,
high-value customers. Those customers spend more anyway, so a plain treated-vs-control
comparison credits the coupon for spend that would have happened regardless. The pipeline
removes that bias with propensity weighting and doubly-robust estimation, checks that the
corrected comparison is balanced, stress-tests the estimate with placebo and sensitivity
checks, and then asks the question that matters for the budget: *whose* spend does a coupon
actually change?
"""
    )

    # Treatment distribution
    st.subheader("Treatment & Outcome Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_treatment = px.histogram(
            df,
            x="treatment",
            title="Treatment Distribution",
            labels={"treatment": "Treatment Group", "count": "Count"},
            color_discrete_sequence=["steelblue"],
        )
        st.plotly_chart(fig_treatment, width="stretch")

    with col2:
        fig_outcome = px.histogram(
            df,
            x="outcome",
            color="treatment",
            title="Outcome Distribution by Treatment",
            labels={"outcome": "Outcome ($)", "treatment": "Group"},
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig_outcome, width="stretch")

# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------
with tab_ate:
    st.header("Average Treatment Effect (ATE) Estimates")

    if results is None:
        missing_results()
    else:
        ate_results = results["ate_results"]
        headline_method = results.get("headline_ate_method", "aipw_linear")
        headline = ate_results[headline_method]["ate"]
        naive = ate_results["naive"]["ate"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Naive difference in means", f"${naive:.2f}")
        c2.metric(
            f"Causal estimate ({headline_method.replace('_', ' ').upper()})",
            f"${headline:.2f}",
            delta=f"{headline - naive:+.2f} vs naive",
            delta_color="off",
        )
        c3.metric("Naive overstates the effect by", f"{(naive / headline - 1):+.0%}")

        # Display ATE table
        ate_summary = pd.DataFrame(
            [
                {
                    "Method": method.replace("_", " ").title(),
                    "ATE": f"${res['ate']:.2f}",
                    "95% CI": f"[${res['ci_lower']:.2f}, ${res['ci_upper']:.2f}]",
                }
                for method, res in ate_results.items()
            ]
        )

        st.dataframe(ate_summary, width="stretch")

        # ATE comparison plot
        methods = [m.replace("_", " ").title() for m in ate_results.keys()]
        estimates = [res["ate"] for res in ate_results.values()]
        ci_lower = [res["ci_lower"] for res in ate_results.values()]
        ci_upper = [res["ci_upper"] for res in ate_results.values()]

        fig_ate = go.Figure()

        fig_ate.add_trace(
            go.Scatter(
                x=estimates,
                y=methods,
                mode="markers",
                marker={"size": 12, "color": "steelblue"},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [max(u - e, 0) for e, u in zip(estimates, ci_upper)],
                    "arrayminus": [max(e - lo, 0) for e, lo in zip(estimates, ci_lower)],
                },
                name="ATE",
            )
        )

        fig_ate.add_vline(x=0, line_dash="dash", line_color="red")
        if ground_truth is not None:
            fig_ate.add_vline(
                x=ground_truth["true_ate"],
                line_dash="dot",
                line_color="seagreen",
                annotation_text="True ATE",
            )

        fig_ate.update_layout(
            title="ATE Estimates with 95% Confidence Intervals",
            xaxis_title="Average Treatment Effect ($)",
            yaxis_title="",
            height=400,
        )

        st.plotly_chart(fig_ate, width="stretch")

        if ground_truth is not None:
            st.success(f"✓ True ATE: ${ground_truth['true_ate']:.2f}")

# ---------------------------------------------------------------------------
# Assumptions & refutation
# ---------------------------------------------------------------------------
with tab_assumptions:
    st.header("Can we trust the estimate?")

    if results is None:
        missing_results()
    elif not needs_rerun(results, "refutation"):
        refutation = results["refutation"]
        dowhy_res = refutation.get("dowhy", {})

        st.subheader("1. The assumed causal graph")
        st.markdown(
            "Causal inference starts from an explicit statement of what causes what. "
            "DoWhy reads this graph and derives *which variables must be adjusted for* "
            "(the backdoor set). Everything downstream conditions on exactly those."
        )
        g1, g2 = st.columns([3, 2])
        with g1:
            st.graphviz_chart(results.get("dag", ""))
        with g2:
            if dowhy_res:
                st.markdown("**Backdoor adjustment set**")
                st.write(", ".join(f"`{v}`" for v in dowhy_res["backdoor_variables"]))
                st.markdown(
                    f"**DoWhy regression-adjustment estimate:** ${dowhy_res['estimate']:.2f}"
                )
                with st.expander("Identified estimand (DoWhy output)"):
                    st.code(dowhy_res["estimand"])

        st.subheader("2. Positivity: do treated and control customers overlap?")
        e = results["propensity_scores"]
        overlap_df = pd.DataFrame({"propensity": e, "group": np.where(df["treatment"] == 1, "Treated", "Control")})
        fig_overlap = px.histogram(
            overlap_df,
            x="propensity",
            color="group",
            barmode="overlay",
            opacity=0.6,
            nbins=40,
            histnorm="probability density",
            title=f"Propensity scores by group ({results.get('propensity_model_type', 'logistic')} model)",
            labels={"propensity": "P(coupon | X)"},
            color_discrete_map={"Treated": "coral", "Control": "steelblue"},
        )
        st.plotly_chart(fig_overlap, width="stretch")
        overlap = results.get("overlap")
        if overlap:
            o1, o2, o3 = st.columns(3)
            o1.metric("Common support", f"[{overlap['common_support_min']:.2f}, {overlap['common_support_max']:.2f}]")
            o2.metric("Treated with p > 0.9", f"{overlap['treated_poor_overlap']:.1%}")
            o3.metric("Control with p < 0.1", f"{overlap['control_poor_overlap']:.1%}")

        st.subheader("3. Balance: does weighting make the groups comparable?")
        bal = pd.merge(
            results["balance_before"][["feature", "smd_before"]],
            results["balance_after"][["feature", "smd_after"]],
            on="feature",
        ).sort_values("smd_before", ascending=True).tail(15)
        fig_love = go.Figure()
        for _, row in bal.iterrows():
            fig_love.add_shape(
                type="line", x0=row["smd_before"], x1=row["smd_after"], y0=row["feature"], y1=row["feature"],
                line={"color": "lightgray", "width": 1},
            )
        fig_love.add_trace(go.Scatter(x=bal["smd_before"], y=bal["feature"], mode="markers", name="Before weighting", marker={"color": "coral", "size": 11}))
        fig_love.add_trace(go.Scatter(x=bal["smd_after"], y=bal["feature"], mode="markers", name="After weighting", marker={"color": "steelblue", "size": 11, "symbol": "square"}))
        fig_love.add_vline(x=0.1, line_dash="dash", line_color="red", annotation_text="SMD = 0.1")
        fig_love.update_layout(title="Covariate balance (Love plot)", xaxis_title="Standardized mean difference", height=520)
        st.plotly_chart(fig_love, width="stretch")
        bs = results["balance_summary"]
        b1, b2, b3 = st.columns(3)
        b1.metric("Mean SMD before", f"{bs['mean_smd_before']:.3f}")
        b2.metric("Mean SMD after", f"{bs['mean_smd_after']:.3f}")
        b3.metric("Covariates below 0.1 after", f"{bs['fraction_below_0.1_after']:.0%}")

        st.subheader("4. Refutation checks")
        summary = refutation["summary"]
        n_pass, n_total = int(summary["passed"].sum()), len(summary)
        rv = refutation.get("robustness", {})
        r1, r2, r3 = st.columns(3)
        r1.metric("Checks passed", f"{n_pass} / {n_total}")
        if rv:
            r2.metric("Robustness value", f"{rv['robustness_value']:.0%}", help=(
                "Cinelli & Hazlett (2020). An unobserved confounder would need to explain at least "
                "this share of the remaining variance of BOTH treatment and outcome to reduce the "
                "estimate to zero."
            ))
            r3.metric("Partial R² of treatment", f"{rv['partial_r2_treatment']:.0%}")

        show = summary.copy()
        show["status"] = np.where(show["passed"], "✅ pass", "❌ fail")
        show = show[["status", "test", "source", "expected", "new_effect", "p_value"]].rename(
            columns={"expected": "expected effect", "new_effect": "re-estimated effect", "p_value": "p-value"}
        )
        st.dataframe(show.round(3), width="stretch", hide_index=True)
        st.caption(
            "Placebo tests should come back near zero; stability tests should stay at the original "
            "estimate. DoWhy rows refute its regression-adjustment estimate from the graph above; "
            "pipeline rows stress the headline AIPW estimate directly."
        )
        with st.expander("Leave-one-covariate-out: which variables carry the confounding?"):
            st.dataframe(refutation["leave_one_out"].round(3), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Uplift & policy
# ---------------------------------------------------------------------------
with tab_policy:
    st.header("Uplift Modeling & Policy Evaluation")

    if results is None:
        missing_results()
    else:
        cate_scores = results["cate_scores"]
        auuc = results["auuc"]
        cate_method = results.get("cate_method", "cate")

        col1, col2, col3 = st.columns(3)
        col1.metric("CATE model", cate_method.replace("_", " ").title())
        col2.metric("AUUC (normalized, held-out)", f"{auuc:.3f}")
        col3.metric("Mean predicted CATE", f"${cate_scores.mean():.2f}")

        # CATE distribution
        fig_cate = px.histogram(
            x=cate_scores,
            nbins=50,
            title="Predicted CATE Distribution (all customers)",
            labels={"x": "Predicted Treatment Effect ($)", "count": "Frequency"},
            color_discrete_sequence=["steelblue"],
        )
        fig_cate.add_vline(x=cate_scores.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
        st.plotly_chart(fig_cate, width="stretch")

        # Qini curve (held-out when available)
        st.subheader("Qini Curve - Uplift Gain")

        if "split" in results:
            idx = results["split"]["test_idx"]
            scores_q = results["cate_scores_test"]
            qini_note = f"held-out set, {len(idx):,} customers the model never saw"
        else:
            idx = np.arange(len(df))
            scores_q = cate_scores
            qini_note = "in-sample"
        T = df["treatment"].values[idx]
        Y = df["outcome"].values[idx]

        fractions, qini_values, random_baseline = policy.compute_qini_curve(
            scores_q, T, Y, n_bins=20
        )

        fig_qini = go.Figure()
        fig_qini.add_trace(go.Scatter(x=fractions * 100, y=qini_values, mode="lines", name="Learned Policy", line={"color": "steelblue", "width": 3}))
        fig_qini.add_trace(go.Scatter(x=fractions * 100, y=random_baseline, mode="lines", name="Random Policy", line={"color": "gray", "width": 2, "dash": "dash"}))
        fig_qini.update_layout(
            title=f"Qini Curve ({qini_note})",
            xaxis_title="% of Population Targeted",
            yaxis_title="Cumulative Uplift",
            height=400,
        )
        st.plotly_chart(fig_qini, width="stretch")

        # Policy recommendations
        st.subheader("Policy Recommendations")

        budget_frac = st.slider(
            "Budget (% of customers to target)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
        )

        threshold, treatment_indicator = policy.optimal_policy_threshold(
            cate_scores,
            cost_per_treatment=1.0,
            treatment_capacity=int(budget_frac * len(df)),
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Customers to Target", f"{treatment_indicator.sum():,}")
        col2.metric("Avg Predicted Uplift", f"${cate_scores[treatment_indicator == 1].mean():.2f}")
        col3.metric("Total Predicted Uplift", f"${cate_scores[treatment_indicator == 1].sum():,.2f}")

        with st.expander("Budget table (held-out evaluation)"):
            st.dataframe(results["policy_results"].round(2), width="stretch", hide_index=True)
            st.caption(
                "`dr_*` columns use doubly-robust pseudo-outcomes and are the unbiased numbers; "
                "`observed_*` columns are the raw treated-vs-control gap inside the targeted group."
            )

# ---------------------------------------------------------------------------
# ML vs causal targeting
# ---------------------------------------------------------------------------
with tab_targeting:
    st.header("Predictive ML vs causal targeting")

    if results is None:
        missing_results()
    elif not needs_rerun(results, "targeting_comparison"):
        cmp = results["targeting_comparison"].copy()
        budget = cmp["budget_fraction"].iloc[0]
        st.markdown(
            f"""
A standard predictive model asks *"who will spend the most if we send a coupon?"* and targets
those customers. A causal uplift model asks *"whose spend will the coupon **change** the most?"*
Both models were fit on the training split; the numbers below are for the **top {budget:.0%}** of
the held-out customers each strategy picks, scored with doubly-robust pseudo-outcomes
(and against the true effect, since this data is synthetic).
"""
        )
        has_truth = "true_uplift_per_customer" in cmp.columns
        order = cmp.sort_values("dr_uplift_per_customer")
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                y=[STRATEGY_LABELS.get(s, s) for s in order["strategy"]],
                x=order["dr_uplift_per_customer"],
                orientation="h",
                name="Doubly-robust estimate",
                marker_color=[STRATEGY_COLORS.get(s, "seagreen") for s in order["strategy"]],
            )
        )
        if has_truth:
            fig_bar.add_trace(
                go.Bar(
                    y=[STRATEGY_LABELS.get(s, s) for s in order["strategy"]],
                    x=order["true_uplift_per_customer"],
                    orientation="h",
                    name="True (synthetic ground truth)",
                    marker_color="rgba(0,0,0,0.25)",
                )
            )
        fig_bar.update_layout(
            barmode="group",
            title=f"Incremental spend per targeted customer at {budget:.0%} budget",
            xaxis_title="$ per targeted customer",
            height=380,
        )
        st.plotly_chart(fig_bar, width="stretch")

        val = cmp.set_index("strategy")["dr_uplift_per_customer"]
        if {"uplift_model", "response_model", "random"} <= set(val.index):
            u, r, rnd = val["uplift_model"], val["response_model"], val["random"]
            st.markdown(
                f"**Targeting by predicted spend yields ${r:.2f} per customer, "
                f"{'below' if r < rnd else 'above'} random (${rnd:.2f}). "
                f"Targeting by predicted uplift yields ${u:.2f}, "
                f"{u / max(r, 1e-9):.1f}× the predictive model.**"
            )

        curves = results.get("targeting_curves", {})
        if curves:
            fig_curve = go.Figure()
            for name, (fr, gain) in curves.items():
                fig_curve.add_trace(go.Scatter(x=fr * 100, y=gain, mode="lines", name=STRATEGY_LABELS.get(name, name), line={"color": STRATEGY_COLORS.get(name), "width": 3}))
            fr0 = next(iter(curves.values()))[0]
            fig_curve.add_trace(go.Scatter(x=fr0 * 100, y=results["targeting_random_baseline"], mode="lines", name="Random targeting", line={"color": "gray", "dash": "dash"}))
            fig_curve.update_layout(
                title="Cumulative incremental spend by share of customers targeted (held-out, doubly-robust)",
                xaxis_title="% of customers targeted",
                yaxis_title="Cumulative incremental spend ($)",
                height=420,
            )
            st.plotly_chart(fig_curve, width="stretch")

        st.dataframe(cmp.round(2), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
with tab_validation:
    st.header("Held-out validation of the uplift model")

    if results is None:
        missing_results()
    elif not needs_rerun(results, "cate_calibration"):
        metrics = results["cate_heldout_metrics"]
        st.subheader("All CATE models on the held-out set")
        st.dataframe(metrics.round(3), width="stretch", hide_index=True)
        st.caption(
            "`auuc_heldout` uses observed outcomes (standard uplift metric); `dr_uplift_top*` is the "
            "doubly-robust incremental spend per customer among the top share by predicted effect; "
            "`corr_true_cate_heldout` is only available on synthetic data."
        )

        cal = results["cate_calibration"]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=cal["decile"], y=cal["predicted_cate"], mode="lines+markers", name="Predicted CATE", line={"color": "steelblue", "width": 3}))
        fig_cal.add_trace(
            go.Scatter(
                x=cal["decile"], y=cal["dr_estimate"], mode="markers", name="Doubly-robust estimate ± 95% CI",
                marker={"color": "coral", "size": 10, "symbol": "square"},
                error_y={"type": "data", "array": 1.96 * cal["dr_se"], "visible": True},
            )
        )
        if "true_cate" in cal.columns:
            fig_cal.add_trace(go.Scatter(x=cal["decile"], y=cal["true_cate"], mode="lines+markers", name="True effect", line={"color": "black", "dash": "dot"}))
        fig_cal.update_layout(
            title=f"Calibration by decile of predicted effect ({results.get('cate_method', 'CATE')}, held-out)",
            xaxis_title="Decile of predicted effect (1 = lowest)",
            yaxis_title="Treatment effect ($)",
            height=420,
        )
        st.plotly_chart(fig_cal, width="stretch")

        if results.get("true_cate_test") is not None:
            tc = results["true_cate_test"]
            pc = results["cate_scores_test"]
            corr = np.corrcoef(tc, pc)[0, 1]
            fig_sc = px.scatter(
                x=tc, y=pc, opacity=0.3,
                labels={"x": "True treatment effect ($)", "y": "Predicted treatment effect ($)"},
                title=f"Predicted vs true effect on held-out customers (correlation {corr:.3f})",
            )
            lo, hi = float(tc.min()), float(tc.max())
            fig_sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line={"color": "red", "dash": "dash"}))
            fig_sc.update_layout(height=450)
            st.plotly_chart(fig_sc, width="stretch")

# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
with tab_segments:
    st.header("Segment Analysis")

    if results is None or results.get("segment_stats") is None:
        st.info("Segment analysis not available. This requires a 'customer_segment' column.")
    else:
        segment_stats = results["segment_stats"]

        st.dataframe(segment_stats, width="stretch")

        # Segment uplift chart
        fig_segment = px.bar(
            segment_stats,
            x="avg_predicted_uplift",
            y="segment",
            orientation="h",
            title="Predicted Uplift by Customer Segment",
            labels={"avg_predicted_uplift": "Avg Predicted Uplift ($)", "segment": "Segment"},
            color="avg_predicted_uplift",
            color_continuous_scale="RdYlGn",
        )

        st.plotly_chart(fig_segment, width="stretch")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard presents causal impact analysis for coupon campaigns using "
    "doubly-robust methods (AIPW, DRLearner, Causal Forest), DoWhy refutation "
    "checks, and policy learning via uplift modeling."
)

st.sidebar.markdown("### Actions")
if st.sidebar.button("Download CATE Scores"):
    if results is not None:
        cate_df = df.copy()
        cate_df["predicted_cate"] = results["cate_scores"]
        csv = cate_df.to_csv(index=False)
        st.sidebar.download_button(
            "Download CSV",
            csv,
            "cate_scores.csv",
            "text/csv",
        )
