# path: app/streamlit_app.py
"""
Streamlit dashboard for Coupon Causal Impact Analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from coupon_causal import data, features, policy, utils

# Page config
st.set_page_config(
    page_title="Coupon Causal Impact Dashboard",
    page_icon="🎟️",
    layout="wide",
)

st.title("🎟️ Coupon Causal Impact Analysis Dashboard")
st.markdown("### Production-Quality Uplift Modeling for Retail Campaigns")


@st.cache_data
def load_data(data_source, config_path="config/default.yaml"):
    """Load and cache data."""
    config = utils.load_config(config_path)
    utils.set_random_seed(config["random_state"])

    if data_source == "Synthetic":
        df, ground_truth = data.generate_synthetic_coupon_data(
            n_samples=config["synthetic"]["n_samples"],
            treatment_rate=config["synthetic"]["treatment_rate"],
            true_ate=config["synthetic"]["true_ate"],
            random_state=config["random_state"],
        )
        return df, config, ground_truth
    else:
        # Try to load processed data
        try:
            df = pd.read_parquet("data/processed/coupon_data.parquet")
            return df, config, None
        except:
            st.error("Public data not found. Please run the pipeline first or use Synthetic data.")
            return None, None, None


@st.cache_resource
def load_results():
    """Load cached estimation results."""
    try:
        results = utils.load_artifact("models/estimation_results.joblib")
        return results
    except:
        return None


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

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 ATE Results", "🎯 Uplift & Policy", "👥 Segments"])

with tab1:
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
        st.plotly_chart(fig_treatment, use_container_width=True)

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
        st.plotly_chart(fig_outcome, use_container_width=True)

with tab2:
    st.header("Average Treatment Effect (ATE) Estimates")

    results = load_results()

    if results is None:
        st.warning("No estimation results found. Please run the pipeline first.")
        st.code("make data && make train", language="bash")
    else:
        ate_results = results["ate_results"]

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

        st.dataframe(ate_summary, use_container_width=True)

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
                marker=dict(size=12, color="steelblue"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[u - e for e, u in zip(estimates, ci_upper)],
                    arrayminus=[e - l for e, l in zip(estimates, ci_lower)],
                ),
                name="ATE",
            )
        )

        fig_ate.add_vline(x=0, line_dash="dash", line_color="red")

        fig_ate.update_layout(
            title="ATE Estimates with 95% Confidence Intervals",
            xaxis_title="Average Treatment Effect ($)",
            yaxis_title="",
            height=400,
        )

        st.plotly_chart(fig_ate, use_container_width=True)

        if ground_truth is not None:
            st.success(f"✓ True ATE: ${ground_truth['true_ate']:.2f}")

with tab3:
    st.header("Uplift Modeling & Policy Evaluation")

    results = load_results()

    if results is None:
        st.warning("No estimation results found. Please run the pipeline first.")
    else:
        cate_scores = results["cate_scores"]
        auuc = results["auuc"]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("AUUC (normalized)", f"{auuc:.3f}")

        with col2:
            st.metric("Mean CATE", f"${cate_scores.mean():.2f}")

        # CATE distribution
        fig_cate = px.histogram(
            x=cate_scores,
            nbins=50,
            title="Predicted CATE Distribution",
            labels={"x": "Predicted Treatment Effect ($)", "count": "Frequency"},
            color_discrete_sequence=["steelblue"],
        )
        fig_cate.add_vline(x=cate_scores.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
        st.plotly_chart(fig_cate, use_container_width=True)

        # Qini curve
        st.subheader("Qini Curve - Uplift Gain")

        T = df["treatment"].values
        Y = df["outcome"].values

        fractions, qini_values, random_baseline = policy.compute_qini_curve(
            cate_scores, T, Y, n_bins=20
        )

        fig_qini = go.Figure()

        fig_qini.add_trace(
            go.Scatter(
                x=fractions * 100,
                y=qini_values,
                mode="lines",
                name="Learned Policy",
                line=dict(color="steelblue", width=3),
            )
        )

        fig_qini.add_trace(
            go.Scatter(
                x=fractions * 100,
                y=random_baseline,
                mode="lines",
                name="Random Policy",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        fig_qini.update_layout(
            title="Qini Curve - Uplift Model Performance",
            xaxis_title="% of Population Targeted",
            yaxis_title="Cumulative Uplift",
            height=400,
        )

        st.plotly_chart(fig_qini, use_container_width=True)

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

        with col1:
            st.metric("Customers to Target", f"{treatment_indicator.sum():,}")

        with col2:
            avg_uplift = cate_scores[treatment_indicator == 1].mean()
            st.metric("Avg Predicted Uplift", f"${avg_uplift:.2f}")

        with col3:
            total_uplift = cate_scores[treatment_indicator == 1].sum()
            st.metric("Total Predicted Uplift", f"${total_uplift:,.2f}")

with tab4:
    st.header("Segment Analysis")

    results = load_results()

    if results is None or results.get("segment_stats") is None:
        st.info("Segment analysis not available. This requires a 'customer_segment' column.")
    else:
        segment_stats = results["segment_stats"]

        st.dataframe(segment_stats, use_container_width=True)

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

        st.plotly_chart(fig_segment, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard presents causal impact analysis for coupon campaigns using "
    "doubly-robust methods (AIPW, DRLearner, Orthogonal Random Forest) and "
    "policy learning via uplift modeling."
)

st.sidebar.markdown("### Actions")
if st.sidebar.button("Download CATE Scores"):
    results = load_results()
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
