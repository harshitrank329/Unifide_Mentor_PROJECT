from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from nassau_optimization.data import CURRENT_FACTORY_BY_PRODUCT, FACTORY_COORDINATES
from nassau_optimization.modeling import evaluate_model_suite, train_model_suite
from nassau_optimization.simulation import build_route_clusters, recommend_factory_reassignments

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Nassau Candy Distributor.csv"

st.set_page_config(page_title="Nassau Candy Optimization", layout="wide")


@st.cache_resource(show_spinner=False)
def load_bundle():
    return train_model_suite(str(DATA_PATH))


@st.cache_data(show_spinner=False)
def load_raw_data():
    return pd.read_csv(DATA_PATH)


st.title("Factory Reallocation & Shipping Optimization")
st.caption("Predict lead times, simulate factory reassignment scenarios, and rank the best operational moves.")

# The dashboard uses the CSV file as its only data source.
if not DATA_PATH.exists():
    st.error("Could not find Nassau Candy Distributor.csv in the project root.")
    st.stop()

raw_df = load_raw_data()
bundle = load_bundle()
metrics_df = evaluate_model_suite(bundle, str(DATA_PATH))

with st.sidebar:
    st.header("Scenario Controls")
    # These filters let the user test one product and one shipping setup at a time.
    product_name = st.selectbox("Product", sorted(raw_df["Product Name"].dropna().unique().tolist()))
    region = st.selectbox("Region", ["All"] + sorted(raw_df["Region"].dropna().unique().tolist()))
    ship_mode = st.selectbox("Ship Mode", ["All"] + sorted(raw_df["Ship Mode"].dropna().unique().tolist()))
    speed_weight = st.slider("Speed vs Profit", 0.0, 1.0, 0.7, 0.05)
    top_n = st.slider("Top recommendations", 3, 10, 5)

region_value = None if region == "All" else region
ship_mode_value = None if ship_mode == "All" else ship_mode

st.subheader("Model Selection")
left, middle, right = st.columns(3)
left.metric("Champion Model", bundle.champion_name)
middle.metric("Dataset Rows", f"{len(raw_df):,}")
right.metric("Unique Products", f"{raw_df['Product Name'].nunique():,}")

st.dataframe(metrics_df, width="stretch", hide_index=True)

recommendations = recommend_factory_reassignments(
    str(DATA_PATH),
    bundle,
    product_name=product_name,
    region=region_value,
    ship_mode=ship_mode_value,
    speed_weight=speed_weight,
    top_n=top_n,
)

st.subheader("KPI Overview")
# These summary cards turn the model output into simple business metrics.
best_overall = recommendations.iloc[0] if not recommendations.empty else None
kpi_cols = st.columns(4)
kpi_cols[0].metric(
    "Lead Time Reduction (%)",
    f"{best_overall['lead_time_reduction_pct']:.2f}" if best_overall is not None else "—",
)
kpi_cols[1].metric(
    "Profit Impact Stability",
    f"{(1.0 - float(best_overall['risk_score'])) * 100:.1f}%" if best_overall is not None else "—",
)
kpi_cols[2].metric(
    "Scenario Confidence Score",
    f"{best_overall['scenario_confidence']:.2f}" if best_overall is not None else "—",
)
kpi_cols[3].metric(
    "Recommendation Coverage",
    f"{len(recommendations) / max(1, len(FACTORY_COORDINATES) - 1) * 100:.1f}%" if best_overall is not None else "—",
)

with st.expander("Factory reference data", expanded=False):
    # This section shows the known factory locations used in the scenario analysis.
    factory_frame = pd.DataFrame(
        [
            {"Factory": factory, "Latitude": coords[0], "Longitude": coords[1]}
            for factory, coords in FACTORY_COORDINATES.items()
        ]
    )
    ref_left, ref_right = st.columns([1, 1])
    with ref_left:
        factory_fig = px.scatter_geo(
            factory_frame,
            lat="Latitude",
            lon="Longitude",
            text="Factory",
            title="Factory Locations",
        )
        factory_fig.update_traces(marker=dict(size=12, color="#7c3aed"))
        factory_fig.update_geos(showland=True, landcolor="rgb(240, 240, 240)")
        st.plotly_chart(factory_fig, width="stretch")
    with ref_right:
        st.dataframe(factory_frame, width="stretch", hide_index=True)

    mapping_rows = [
        {"Division": division, "Product Name": product, "Factory": factory}
        for product, factory in CURRENT_FACTORY_BY_PRODUCT.items()
        for division in raw_df.loc[raw_df["Product Name"] == product, "Division"].dropna().unique()[:1]
    ]
    mapping_frame = pd.DataFrame(mapping_rows).drop_duplicates().sort_values(["Division", "Product Name"])
    st.caption("Products and factories correlation provided for scenario simulation and current-assignment baselining.")
    st.dataframe(mapping_frame, width="stretch", hide_index=True)

tabs = st.tabs(["Factory Optimization Simulator", "What-If Scenario Analysis", "Recommendation Dashboard", "Risk & Impact Panel"])

with tabs[0]:
    # The simulator compares predicted lead time across alternate factories.
    if recommendations.empty:
        st.warning("No matching history found for this product and filter combination.")
    else:
        sim_left, sim_right = st.columns([1, 1])
        with sim_left:
            chart_df = recommendations[["recommended_factory", "current_predicted_lead_time", "recommended_predicted_lead_time"]].copy()
            chart_df = chart_df.melt("recommended_factory", var_name="scenario", value_name="predicted_lead_time")
            fig = px.bar(
                chart_df,
                x="recommended_factory",
                y="predicted_lead_time",
                color="scenario",
                barmode="group",
                title="Predicted Lead Time by Factory",
            )
            st.plotly_chart(fig, width="stretch")
        with sim_right:
            sim_table = recommendations[[
                "recommended_factory",
                "current_predicted_lead_time",
                "recommended_predicted_lead_time",
                "lead_time_reduction_pct",
                "composite_score",
            ]].copy()
            st.dataframe(sim_table.round(2), width="stretch", hide_index=True)

with tabs[1]:
    # This tab compares the current assignment with the best recommendation.
    if recommendations.empty:
        st.warning("No scenario comparison is available for this filter combination.")
    else:
        best = recommendations.iloc[0]
        comparison = pd.DataFrame(
            {
                "Assignment": ["Current", "Recommended"],
                "Predicted Lead Time": [best["current_predicted_lead_time"], best["recommended_predicted_lead_time"]],
                "Scenario Confidence": [best["scenario_confidence"], best["scenario_confidence"]],
                "Profit Impact Proxy": [0.0, best["profit_impact_proxy"]],
            }
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(
                comparison,
                x="Assignment",
                y="Predicted Lead Time",
                color="Assignment",
                title="Current vs Recommended Lead Time",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(comparison.round(2), width="stretch", hide_index=True)
        st.success(
            f"Recommended reassignment: {best['current_factory']} → {best['recommended_factory']} for {best['product_name']}."
        )

with tabs[2]:
    # This table is the ranked list of suggested factory changes.
    if recommendations.empty:
        st.info("No ranked recommendations available.")
    else:
        display_df = recommendations.copy()
        display_df["lead_time_reduction_pct"] = display_df["lead_time_reduction_pct"].round(2)
        display_df["profit_impact_proxy"] = display_df["profit_impact_proxy"].round(2)
        display_df["risk_score"] = display_df["risk_score"].round(3)
        display_df["composite_score"] = display_df["composite_score"].round(2)
        display_df["scenario_confidence"] = display_df["scenario_confidence"].round(3)
        st.dataframe(display_df, width="stretch", hide_index=True)
        st.download_button(
            "Download recommendations as CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="nassau_recommendations.csv",
            mime="text/csv",
        )

        best = display_df.iloc[0]
        summary_cards = st.columns(3)
        summary_cards[0].metric("Expected Efficiency Gain", f"{best['lead_time_reduction_pct']:.2f}%")
        summary_cards[1].metric("Risk-Adjusted Score", f"{best['composite_score']:.2f}")
        summary_cards[2].metric("Confidence", f"{best['scenario_confidence']:.2f}")

with tabs[3]:
    # This view highlights risk so the user can avoid unsafe recommendations.
    if recommendations.empty:
        st.warning("Risk analysis is unavailable for this selection.")
    else:
        best = recommendations.iloc[0]
        alert_col, warning_col = st.columns(2)
        alert_col.metric("Profit Impact Stability", f"{(1.0 - float(best['risk_score'])) * 100:.1f}%")
        warning_col.metric("High-Risk Warning", "Yes" if best["risk_score"] > 0.6 else "No")

        if best["risk_score"] > 0.6:
            st.warning(
                f"High-risk reassignment: {best['recommended_factory']} may reduce lead time but carries elevated profit uncertainty."
            )
        else:
            st.success("Risk level is acceptable for operational review.")

        risk_chart = pd.DataFrame(
            {
                "Scenario": ["Current", "Recommended"],
                "Risk Score": [best["risk_score"], max(0.0, best["risk_score"] - best["scenario_confidence"] / 2)],
            }
        )
        risk_fig = px.line(risk_chart, x="Scenario", y="Risk Score", markers=True, title="Risk Comparison")
        st.plotly_chart(risk_fig, width="stretch")

with st.expander("Route clustering", expanded=False):
    # Clusters help students spot slow or congested route groups.
    clusters = build_route_clusters(str(DATA_PATH))
    cluster_focus = clusters.loc[clusters["Product Name"] == product_name].head(20)
    if not cluster_focus.empty:
        st.dataframe(cluster_focus, width="stretch", hide_index=True)
    else:
        st.info("Route clustering is available, but this product does not have a summarized cluster view yet.")
