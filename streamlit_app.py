"""
Superstore Sales Forecasting & Analytics Dashboard
Task 7 — Deployment: Interactive Dashboard using Streamlit

Pages:
  1. Sales Overview Dashboard
  2. Forecast Explorer
  3. Anomaly Report
  4. Product Demand Segments
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Superstore Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
ART_DIR = "artifacts"


# --------------------------------------------------------------------------
# Cached data loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_raw():
    df = pd.read_csv(f"{DATA_DIR}/clean.csv", parse_dates=["Order Date"])
    df["YearMonth"] = df["Order Date"].values.astype("datetime64[M]")
    return df


@st.cache_data
def load_monthly_overall():
    return pd.read_csv(f"{DATA_DIR}/monthly_overall.csv", parse_dates=["ds"])


@st.cache_data
def load_yearly():
    return pd.read_csv(f"{DATA_DIR}/yearly_sales.csv")


@st.cache_data
def load_model_comparison():
    return pd.read_csv(f"{ART_DIR}/model_comparison.csv")


@st.cache_data
def load_segment_forecasts():
    with open(f"{ART_DIR}/segment_forecasts.json") as f:
        return json.load(f)


@st.cache_data
def load_anomalies():
    return pd.read_csv(f"{ART_DIR}/anomalies_full.csv", parse_dates=["ds"])


@st.cache_data
def load_clusters():
    return pd.read_csv(f"{ART_DIR}/clusters.csv")


@st.cache_data
def load_subcat_pattern():
    return pd.read_csv(f"{ART_DIR}/subcat_monthly_pattern.csv")


@st.cache_data
def load_sarima_artifacts():
    actual = pd.read_csv(f"{ART_DIR}/sarima_actual.csv", parse_dates=["ds"])
    test_fc = pd.read_csv(f"{ART_DIR}/sarima_test_forecast.csv", parse_dates=["ds"])
    future_fc = pd.read_csv(f"{ART_DIR}/sarima_future_forecast.csv", parse_dates=["ds"])
    with open(f"{ART_DIR}/sarima_metrics.json") as f:
        metrics = json.load(f)
    return actual, test_fc, future_fc, metrics


@st.cache_data
def load_prophet_artifacts():
    full_fc = pd.read_csv(f"{ART_DIR}/prophet_full_forecast.csv", parse_dates=["ds"])
    weekly = pd.read_csv(f"{ART_DIR}/prophet_weekly_seasonality.csv")
    yearly = pd.read_csv(f"{ART_DIR}/prophet_yearly_seasonality.csv")
    with open(f"{ART_DIR}/prophet_metrics.json") as f:
        metrics = json.load(f)
    return full_fc, weekly, yearly, metrics


@st.cache_data
def load_xgb_artifacts():
    test_fc = pd.read_csv(f"{ART_DIR}/xgb_test_forecast.csv", parse_dates=["ds"])
    future_fc = pd.read_csv(f"{ART_DIR}/xgb_future_forecast.csv", parse_dates=["ds"])
    with open(f"{ART_DIR}/xgb_metrics.json") as f:
        metrics = json.load(f)
    return test_fc, future_fc, metrics


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("📦 Superstore Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "1️⃣  Sales Overview",
        "2️⃣  Forecast Explorer",
        "3️⃣  Anomaly Report",
        "4️⃣  Product Demand Segments",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Superstore order-level sales (2015–2018). "
    "Forecasting models: SARIMA, Prophet, XGBoost. "
    "Built for Task 7 — Streamlit Deployment."
)

# ==========================================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ==========================================================================
if page.startswith("1"):
    st.title("📊 Sales Overview Dashboard")
    df = load_raw()
    yearly = load_yearly()
    overall = load_monthly_overall()

    total_sales = df["Sales"].sum()
    n_orders = df["Order ID"].nunique()
    avg_order = df.groupby("Order ID")["Sales"].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Orders", f"{n_orders:,}")
    c3.metric("Avg. Order Value", f"${avg_order:,.2f}")
    c4.metric("Date Range", f"{df['Order Date'].dt.year.min()}–{df['Order Date'].dt.year.max()}")

    st.markdown("### Total Sales by Year")
    fig_year = px.bar(
        yearly, x="Year", y="Sales", text_auto=".2s",
        color="Sales", color_continuous_scale="Blues",
    )
    fig_year.update_layout(showlegend=False, yaxis_title="Sales ($)", coloraxis_showscale=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Monthly Sales Trend")
    fig_trend = px.line(overall, x="ds", y="y", markers=True)
    fig_trend.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Sales by Region & Category (Interactive Filters)")
    fcol1, fcol2 = st.columns(2)
    regions = fcol1.multiselect(
        "Filter by Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique())
    )
    categories = fcol2.multiselect(
        "Filter by Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
    )

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        reg_cat = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig_reg = px.bar(
            reg_cat, x="Region", y="Sales", color="Category", barmode="group",
            title="Sales by Region & Category",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    with gcol2:
        cat_total = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(cat_total, names="Category", values="Sales", title="Category Share", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Sub-Category Breakdown (Filtered)")
    subcat_total = (
        filtered.groupby("Sub-Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=True)
    )
    fig_subcat = px.bar(subcat_total, x="Sales", y="Sub-Category", orientation="h")
    st.plotly_chart(fig_subcat, use_container_width=True)

# ==========================================================================
# PAGE 2 — FORECAST EXPLORER
# ==========================================================================
elif page.startswith("2"):
    st.title("🔮 Forecast Explorer")
    st.caption(
        "Forecasts below use **XGBoost** — the best-performing model overall "
        "(see Model Comparison in the notebook / summary below)."
    )

    seg = load_segment_forecasts()

    ecol1, ecol2 = st.columns(2)
    dim = ecol1.selectbox("Select dimension", ["Category", "Region"])
    options = list(seg[dim].keys())
    key = ecol2.selectbox(f"Select {dim}", options)

    horizon = st.select_slider(
        "Forecast horizon (months ahead)", options=[1, 2, 3], value=3
    )

    data = seg[dim][key]
    hist_dates = pd.to_datetime(data["history"]["ds"])
    hist_sales = data["history"]["sales"]
    future_dates = pd.to_datetime(data["future"]["ds"])[:horizon]
    future_fc = data["future"]["forecast"][:horizon]
    test_dates = pd.to_datetime(data["test"]["ds"])
    test_actual = data["test"]["actual"]
    test_fc = data["test"]["forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_sales, mode="lines+markers", name="Historical Sales"))
    fig.add_trace(go.Scatter(
        x=list(test_dates), y=test_fc, mode="lines+markers", name="Validation Forecast",
        line=dict(dash="dot", color="orange"),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates), y=future_fc, mode="lines+markers", name=f"Future Forecast ({horizon}mo)",
        line=dict(dash="dash", color="green"), marker=dict(size=10),
    ))
    fig.update_layout(
        title=f"{dim}: {key} — Sales Forecast",
        xaxis_title="Month", yaxis_title="Sales ($)", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Forecasted Values")
    fc_table = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in future_dates],
        "Forecasted Sales ($)": [round(v, 2) for v in future_fc],
    })
    st.table(fc_table)

    st.markdown("#### Model Performance (on held-out last 3 months)")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"${data['MAE']:,.2f}")
    m2.metric("RMSE", f"${data['RMSE']:,.2f}")

    with st.expander("📋 Overall Model Comparison (Total Sales, All Segments)"):
        comp = load_model_comparison()
        st.dataframe(comp, use_container_width=True)
        best_row = comp.loc[comp["MAE"].idxmin()]
        st.success(
            f"**Recommended for production:** {best_row['Model']} — lowest MAE "
            f"(${best_row['MAE']:,.2f}) and MAPE ({best_row['MAPE (%)']}%) on the held-out test months."
        )

# ==========================================================================
# PAGE 3 — ANOMALY REPORT
# ==========================================================================
elif page.startswith("3"):
    st.title("🚨 Anomaly Report")
    st.caption(
        "Anomalies are detected on monthly total sales via seasonal decomposition: "
        "the residual component (actual − trend − seasonality) is standardized, and any month "
        "with |z-score| > 2.0 is flagged."
    )

    anomalies = load_anomalies()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomalies["ds"], y=anomalies["sales"], mode="lines+markers", name="Monthly Sales",
        line=dict(color="steelblue"),
    ))
    flagged = anomalies[anomalies["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=flagged["ds"], y=flagged["sales"], mode="markers", name="Anomaly",
        marker=dict(color="red", size=14, symbol="x"),
    ))
    fig.update_layout(title="Sales Over Time — Anomalies Highlighted", xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Detected Anomalies ({len(flagged)} of {len(anomalies)} months)")
    display_tbl = flagged[["ds", "sales", "zscore"]].copy()
    display_tbl.columns = ["Date", "Sales ($)", "Z-Score"]
    display_tbl["Date"] = display_tbl["Date"].dt.strftime("%b %Y")
    display_tbl["Sales ($)"] = display_tbl["Sales ($)"].round(2)
    display_tbl["Z-Score"] = display_tbl["Z-Score"].round(2)
    st.table(display_tbl)

    if len(flagged):
        for _, row in flagged.iterrows():
            direction = "spike above" if row["zscore"] > 0 else "dip below"
            st.info(
                f"**{row['ds'].strftime('%B %Y')}**: sales of ${row['sales']:,.0f} is a "
                f"{direction} the seasonal trend (z = {row['zscore']:.2f})."
            )

# ==========================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ==========================================================================
elif page.startswith("4"):
    st.title("🧩 Product Demand Segments")
    st.caption(
        "Sub-categories are clustered (K-Means, k=4) using total sales, average monthly sales, "
        "demand volatility (coefficient of variation), and trend slope."
    )

    clusters = load_clusters()
    pattern = load_subcat_pattern()

    st.markdown("### Cluster Chart — Sales vs. Volatility")
    fig = px.scatter(
        clusters, x="avg_monthly", y="volatility_cv", size="total_sales", color="cluster_name",
        hover_name="Sub-Category", text="Sub-Category",
        labels={"avg_monthly": "Avg Monthly Sales ($)", "volatility_cv": "Volatility (CV)"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Demand Pattern by Sub-Category")
    pattern_long = pattern.melt(id_vars="Sub-Category", var_name="Month", value_name="Sales")
    pattern_long["Month"] = pd.to_datetime(pattern_long["Month"])
    merged = pattern_long.merge(clusters[["Sub-Category", "cluster_name"]], on="Sub-Category")
    selected_cluster = st.selectbox("Highlight cluster", ["All"] + sorted(clusters["cluster_name"].unique()))
    plot_df = merged if selected_cluster == "All" else merged[merged["cluster_name"] == selected_cluster]
    fig2 = px.line(plot_df, x="Month", y="Sales", color="Sub-Category")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Sub-Categories by Demand Cluster")
    tbl = clusters[["Sub-Category", "cluster_name", "total_sales", "avg_monthly", "volatility_cv", "trend_slope"]].copy()
    tbl.columns = ["Sub-Category", "Cluster", "Total Sales ($)", "Avg Monthly Sales ($)", "Volatility (CV)", "Trend Slope"]
    tbl = tbl.sort_values(["Cluster", "Total Sales ($)"], ascending=[True, False])
    for col in ["Total Sales ($)", "Avg Monthly Sales ($)"]:
        tbl[col] = tbl[col].round(0)
    for col in ["Volatility (CV)", "Trend Slope"]:
        tbl[col] = tbl[col].round(3)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Cluster interpretations"):
        st.markdown(
            """
- **Core High-Demand Growth**: Chairs, Accessories, Binders, Storage, Phones, Tables — highest total
  sales, strong positive trend, moderate volatility. Prioritize inventory and marketing here.
- **Stable Low-Demand**: Appliances, Bookcases, Art, Envelopes, Labels, Fasteners, Furnishings, Paper —
  lower, steady sales with low volatility. Good candidates for lean/just-in-time stocking.
- **Niche High-Growth**: Copiers — small buyer base but high per-order value and strong growth;
  volatile month-to-month.
- **Volatile Low-Growth**: Machines, Supplies — high volatility, flat-to-declining trend. Needs closer
  demand monitoring before committing inventory.
            """
        )
