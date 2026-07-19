"""
Superstore Sales Intelligence Dashboard
4 Pages: Sales Overview | Forecast Explorer | Anomaly Report | Demand Segments
Author: Tamilarasan P
Live: https://superstore-analytics-app.streamlit.app/
"""

import os, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve paths (works locally AND on Streamlit Cloud) ─────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
ART  = os.path.join(BASE, "artifacts")

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px;
    border-top: 3px solid #58a6ff;
}
.metric-card.green  { border-top-color: #3fb950; }
.metric-card.amber  { border-top-color: #d29922; }
.metric-card.purple { border-top-color: #bc8cff; }
.metric-val { font-size: 28px; font-weight: 700; margin-top: 4px; }
.metric-lbl { font-size: 11px; font-weight: 600; color: #8b949e;
              text-transform: uppercase; letter-spacing: .06em; }
.metric-sub { font-size: 12px; color: #8b949e; margin-top: 3px; }
.badge { display:inline-block; font-size:10px; padding:2px 8px;
         border-radius:20px; font-weight:600; letter-spacing:.04em;
         background:rgba(88,166,255,.12); color:#58a6ff; }
.badge-g { background:rgba(63,185,80,.12); color:#3fb950; }
.badge-r { background:rgba(248,81,73,.12);  color:#f85149; }
.badge-a { background:rgba(210,153,34,.12); color:#d29922; }
.rec-box { background:rgba(63,185,80,.07); border:1px solid rgba(63,185,80,.25);
           border-radius:8px; padding:14px 18px; margin-top:12px; }
.rec-title { font-weight:700; color:#3fb950; margin-bottom:5px; }
.anom-alert { background:rgba(248,81,73,.08); border:1px solid rgba(248,81,73,.3);
              border-radius:8px; padding:14px 18px; margin-bottom:16px; }
div[data-testid="stHorizontalBlock"] > div { padding: 4px 6px; }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    df = pd.read_csv(os.path.join(DATA, "clean.csv"), parse_dates=["Order Date"])
    df["YearMonth"] = df["Order Date"].values.astype("datetime64[M]")
    return df

@st.cache_data
def load_monthly():
    return pd.read_csv(os.path.join(DATA, "monthly_overall.csv"), parse_dates=["ds"])

@st.cache_data
def load_yearly():
    return pd.read_csv(os.path.join(DATA, "yearly_sales.csv"))

@st.cache_data
def load_by_cat():
    return pd.read_csv(os.path.join(DATA, "monthly_by_category.csv"), parse_dates=["YearMonth"])

@st.cache_data
def load_by_reg():
    return pd.read_csv(os.path.join(DATA, "monthly_by_region.csv"), parse_dates=["YearMonth"])

@st.cache_data
def load_by_subcat():
    return pd.read_csv(os.path.join(DATA, "monthly_by_subcat.csv"), parse_dates=["YearMonth"])

@st.cache_data
def load_anomalies():
    return pd.read_csv(os.path.join(ART, "anomalies_full.csv"), parse_dates=["ds"])

@st.cache_data
def load_clusters():
    return pd.read_csv(os.path.join(ART, "clusters.csv"))

@st.cache_data
def load_subcat_pattern():
    return pd.read_csv(os.path.join(ART, "subcat_monthly_pattern.csv"))

@st.cache_data
def load_model_comparison():
    return pd.read_csv(os.path.join(ART, "model_comparison.csv"))

@st.cache_data
def load_seg_forecasts():
    with open(os.path.join(ART, "segment_forecasts.json")) as f:
        return json.load(f)

@st.cache_data
def load_xgb_future():
    return pd.read_csv(os.path.join(ART, "xgb_future_forecast.csv"), parse_dates=["ds"])

@st.cache_data
def load_xgb_test():
    return pd.read_csv(os.path.join(ART, "xgb_test_forecast.csv"), parse_dates=["ds"])

# ── Plotly theme ──────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
)

COLORS = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "amber":  "#d29922",
    "red":    "#f85149",
    "purple": "#bc8cff",
    "teal":   "#39d353",
}

def fmt(v):
    """Format dollar value."""
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Superstore Analytics")
    st.markdown("**Sales Intelligence Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["1️⃣  Sales Overview", "2️⃣  Forecast Explorer",
         "3️⃣  Anomaly Report", "4️⃣  Demand Segments"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
**Data Source**
- `train.csv` · 9,800 rows
- 2015–2018 · 17 sub-categories

**Models**
- SARIMA(0,1,1)(0,1,1)[12]
- Facebook Prophet
- ⭐ XGBoost (best)
    """)
    st.markdown("---")
    st.caption("Built by Tamilarasan P · [GitHub](https://github.com/tamil1208)")

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page.startswith("1"):
    st.title("📊 Sales Overview Dashboard")
    st.caption("Total performance across 2015–2018 · Values calculated from train.csv (9,800 rows)")

    df      = load_raw()
    yearly  = load_yearly()
    monthly = load_monthly()
    by_cat  = load_by_cat()
    by_reg  = load_by_reg()

    # ── KPI cards ──
    total_sales   = df["Sales"].sum()
    total_orders  = df["Order ID"].nunique()
    total_cust    = df["Customer ID"].nunique()
    avg_order     = df.groupby("Order ID")["Sales"].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Total Revenue</div>
            <div class="metric-val">{fmt(total_sales)}</div>
            <div class="metric-sub">Across all 4 years</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-lbl">Total Orders</div>
            <div class="metric-val">{total_orders:,}</div>
            <div class="metric-sub">Unique order IDs</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-lbl">Customers</div>
            <div class="metric-val">{total_cust:,}</div>
            <div class="metric-sub">Avg order {fmt(avg_order)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        best_year = int(df.groupby(df["Order Date"].dt.year)["Sales"].sum().idxmax())
        best_val  = df.groupby(df["Order Date"].dt.year)["Sales"].sum().max()
        st.markdown(f"""<div class="metric-card purple">
            <div class="metric-lbl">Best Year</div>
            <div class="metric-val">{best_year}</div>
            <div class="metric-sub">{fmt(best_val)} · +20% YoY</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Yearly bar + Monthly trend ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Total Sales by Year")
        bar_colors = [COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["green"]]
        fig = go.Figure(go.Bar(
            x=yearly["Year"].astype(str), y=yearly["Sales"],
            marker_color=bar_colors, marker_line_width=0,
            text=[fmt(v) for v in yearly["Sales"]],
            textposition="outside", textfont=dict(color="#e6edf3"),
        ))
        fig.update_layout(**LAYOUT, height=280, showlegend=False)
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Monthly Sales Trend (2015–2018)")
        fig = go.Figure(go.Scatter(
            x=monthly["ds"], y=monthly["y"],
            mode="lines", fill="tozeroy",
            line=dict(color=COLORS["blue"], width=2),
            fillcolor="rgba(88,166,255,0.1)",
        ))
        fig.update_layout(**LAYOUT, height=280)
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Region + Category + Segment ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### By Region")
        reg_total = df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig = px.bar(reg_total, x="Region", y="Sales",
                     color="Sales", color_continuous_scale=[[0,"#1e3a5f"],[1,COLORS["blue"]]],
                     text=reg_total["Sales"].apply(fmt))
        fig.update_layout(**LAYOUT, height=260, coloraxis_showscale=False, showlegend=False)
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### By Category")
        cat_total = df.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(cat_total, names="Category", values="Sales",
                     color_discrete_sequence=[COLORS["amber"], COLORS["blue"], COLORS["green"]],
                     hole=0.6)
        fig.update_layout(**LAYOUT, height=260)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("#### By Segment")
        seg_total = df.groupby("Segment")["Sales"].sum().reset_index()
        fig = px.pie(seg_total, names="Segment", values="Sales",
                     color_discrete_sequence=[COLORS["blue"], COLORS["purple"], COLORS["teal"]],
                     hole=0.6)
        fig.update_layout(**LAYOUT, height=260)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # ── Filters: Region & Category ──
    st.markdown("---")
    st.markdown("#### 🔽 Interactive Filters — Sales by Region & Category")
    f1, f2 = st.columns(2)
    sel_regions = f1.multiselect("Filter by Region", sorted(df["Region"].unique()),
                                  default=sorted(df["Region"].unique()))
    sel_cats    = f2.multiselect("Filter by Category", sorted(df["Category"].unique()),
                                  default=sorted(df["Category"].unique()))
    filtered = df[df["Region"].isin(sel_regions) & df["Category"].isin(sel_cats)]

    col1, col2 = st.columns(2)
    with col1:
        rc = filtered.groupby(["Region","Category"])["Sales"].sum().reset_index()
        fig = px.bar(rc, x="Region", y="Sales", color="Category", barmode="group",
                     color_discrete_sequence=[COLORS["amber"], COLORS["blue"], COLORS["green"]])
        fig.update_layout(**LAYOUT, height=280, title="Region × Category")
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sc = filtered.groupby("Sub-Category")["Sales"].sum().reset_index().sort_values("Sales")
        fig = px.bar(sc, x="Sales", y="Sub-Category", orientation="h",
                     color="Sales", color_continuous_scale=[[0,"#1e3a5f"],[1,COLORS["blue"]]])
        fig.update_layout(**LAYOUT, height=280, title="Sub-Category Breakdown",
                          coloraxis_showscale=False)
        fig.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    # ── Ship mode + Top customers ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Ship Mode Distribution")
        ship = df.groupby("Ship Mode")["Order ID"].nunique().reset_index()
        ship.columns = ["Ship Mode","Orders"]
        fig = px.bar(ship.sort_values("Orders"), x="Orders", y="Ship Mode", orientation="h",
                     color="Orders", color_continuous_scale=[[0,"#1e3a5f"],[1,COLORS["purple"]]])
        fig.update_layout(**LAYOUT, height=240, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Top 10 Customers by Revenue")
        top10 = df.groupby("Customer Name")["Sales"].sum().nlargest(10).reset_index()
        fig = px.bar(top10.sort_values("Sales"), x="Sales", y="Customer Name",
                     orientation="h", color="Sales",
                     color_continuous_scale=[[0,"#1e3a5f"],[1,COLORS["teal"]]])
        fig.update_layout(**LAYOUT, height=280, coloraxis_showscale=False)
        fig.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ════════════════════════════════════════════════════════════════════
elif page.startswith("2"):
    st.title("🔮 Forecast Explorer")
    st.caption("XGBoost walk-forward lag-feature forecasting · per Category / Region · 1–3 month horizon")

    seg_fc   = load_seg_forecasts()
    monthly  = load_monthly()
    xgb_test = load_xgb_test()
    xgb_fut  = load_xgb_future()
    comp     = load_model_comparison()

    # ── Controls ──
    c1, c2, c3 = st.columns([1.2, 1.5, 1.5])
    dim     = c1.selectbox("Dimension", ["Category", "Region"])
    options = list(seg_fc[dim].keys())
    key     = c2.selectbox(f"Select {dim}", options)
    horizon = c3.select_slider("Forecast Horizon", options=[1, 2, 3], value=3)

    # ── Metric cards ──
    data = seg_fc[dim][key]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${data['MAE']:,.0f}")
    m2.metric("RMSE", f"${data['RMSE']:,.0f}")
    m3.metric("Model", "XGBoost ⭐")

    # ── Forecast chart ──
    hist_dates  = pd.to_datetime(data["history"]["ds"])
    hist_sales  = data["history"]["sales"]
    fut_dates   = pd.to_datetime(data["future"]["ds"])[:horizon]
    fut_fc      = data["future"]["forecast"][:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_sales, name="Historical Sales",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=fut_fc, name=f"Forecast ({horizon} month{'s' if horizon>1 else ''})",
        mode="lines+markers",
        line=dict(color=COLORS["green"], width=2.5, dash="dash"),
        marker=dict(size=10, color=COLORS["green"]),
    ))
    fig.update_layout(**LAYOUT, height=340, hovermode="x unified",
                      title=f"{dim}: {key} — Sales + {horizon}-Month Forecast")
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast values table ──
    st.markdown("#### Forecasted Values")
    fc_df = pd.DataFrame({
        "Month":            [d.strftime("%b %Y") for d in fut_dates],
        "Forecasted Sales": [f"${v:,.0f}" for v in fut_fc],
    })
    st.table(fc_df)

    # ── Model comparison ──
    st.markdown("---")
    st.markdown("#### 📊 Model Comparison — Holdout Test (Oct–Dec 2018)")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""**SARIMA(0,1,1)(0,1,1)[12]**""")
        st.metric("MAE",  "$19,299", delta=None)
        st.metric("RMSE", "$20,206", delta=None)
        st.metric("MAPE", "20.4%",   delta=None)
    with mc2:
        st.markdown("""**Prophet**""")
        st.metric("MAE",  "$21,672", delta=None)
        st.metric("RMSE", "$22,618", delta=None)
        st.metric("MAPE", "24.4%",   delta=None)
    with mc3:
        st.markdown("""**⭐ XGBoost (Best)**""")
        st.metric("MAE",  "$16,977", delta="-$2,322 vs SARIMA", delta_color="inverse")
        st.metric("RMSE", "$20,558", delta=None)
        st.metric("MAPE", "16.7%",   delta="-3.7pp vs SARIMA",  delta_color="inverse")

    with st.expander("📋 Full Model Comparison Table"):
        st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("""<div class="rec-box">
        <div class="rec-title">✅ Recommended for Production: XGBoost</div>
        <div>Lowest MAE ($16,977) and MAPE (16.7%) on the held-out test period.
        RMSE within 2% of SARIMA. Scales cleanly to per-Category and per-Region
        forecasting using lag features: lag1, lag2, lag3, 3-month rolling mean,
        month, quarter, season.</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ════════════════════════════════════════════════════════════════════
elif page.startswith("3"):
    st.title("🚨 Anomaly Report")
    st.caption("Seasonal decomposition (additive, period=12) · residual z-score threshold |z| > 2.0")

    anomalies = load_anomalies()
    flagged   = anomalies[anomalies["is_anomaly"]]

    # ── Alert ──
    st.markdown(f"""<div class="anom-alert">
        <strong style="color:#f85149">🚨 {len(flagged)} Anomalies Detected in {len(anomalies)} Months ({len(flagged)/len(anomalies)*100:.1f}%)</strong><br>
        <span style="color:#8b949e">May 2017 and November 2018 show unusual positive sales spikes beyond seasonal expectations.
        Both detected via seasonal decomposition residual z-scores exceeding ±2.0.</span>
    </div>""", unsafe_allow_html=True)

    # ── Anomaly chart ──
    st.markdown("#### Monthly Sales — Anomalies Highlighted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomalies["ds"], y=anomalies["sales"], name="Monthly Sales",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=flagged["ds"], y=flagged["sales"], name="Anomaly",
        mode="markers",
        marker=dict(color=COLORS["red"], size=14, symbol="x",
                    line=dict(width=3, color=COLORS["red"])),
    ))
    fig.update_layout(**LAYOUT, height=320, hovermode="x unified")
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Z-score chart ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Z-Score Distribution")
        colors = [COLORS["red"] if abs(z) > 2 else COLORS["amber"] if abs(z) > 1.5 else COLORS["blue"]
                  for z in anomalies["zscore"]]
        fig = go.Figure(go.Bar(
            x=anomalies["ds"], y=anomalies["zscore"],
            marker_color=colors, marker_line_width=0,
        ))
        fig.add_hline(y=2.0,  line=dict(color=COLORS["red"],  dash="dash", width=1.5))
        fig.add_hline(y=-2.0, line=dict(color=COLORS["red"],  dash="dash", width=1.5))
        fig.update_layout(**LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Monthly Seasonal Pattern")
        monthly = load_monthly()
        monthly["month"] = monthly["ds"].dt.month
        monthly["month_name"] = monthly["ds"].dt.strftime("%b")
        by_month = monthly.groupby("month")["y"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        by_month["month_name"] = month_names
        fig = px.bar(by_month, x="month_name", y="y",
                     color="y", color_continuous_scale=[[0,"#1e3a5f"],[1,COLORS["blue"]]],
                     category_orders={"month_name": month_names})
        fig.update_layout(**LAYOUT, height=280, coloraxis_showscale=False, showlegend=False)
        fig.update_yaxes(tickformat="$,.0f", title="Avg Monthly Sales")
        st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly table ──
    st.markdown("#### Detected Anomaly Details")
    display = flagged[["ds","sales","zscore"]].copy()
    display.columns = ["Date","Monthly Sales ($)","Z-Score"]
    display["Date"]             = display["Date"].dt.strftime("%B %Y")
    display["Monthly Sales ($)"]= display["Monthly Sales ($)"].apply(lambda v: f"${v:,.2f}")
    display["Z-Score"]          = display["Z-Score"].apply(lambda v: f"+{v:.2f}" if v > 0 else f"{v:.2f}")
    display["Direction"]        = "📈 Positive spike"
    display["Interpretation"]   = [
        "Unusual mid-year peak — spike above seasonal trend",
        "Strongest month on record — Q4 far above expected"
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND SEGMENTS
# ════════════════════════════════════════════════════════════════════
elif page.startswith("4"):
    st.title("🧩 Product Demand Segments")
    st.caption("K-Means clustering · k=4 · silhouette=0.46 · features: total sales, avg monthly, volatility (CV), trend slope")

    clusters = load_clusters()
    pattern  = load_subcat_pattern()

    # ── KPI cards ──
    cluster_counts = clusters["cluster_name"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, name, css, emoji in [
        (c1, "Core High-Demand Growth",  "green",  "🚀"),
        (c2, "Stable Low-Demand",        "",       "📦"),
        (c3, "Niche High-Growth",        "amber",  "🔍"),
        (c4, "Volatile Low-Growth",      "purple", "⚠️"),
    ]:
        cnt = int(cluster_counts.get(name, 0))
        with col:
            st.markdown(f"""<div class="metric-card {css}">
                <div class="metric-lbl">{emoji} {name}</div>
                <div class="metric-val">{cnt}</div>
                <div class="metric-sub">sub-categories</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter + Donut ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cluster Map — Avg Sales vs Volatility")
        cmap = {
            "Core High-Demand Growth":  COLORS["green"],
            "Stable Low-Demand":        "#8b949e",
            "Niche High-Growth":        COLORS["amber"],
            "Volatile Low-Growth":      COLORS["red"],
        }
        fig = px.scatter(
            clusters, x="avg_monthly", y="volatility_cv",
            size="total_sales", color="cluster_name",
            color_discrete_map=cmap,
            hover_name="Sub-Category",
            size_max=50,
        )
        fig.update_layout(**LAYOUT, height=340, legend=dict(
            bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1,
            x=0.01, y=0.99, font=dict(size=11),
        ))
        fig.update_xaxes(tickformat="$,.0f", title="Avg Monthly Sales ($)")
        fig.update_yaxes(title="Volatility (CV)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Revenue by Cluster")
        cl_totals = clusters.groupby("cluster_name")["total_sales"].sum().reset_index()
        fig = px.pie(
            cl_totals, names="cluster_name", values="total_sales",
            color="cluster_name", color_discrete_map=cmap, hole=0.55,
        )
        fig.update_layout(**LAYOUT, height=340)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    # ── Monthly demand by cluster selector ──
    st.markdown("#### Monthly Demand Pattern by Cluster")
    sel_cluster = st.selectbox(
        "Filter by cluster",
        ["All"] + sorted(clusters["cluster_name"].unique()),
    )
    pattern_long = pattern.melt(id_vars="Sub-Category", var_name="Month", value_name="Sales")
    pattern_long["Month"] = pd.to_datetime(pattern_long["Month"])
    merged = pattern_long.merge(clusters[["Sub-Category","cluster_name"]], on="Sub-Category")
    plot_df = merged if sel_cluster == "All" else merged[merged["cluster_name"] == sel_cluster]

    fig = px.line(
        plot_df, x="Month", y="Sales", color="Sub-Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(**LAYOUT, height=300)
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Full cluster table ──
    st.markdown("#### Sub-Category Cluster Mapping")

    action_map = {
        "Core High-Demand Growth":  "🚀 Invest & scale",
        "Stable Low-Demand":        "📦 Lean stocking",
        "Niche High-Growth":        "🔍 Monitor & grow",
        "Volatile Low-Growth":      "⚠️ Watch closely",
    }
    display = clusters.sort_values(
        ["cluster_name","total_sales"], ascending=[True, False]
    ).copy()
    display["Strategy"]     = display["cluster_name"].map(action_map)
    display["total_sales"]  = display["total_sales"].apply(lambda v: f"${v:,.0f}")
    display["avg_monthly"]  = display["avg_monthly"].apply(lambda v: f"${v:,.0f}")
    display["volatility_cv"]= display["volatility_cv"].round(3)
    display["trend_slope"]  = display["trend_slope"].apply(
        lambda v: f"+{v:.1f}" if v > 0 else f"{v:.1f}"
    )
    display = display.rename(columns={
        "Sub-Category":"Sub-Category",
        "cluster_name":"Cluster",
        "total_sales":"Total Sales",
        "avg_monthly":"Avg Monthly",
        "volatility_cv":"Volatility (CV)",
        "trend_slope":"Trend Slope",
    })
    st.dataframe(
        display[["Sub-Category","Cluster","Total Sales","Avg Monthly","Volatility (CV)","Trend Slope","Strategy"]],
        use_container_width=True, hide_index=True,
    )

    with st.expander("ℹ️ Cluster Interpretation"):
        st.markdown("""
| Cluster | Sub-Categories | Strategy |
|---|---|---|
| 🚀 **Core High-Demand Growth** | Chairs, Phones, Storage, Tables, Binders, Accessories | Highest revenue + growth. Prioritize inventory investment. |
| 📦 **Stable Low-Demand** | Appliances, Bookcases, Paper, Art, Furnishings, Labels, Envelopes, Fasteners | Steady, low-volatility. Good for lean / just-in-time stocking. |
| 🔍 **Niche High-Growth** | Copiers | High per-order value + strong growth slope. Monitor demand closely. |
| ⚠️ **Volatile Low-Growth** | Machines, Supplies | High volatility + flat/declining trend. Reduce overstock risk. |
        """)
