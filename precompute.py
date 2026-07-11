"""
Precompute script: runs forecasting, anomaly detection, and clustering
on the Superstore sales data, and saves all outputs to /data as
parquet/csv/json so the Streamlit app just loads them (fast + no
retraining on every page load / reboot on Streamlit Cloud).
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW = "data/train.csv"

# ---------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------
df = pd.read_csv(RAW)
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
df = df.dropna(subset=["Order Date", "Sales"])
df["Sales"] = df["Sales"].astype(float)
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

df.to_csv("data/clean_sales.csv", index=False)

# ---------------------------------------------------------------
# 1. Sales overview aggregates
# ---------------------------------------------------------------
yearly = df.groupby("Year", as_index=False)["Sales"].sum()
yearly.to_csv("data/yearly_sales.csv", index=False)

monthly = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
monthly.to_csv("data/monthly_sales.csv", index=False)

region_cat = df.groupby(["Region", "Category", "Month"], as_index=False)["Sales"].sum()
region_cat.to_csv("data/region_category_monthly.csv", index=False)

# ---------------------------------------------------------------
# 2. Forecasting — build monthly series per Category and per Region,
#    fit a RandomForest on lag/calendar features, forecast forward.
# ---------------------------------------------------------------
def build_features(series_df):
    """series_df: columns Month, Sales sorted by Month. Adds lag & calendar feats."""
    s = series_df.copy().reset_index(drop=True)
    s["month_num"] = s["Month"].dt.month
    s["year"] = s["Month"].dt.year
    s["t"] = np.arange(len(s))
    for lag in [1, 2, 3, 6, 12]:
        s[f"lag_{lag}"] = s["Sales"].shift(lag)
    s["rolling_mean_3"] = s["Sales"].shift(1).rolling(3).mean()
    return s

def train_forecast_model(series_df, horizon_months=3):
    """Train RF on available history, evaluate on last 20% holdout,
    then forecast `horizon_months` ahead recursively."""
    feat = build_features(series_df).dropna().reset_index(drop=True)
    feature_cols = ["month_num", "year", "t", "lag_1", "lag_2", "lag_3",
                     "lag_6", "lag_12", "rolling_mean_3"]

    if len(feat) < 10:
        # not enough history for lag_12 model; fall back to simpler lag set
        feat = build_features(series_df)
        feature_cols = ["month_num", "year", "t", "lag_1", "lag_2", "rolling_mean_3"]
        feat = feat.dropna(subset=feature_cols + ["Sales"]).reset_index(drop=True)

    split = max(int(len(feat) * 0.8), len(feat) - 6)
    train, test = feat.iloc[:split], feat.iloc[split:]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(train[feature_cols], train["Sales"])

    if len(test) > 0:
        preds = model.predict(test[feature_cols])
        mae = mean_absolute_error(test["Sales"], preds)
        rmse = mean_squared_error(test["Sales"], preds) ** 0.5
    else:
        mae, rmse = np.nan, np.nan

    # refit on full data for the actual forecast
    model_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model_full.fit(feat[feature_cols], feat["Sales"])

    # recursive forecast
    history = series_df.copy().reset_index(drop=True)
    last_month = history["Month"].max()
    forecasts = []
    for i in range(1, horizon_months + 1):
        next_month = (last_month + pd.DateOffset(months=i))
        tmp = pd.concat([history, pd.DataFrame({"Month": [next_month], "Sales": [np.nan]})],
                         ignore_index=True)
        tmp_feat = build_features(tmp)
        row = tmp_feat.iloc[[-1]]
        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[feature_cols].ffill(axis=0).fillna(0)
        pred = model_full.predict(row)[0]
        forecasts.append({"Month": next_month, "Forecast": float(pred)})
        history = pd.concat([history, pd.DataFrame({"Month": [next_month], "Sales": [pred]})],
                             ignore_index=True)

    return forecasts, mae, rmse

forecast_results = {"Category": {}, "Region": {}}
metrics_results = {"Category": {}, "Region": {}}

for dim in ["Category", "Region"]:
    for val in df[dim].unique():
        sub = df[df[dim] == val].groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
        if len(sub) < 6:
            continue
        fc, mae, rmse = train_forecast_model(sub, horizon_months=3)
        forecast_results[dim][val] = {
            "history": sub.assign(Month=sub["Month"].dt.strftime("%Y-%m-%d")).to_dict("records"),
            "forecast": [{"Month": f["Month"].strftime("%Y-%m-%d"), "Forecast": f["Forecast"]} for f in fc],
        }
        metrics_results[dim][val] = {"MAE": None if np.isnan(mae) else round(mae, 2),
                                      "RMSE": None if np.isnan(rmse) else round(rmse, 2)}

with open("data/forecasts.json", "w") as f:
    json.dump(forecast_results, f)
with open("data/forecast_metrics.json", "w") as f:
    json.dump(metrics_results, f)

# ---------------------------------------------------------------
# 3. Anomaly detection on overall monthly sales
#    (IQR-based, robust and explainable)
# ---------------------------------------------------------------
m = monthly.copy().sort_values("Month").reset_index(drop=True)
m["pct_change"] = m["Sales"].pct_change()
Q1, Q3 = m["Sales"].quantile(0.25), m["Sales"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
m["is_anomaly"] = (m["Sales"] < lower) | (m["Sales"] > upper)

# also flag anomalies via z-score on residuals from rolling mean (captures spikes/drops relative to trend)
m["rolling_mean"] = m["Sales"].rolling(3, center=True, min_periods=1).mean()
m["residual"] = m["Sales"] - m["rolling_mean"]
resid_std = m["residual"].std()
m["z_resid"] = m["residual"] / resid_std
m["is_anomaly"] = m["is_anomaly"] | (m["z_resid"].abs() > 2)

m.to_csv("data/anomaly_monthly.csv", index=False)

anomalies = m[m["is_anomaly"]][["Month", "Sales"]].copy()
anomalies["Month"] = anomalies["Month"].dt.strftime("%Y-%m")
anomalies.to_csv("data/anomalies_table.csv", index=False)

# ---------------------------------------------------------------
# 4. Product demand segmentation (clustering sub-categories)
#    Features: total sales, avg monthly sales, sales volatility (CV),
#    order frequency, avg order value
# ---------------------------------------------------------------
sub_monthly = df.groupby(["Sub-Category", "Month"], as_index=False)["Sales"].sum()

agg = df.groupby("Sub-Category").agg(
    total_sales=("Sales", "sum"),
    avg_order_value=("Sales", "mean"),
    order_count=("Sales", "count"),
).reset_index()

monthly_stats = sub_monthly.groupby("Sub-Category")["Sales"].agg(["mean", "std"]).reset_index()
monthly_stats.columns = ["Sub-Category", "avg_monthly_sales", "monthly_std"]
monthly_stats["volatility_cv"] = monthly_stats["monthly_std"] / monthly_stats["avg_monthly_sales"]

seg = agg.merge(monthly_stats, on="Sub-Category")
seg["volatility_cv"] = seg["volatility_cv"].fillna(0)

feature_cols = ["total_sales", "avg_order_value", "order_count", "avg_monthly_sales", "volatility_cv"]
X = seg[feature_cols].fillna(0)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
seg["cluster"] = km.fit_predict(Xs)

# label clusters by demand character (high volume / high volatility / steady / low demand)
cluster_summary = seg.groupby("cluster")[["total_sales", "avg_monthly_sales", "volatility_cv"]].mean()
cluster_summary = cluster_summary.sort_values("total_sales", ascending=False)
rank_map = {c: i for i, c in enumerate(cluster_summary.index)}
labels = ["High-Volume Demand", "Steady Demand", "Moderate/Variable Demand", "Low-Volume Demand"]
label_map = {c: labels[min(i, len(labels)-1)] for c, i in rank_map.items()}
seg["cluster_label"] = seg["cluster"].map(label_map)

seg_out = seg[["Sub-Category", "total_sales", "avg_order_value", "order_count",
               "avg_monthly_sales", "volatility_cv", "cluster", "cluster_label"]]
seg_out.to_csv("data/product_clusters.csv", index=False)

# 2D projection for plotting (PCA-lite: just pick 2 informative features, scaled)
seg_out2 = seg_out.copy()
seg_out2["x_total_sales"] = X["total_sales"].values
seg_out2["y_volatility"] = X["volatility_cv"].values
seg_out2.to_csv("data/product_clusters_plot.csv", index=False)

print("Precompute complete.")
print("Yearly:", len(yearly), "rows")
print("Monthly:", len(monthly), "rows")
print("Anomalies found:", m["is_anomaly"].sum())
print("Clusters:", seg["cluster_label"].value_counts().to_dict())
