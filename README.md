# 📊 Superstore Sales Intelligence Dashboard

A multi-page Streamlit web app built on the Superstore Sales dataset (2015–2018), covering sales analytics, forecasting, anomaly detection, and product demand segmentation.

**Live app:** _add your Streamlit Community Cloud link here after deploying_

---

## Pages

### 1. Sales Overview
- Total sales by year (bar chart)
- Monthly sales trend (line chart)
- Sales by region and category with interactive filters (Region, Category, Year)
- Region × Category heatmap
- Filtered raw data table

### 2. Forecast Explorer
- Dropdown to select **Category** or **Region**
- Slider to choose forecast horizon (1, 2, or 3 months ahead)
- Forecast chart (actual vs. predicted) using a Random Forest model
- MAE and RMSE displayed below the chart

### 3. Anomaly Report
- Monthly sales chart with anomalies highlighted
- Table of detected anomaly dates and their sales values
- Detection method: IQR bounds + rolling z-score (|z| > 2) on residuals

### 4. Product Demand Segments
- Cluster scatter plot of sub-categories by demand behavior
- Table mapping each sub-category to its demand cluster
- Cluster summary (count, total sales, avg volatility per cluster)

---

## How it works

All heavy computation (forecasting, anomaly detection, clustering) is done **once**, offline, by `precompute.py`, which reads `data/train.csv` and writes lightweight CSV/JSON artifacts into `data/`. The Streamlit app only reads these precomputed files, so pages load instantly with no retraining on every visit or app reboot.

**Models used:**
- **Forecasting:** `RandomForestRegressor` per Category/Region, trained on monthly aggregated sales with lag features (1, 2, 3, 6, 12 months), 3-month rolling mean, and calendar features (month, year, time index). Evaluated on a held-out tail of the series (MAE, RMSE).
- **Anomaly detection:** IQR method (Q1 − 1.5×IQR to Q3 + 1.5×IQR) combined with a rolling z-score on residuals from a 3-month centered moving average.
- **Clustering:** `KMeans` (k=4) on standardized sub-category features — total sales, avg order value, order count, avg monthly sales, sales volatility (coefficient of variation) — labeled by demand character (High-Volume, Steady, Moderate/Variable, Low-Volume).

---

## Project structure

```
dashboard/
├── Home.py                          # Landing page with KPIs
├── precompute.py                    # Data pipeline: forecasting, anomalies, clustering
├── requirements.txt
├── data/
│   ├── train.csv                    # Raw Superstore dataset
│   ├── clean_sales.csv              # Cleaned/typed transaction data
│   ├── yearly_sales.csv
│   ├── monthly_sales.csv
│   ├── region_category_monthly.csv
│   ├── forecasts.json               # Per Category/Region forecast + history
│   ├── forecast_metrics.json        # MAE/RMSE per Category/Region
│   ├── anomaly_monthly.csv
│   ├── anomalies_table.csv
│   ├── product_clusters.csv
│   └── product_clusters_plot.csv
└── pages/
    ├── 1_Sales_Overview.py
    ├── 2_Forecast_Explorer.py
    ├── 3_Anomaly_Report.py
    └── 4_Product_Demand_Segments.py
```

---

## Running locally

```bash
pip install -r requirements.txt
python precompute.py      # regenerates data/ artifacts from train.csv (optional, already included)
streamlit run Home.py
```

App will be available at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**.
3. Select this repo, branch `main`, main file path `Home.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and builds the app.
5. Copy the generated `*.streamlit.app` URL and add it to the top of this README.

---

## Tech stack

- **Streamlit** — multi-page app framework
- **Plotly** — interactive charts
- **scikit-learn** — RandomForestRegressor, KMeans, StandardScaler
- **pandas / numpy** — data processing

---

## Dataset

Superstore Sales dataset — retail transactions from 2015–2018 across US regions, product categories, and sub-categories, including order dates, ship dates, customer segments, and sales values.
