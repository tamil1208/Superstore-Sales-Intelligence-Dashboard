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

````markdown
# 📊 Superstore Sales Intelligence Dashboard

An end-to-end **Sales Forecasting and Demand Intelligence System** built using **Python, Machine Learning, and Streamlit**. This project analyzes historical Superstore sales data, forecasts future demand, detects anomalies, segments products based on demand, and provides an interactive dashboard to support data-driven business decisions.

---

## 🌐 Live Dashboard

🚀 Explore the interactive dashboard here:

**https://superstore-analytics-app.netlify.app/**

---

## 📓 Google Colab

Run the complete project directly in **Google Colab** without installing any software locally.

### 🚀 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s-TdLNkgoUUQ0b6WKYarN_jcgth7auOk#scrollTo=gxeVFkBh3ljQ)

Or access it directly:

**https://colab.research.google.com/drive/1s-TdLNkgoUUQ0b6WKYarN_jcgth7auOk#scrollTo=gxeVFkBh3ljQ**

### 📚 Notebook Includes

- 📥 Data Loading & Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- 📈 Time Series Analysis
- 🔮 Sales Forecasting (SARIMA, Prophet & XGBoost)
- 🚨 Anomaly Detection using Isolation Forest
- 🎯 Product Demand Segmentation using K-Means Clustering
- 📉 Model Performance Evaluation
- 📋 Business Insights & Recommendations

---

## 📌 Features

- 📈 Interactive Sales Performance Dashboard
- 🔮 3-Month Sales Forecasting
- 📊 Sales Trend Analysis
- 🌍 Regional & Category-wise Analysis
- 🚨 Anomaly Detection using Isolation Forest
- 🎯 Product Demand Segmentation using K-Means Clustering
- 📅 Time Series Decomposition
- 📦 Executive Business Insights
- 📉 Forecast Comparison Across Models
- 📊 Interactive Charts with Plotly

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost
- Prophet
- Statsmodels

### Visualization
- Plotly
- Matplotlib

### Deployment
- Streamlit
- Netlify

---

## 📂 Project Structure

```text
Superstore-Sales-Intelligence-Dashboard/
│
├── app.py
├── requirements.txt
├── train.csv
├── notebooks/
├── models/
├── images/
├── reports/
└── README.md
```

---

## 📊 Dashboard Overview

The dashboard provides:

- 📈 Sales KPI Overview
- 📅 Monthly & Yearly Sales Trends
- 🛍️ Category-wise Performance
- 🌍 Regional Sales Analysis
- 🔮 Future Sales Forecast
- 🎯 Product Demand Segmentation
- 🚨 Anomaly Detection Report
- 📊 Interactive Business Visualizations

---

## 🤖 Machine Learning Models

| Model | Purpose |
|-------|---------|
| SARIMA | Time Series Forecasting |
| Prophet | Seasonal Sales Forecasting |
| XGBoost | Machine Learning Forecasting |
| Isolation Forest | Anomaly Detection |
| K-Means | Product Demand Segmentation |

---

## 📈 Business Insights

This project helps businesses:

- Forecast future sales demand
- Optimize inventory planning
- Identify high-demand products
- Detect unusual sales behavior
- Improve supply chain decisions
- Support executive decision-making using data

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/tamil1208/Superstore-Sales-Intelligence-Dashboard.git
```

### Navigate to the Project

```bash
cd Superstore-Sales-Intelligence-Dashboard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Project Screenshots

Add screenshots of:

- Dashboard Home
- Sales Forecast
- Regional Analysis
- Demand Segmentation
- Anomaly Detection

inside the `images/` folder and reference them here.

Example:

```markdown
![Dashboard](images/dashboard.png)
```

---

## 📊 Dataset

- **Dataset:** Superstore Sales Dataset
- Includes:
  - Orders
  - Sales
  - Profit
  - Discount
  - Region
  - Category
  - Sub-Category
  - Shipping Details

---

## 🎯 Future Improvements

- Real-time sales data integration
- Deep Learning forecasting models (LSTM)
- Customer Segmentation
- Inventory Optimization
- Power BI Dashboard
- Automated Email Reports

---

## 👨‍💻 Author

**Tamilarasan P**

- GitHub: https://github.com/tamil1208
- LinkedIn: *(Add your LinkedIn profile URL)*
- Email: *(Add your email address)*

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. It helps others discover the project and motivates future improvements.

**Thank you for visiting this repository! 🚀**
````

## Dataset

Superstore Sales dataset — retail transactions from 2015–2018 across US regions, product categories, and sub-categories, including order dates, ship dates, customer segments, and sales values.
