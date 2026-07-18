

# 📊 Superstore Sales Intelligence Dashboard 

An end-to-end **Sales Forecasting and Demand Intelligence System** built using **Python, Machine Learning, and Streamlit**. This project analyzes historical Superstore sales data, forecasts future demand, detects anomalies, segments products based on demand, and provides an interactive dashboard to support data-driven business decisions.


## 🌐 Live Dashboard

🚀 Explore the interactive dashboard:

**Streamlit App:** https://superstore-analytics-app.streamlit.app/

> **Note:** If the Streamlit application is temporarily unavailable, you can explore the complete project using the Google Colab notebook below.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tamil1208/Superstore-Sales-Intelligence-Dashboard.git
```

### 2. Navigate to the Project

```bash
cd Superstore-Sales-Intelligence-Dashboard
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```
git add .
git commit -m "Fix dataset path"
git push
```
The dashboard will open at:

```
http://localhost:8501
```

---

## 🔧 Troubleshooting

### FileNotFoundError

If the dashboard loads but no graphs appear, verify that your dataset exists.

Example:

```text
Superstore-Sales-Intelligence-Dashboard/
│
├── app.py
├── requirements.txt
├── train.csv
├── clean.csv
└── ...
```

or

```text
data/
└── clean.csv
```

If `clean.csv` does not exist, update the code to load the correct dataset.

Example:

```python
df = pd.read_csv("train.csv", parse_dates=["Order Date"])
```

or

```python
df = pd.read_csv("clean.csv", parse_dates=["Order Date"])
```

Make sure the filename exactly matches the file in the repository.

### After Updating

Commit and push your changes:

```bash
git add .
git commit -m "Fix dataset path"
git push
```

Then redeploy your Streamlit application.

---

## 📸 Project Screenshots

### Dashboard Home

![Dashboard Home](images/dashboard-home.png)

### Sales Forecast

![Sales Forecast](images/sales-forecast.png)

### Regional Analysis

![Regional Analysis](images/regional-analysis.png)

### Product Demand Segmentation

![Demand Segmentation](images/demand-segmentation.png)

### Anomaly Detection

![Anomaly Detection](images/anomaly-detection.png)

---

## 👨‍💻 Author

**Tamilarasan P**

- GitHub: https://github.com/tamil1208
- LinkedIn: https://www.linkedin.com/in/tamilarasan-a2466b274/
- Email: tamilarsan538@gmail.com

**Thank you for visiting this repository! 🚀**
