# 📦 Smart E-Commerce Logistics & Delivery Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning logistics platform and interactive **Streamlit Dashboard** built on the **Brazilian E-Commerce (Olist)** dataset (96,400+ real-world orders). The system predicts delivery duration, flags high-risk late orders before SLA breach, and provides geographic logistics intelligence across all 27 Brazilian states.

---

## 🌟 Key Features

### 1. 🏠 Executive Overview Dashboard
- High-level KPIs: Total Orders (96.4k+), Avg Delivery Time (12.1 days), Late Delivery Rate (7.8%), Total Revenue (R$13.2M), Avg Freight Cost.
- Monthly order volume and transit duration trends.
- Delivery duration distribution with median benchmark.
- On-Time vs. Late delivery ratio visualization.

### 2. 🔍 Deep-Dive EDA Explorer
- **Delivery Patterns:** Intra-state vs. inter-state shipping comparisons.
- **Payment & Pricing:** Installment impact, payment method breakdown (Credit Card, Boleto, Voucher), and freight ratio analysis.
- **Seller Analytics:** Seller density, seller state distribution, items per order impact on delivery turnaround.
- **Temporal Trends:** Hourly order arrival patterns and month-by-year delay heatmap.

### 3. 🤖 Machine Learning Performance
- **Delivery Time Regressor:**
  - Gradient Boosting Regressor (MAE: ~4.95 days, RMSE: ~7.32 days, R²: 0.354).
  - Outperforms baseline median estimator by ~1.5 days.
- **Late Delivery Classifier:**
  - Gradient Boosting Classifier (ROC-AUC: 0.83).
  - Decision threshold calibrated to **0.125** (`optimal_risk_threshold.pkl`) to achieve high recall (>60%) amidst severe 8% class imbalance.
- Interactive radar chart and Precision-Recall-F1 threshold sweep curves.

### 4. 🎯 Live Risk Prediction Simulator
- Real-time simulation using trained Scikit-Learn pipelines.
- Supports **Manual Feature Entry** or **Random Dataset Sampling**.
- Outputs:
  - Estimated Delivery Days.
  - Probability of delay (%) with a responsive risk gauge.
  - Actionable status card: **🚨 HIGH RISK** (with escalation recommendation) vs **✅ ON SCHEDULE**.

### 5. 🗺️ Geographic Logistics Intelligence
- State-level latency and delay hotspot rankings across Brazil's 27 federal states.
- **Sankey Flow Diagram:** Interactive visualization of shipments flowing from Seller State ➔ Customer State.
- Full sortable performance table with color gradient metrics.

---

## 🏗️ Repository Architecture

```text
Smart_Ecommerce_Logistics/
├── app/
│   └── app.py                     # Main Streamlit dashboard application
├── data/
│   └── processed/
│       └── master_dataset.csv     # Cleaned & feature-engineered dataset
├── models/
│   ├── final_delivery_time_regressor.pkl
│   ├── final_late_delivery_classifier.pkl
│   └── optimal_risk_threshold.pkl # Calibrated probability threshold (0.125)
├── notebooks/                     # Exploratory, cleaning & ML modeling notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_regression_model.ipynb
│   ├── 06_classification_model.ipynb
│   ├── 07_model_improvement.ipynb
│   └── final_ecommerce_logistics_pipeline.ipynb
├── src/                           # Utility scripts & feature pipelines
├── requirements.txt               # Production dependencies
└── README.md
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/SatyamGit8/smart-ecommerce-logistics.git
cd smart-ecommerce-logistics
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment Guide

### Deploying to Streamlit Community Cloud (1-Click Free)
1. Fork or push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select:
   - **Repository:** `smart-ecommerce-logistics`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Click **"Deploy!"**

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
