# SaaS Churn Prediction 

An end-to-end data analytics + machine learning project simulating a SaaS business.  
It predicts which customers are most likely to churn next quarter, explains the drivers, and quantifies the revenue at risk.  
This project is designed to demonstrate **finance + analytics skills** and is built to be portfolio-ready.

---

## 📊 Project Overview
- **Business Context**: In SaaS investing and operations, predicting churn is critical for revenue forecasting and retention strategies.  
- **Goal**: Build a machine learning pipeline that:
  - Generates synthetic SaaS customer data (customers, subscriptions, invoices, logins, tickets)
  - Engineers features like tenure, usage, late payments, support tickets
  - Trains classification models (Logistic Regression, XGBoost)
  - Evaluates performance with ROC-AUC, PR-AUC, calibration, and expected business value
  - Exposes results in a **Streamlit dashboard** for easy interpretation

---

## 🛠️ Tech Stack
- **Data & Modeling**: Python, pandas, scikit-learn, XGBoost, SHAP
- **App**: Streamlit (interactive dashboard)
- **Testing & CI**: pytest, GitHub Actions
- **Infra**: Docker, Makefile for reproducibility

---

## 📂 Repo Layout

saas_churn_project/
├─ app/ # Streamlit dashboard
├─ data/ # raw/processed (gitignored)
├─ mlops/ # pipeline configs
├─ src/ # feature engineering + models
├─ tests/ # unit tests
├─ notebooks/ # exploratory analysis
├─ Dockerfile
├─ Makefile
├─ requirements.txt
└─ model_card.md # assumptions & limitations



---

## 🚀 Quickstart
```bash
# clone this repo
git clone https://github.com/tom-kroot/saas-churn.git
cd saas-churn

# create virtual env
python3 -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# generate data, train models, evaluate
make quickstart

# launch dashboard
make app

