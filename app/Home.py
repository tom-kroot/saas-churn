import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="SaaS Churn Dashboard", layout="wide")

st.title("📉 SaaS Churn Prediction — Demo")
st.write("Portfolio-ready demo app showing predictions, drivers, and business impact.")

data_dir = Path("data/processed")
pred_path = data_dir / "predictions.csv"
feat_imp_path = data_dir / "feature_importance.csv"
summary_path = data_dir / "summary_metrics.json"

cols = st.columns(3)
if summary_path.exists():
    summary = pd.read_json(summary_path, typ="series")
    cols[0].metric("ROC-AUC", f"{summary.get('roc_auc', np.nan):.3f}")
    cols[1].metric("PR-AUC", f"{summary.get('pr_auc', np.nan):.3f}")
    cols[2].metric("Calib. Brier", f"{summary.get('brier', np.nan):.3f}")
else:
    st.info("Run `make quickstart` to generate data, train a model, and create predictions.")

st.header("Accounts at Risk")
if pred_path.exists():
    df = pd.read_csv(pred_path)
    st.dataframe(df.head(50))
else:
    st.warning("No predictions yet.")

st.header("Top Features")
if feat_imp_path.exists():
    fi = pd.read_csv(feat_imp_path)
    st.bar_chart(fi.set_index('feature')["importance"])
else:
    st.warning("No feature importances yet.")