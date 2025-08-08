# SaaS Churn Prediction (Portfolio-Ready Project)

An end-to-end ML project that simulates a SaaS business and predicts next-quarter churn at the account level, quantifies revenue at risk, and exposes results via a Streamlit app.

## Tech Stack
- Python, pandas, scikit-learn, XGBoost, SHAP
- Streamlit (dashboard)
- pytest (tests), GitHub Actions (CI)

## Quickstart
```bash
# 1) create & activate a venv (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) generate synthetic data, build features, train, and evaluate
make quickstart

# 4) launch the app
make app
```

## Project Layout
```
project/
├─ app/                         # Streamlit app
├─ data/                        # raw/processed (gitignored; tiny demo included)
├─ mlops/                       # pipeline configs
├─ notebooks/                   # EDA + modeling
├─ src/
│  ├─ features/                 # feature builders
│  ├─ models/                   # training/inference
│  └─ utils/                    # helpers
├─ tests/                       # unit tests
├─ Dockerfile
├─ Makefile
├─ requirements.txt
├─ model_card.md
└─ .github/workflows/ci.yml
```

## Business Framing
We optimize for **precision-recall** under class imbalance and report **expected value** of retention offers. The cost matrix and revenue-at-risk assumptions are configurable in `mlops/config.yaml`.

## Talking Points
- Time-based train/test split to avoid target leakage.
- PR-AUC and calibration for decisioning.
- SHAP for feature importance; verified via ablations.
- Intervention policy: target top-k% risk until budget is exhausted.

## License
MIT