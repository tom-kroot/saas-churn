# Model Card — SaaS Churn Predictor

**Intended Use:** Identify accounts likely to churn in the next quarter for proactive retention interventions.

**Data:** Synthetic tables: customers, subscriptions, usage_events, tickets, invoices, logins. Time horizon: 8 quarters simulated.

**Features (examples):**
- Tenure, MRR, ARPA, seat utilization
- Invoice lateness, support ticket volume/severity
- Usage momentum, login frequency
- Contract term and industry

**Training/Eval:**
- Time-based split (Q1–Q6 train, Q7 validate, Q8 test)
- Models: Logistic Regression baseline, XGBoost main
- Metrics: ROC-AUC, PR-AUC, calibration, cost curves

**Fairness/Limitations:**
- Synthetic dataset; not representative of any real population.
- Do not deploy without re-training and validating on real data.
- Explainability via SHAP is local; treat with care.

**Maintenance:**
- Retrain quarterly; monitor drift in feature distributions and calibration.