import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 1200
Q = 8  # quarters

industries = ["FinTech","HealthTech","EdTech","Cyber","Retail","DevTools"]
plans = ["Basic","Pro","Enterprise"]

def simulate():
    customers = pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS+1),
        "industry": rng.choice(industries, N_CUSTOMERS),
        "plan": rng.choice(plans, N_CUSTOMERS, p=[0.5, 0.35, 0.15]),
        "seats": rng.integers(3, 200, N_CUSTOMERS),
        "start_qtr": rng.integers(1, 3, N_CUSTOMERS),  # most are existing
    })

    subs = []
    invoices = []
    logins = []
    tickets = []

    for q in range(1, Q+1):
        # active if started and not yet churned
        if q == 1:
            churned = np.zeros(N_CUSTOMERS, dtype=bool)
        # base churn probability by plan
        base = np.where(customers["plan"]=="Basic", 0.10,
                 np.where(customers["plan"]=="Pro", 0.07, 0.05))
        # price / MRR
        price = np.where(customers["plan"]=="Basic", 20,
                 np.where(customers["plan"]=="Pro", 40, 80))
        mrr = price * customers["seats"]

        # usage & support
        login_freq = rng.normal(12, 4, N_CUSTOMERS).clip(0, None)
        util = rng.beta(2,2, N_CUSTOMERS)  # 0..1
        tickets_cnt = rng.poisson(0.3 + (1-util)*2.0, N_CUSTOMERS)
        late_invoice = rng.binomial(1, 0.08 + (tickets_cnt>2)*0.05, N_CUSTOMERS)

        # churn driver: low util, many tickets, late invoices, low seats
        churn_prob = base + (0.12*(util<0.3)) + (0.07*(tickets_cnt>3)) + (0.05*late_invoice) + (0.05*(customers["seats"]<10))
        churn_prob = churn_prob.clip(0, 0.85)

        # once churned, stay churned
        active = (np.arange(N_CUSTOMERS) >= 0) & (q >= customers["start_qtr"]) & (~churned)
        churn_now = (rng.random(N_CUSTOMERS) < churn_prob) & active

        # record
        subs.append(pd.DataFrame({
            "quarter": q, "customer_id": customers["customer_id"],
            "active": active.astype(int), "mrr": mrr, "util": util,
            "tickets": tickets_cnt, "late_invoice": late_invoice, "login_freq": login_freq
        }))

        invoices.append(pd.DataFrame({
            "quarter": q, "customer_id": customers["customer_id"],
            "amount": mrr, "late": late_invoice
        }))

        logins.append(pd.DataFrame({
            "quarter": q, "customer_id": customers["customer_id"],
            "logins": (login_freq*7).astype(int)
        }))

        tickets.append(pd.DataFrame({
            "quarter": q, "customer_id": customers["customer_id"],
            "tickets": tickets_cnt
        }))

        churned = churned | churn_now

    customers.to_csv(DATA_DIR/"customers.csv", index=False)
    pd.concat(subs).to_csv(DATA_DIR/"subscriptions.csv", index=False)
    pd.concat(invoices).to_csv(DATA_DIR/"invoices.csv", index=False)
    pd.concat(logins).to_csv(DATA_DIR/"logins.csv", index=False)
    pd.concat(tickets).to_csv(DATA_DIR/"tickets.csv", index=False)
    print("Synthetic data generated in data/raw/*.csv")

if __name__ == "__main__":
    simulate()