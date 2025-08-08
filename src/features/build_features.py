import pandas as pd
import numpy as np
from pathlib import Path
import yaml

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def load_config():
    with open("mlops/config.yaml","r") as f:
        return yaml.safe_load(f)

def build():
    customers = pd.read_csv(RAW/"customers.csv")
    subs = pd.read_csv(RAW/"subscriptions.csv")
    invoices = pd.read_csv(RAW/"invoices.csv")
    logins = pd.read_csv(RAW/"logins.csv")
    tickets = pd.read_csv(RAW/"tickets.csv")

    df = subs.merge(customers, on="customer_id", how="left")
    df = df.merge(logins, on=["customer_id","quarter"], how="left")
    df = df.merge(tickets, on=["customer_id","quarter"], how="left", suffixes=("","_tickets"))
    df = df.merge(invoices[["customer_id","quarter","late"]], on=["customer_id","quarter"], how="left", suffixes=("","_inv"))

    df = df.sort_values(["customer_id","quarter"])
    df["util_prev"] = df.groupby("customer_id")["util"].shift(1)
    df["login_freq_prev"] = df.groupby("customer_id")["login_freq"].shift(1)
    df["tickets_prev"] = df.groupby("customer_id")["tickets"].shift(1)
    df["late_prev"] = df.groupby("customer_id")["late"].shift(1).fillna(0)

    df["active_next"] = df.groupby("customer_id")["active"].shift(-1).fillna(0)
    df["churn_next"] = ((df["active"]==1) & (df["active_next"]==0)).astype(int)
    df["tenure_q"] = df.groupby("customer_id").cumcount()

    df = df[df.groupby("customer_id")["quarter"].transform("max") > df["quarter"]]

    # --- Encode BEFORE split so splits carry the columns
    df["plan_Basic"] = (df["plan"]=="Basic").astype(int)
    df["plan_Pro"] = (df["plan"]=="Pro").astype(int)
    df["industry_hash"] = (pd.util.hash_pandas_object(df["industry"]) % 10).astype(int)

    # time-based split
    cfg = load_config()
    train_end = cfg["split"]["train_end_quarter"]
    valid_q = cfg["split"]["valid_quarter"]
    test_q = cfg["split"]["test_quarter"]

    train = df[df["quarter"]<=train_end]
    valid = df[df["quarter"]==valid_q]
    test = df[df["quarter"]==test_q]

    features = ["tenure_q","seats","mrr","util","util_prev","login_freq","login_freq_prev","tickets","tickets_prev","late","late_prev","plan_Basic","plan_Pro","industry_hash"]

    def pack(d):
        X = d[features].fillna(0.0).astype(float)
        y = d["churn_next"].astype(int)
        return X, y, d[["customer_id","quarter"]]

    X_train, y_train, meta_train = pack(train)
    X_valid, y_valid, meta_valid = pack(valid)
    X_test, y_test, meta_test = pack(test)

    X_train.to_csv(PROC/"X_train.csv", index=False)
    y_train.to_csv(PROC/"y_train.csv", index=False, header=True)
    X_valid.to_csv(PROC/"X_valid.csv", index=False)
    y_valid.to_csv(PROC/"y_valid.csv", index=False, header=True)
    X_test.to_csv(PROC/"X_test.csv", index=False)
    y_test.to_csv(PROC/"y_test.csv", index=False, header=True)

    meta_test.to_csv(PROC/"meta_test.csv", index=False)
    print("Features built in data/processed/*.csv")

if __name__ == "__main__":
    build()