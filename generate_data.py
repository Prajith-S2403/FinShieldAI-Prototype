import numpy as np
import pandas as pd

np.random.seed(42)
n_total = 500
n_mule = 25  # ~5%
n_normal = n_total - n_mule

def gen_mule_accounts(n):
    return pd.DataFrame({
        "account_id":           [f"ACC{str(i).zfill(5)}" for i in range(1, n + 1)],
        "sweep_ratio":          np.round(np.random.uniform(0.75, 1.0,  n), 4),
        "drawdown_score":       np.round(np.random.uniform(0.70, 1.0,  n), 4),
        "credit_debit_ratio":   np.round(np.random.uniform(0.80, 1.0,  n), 4),
        "community_risk_score": np.round(np.random.uniform(0.65, 1.0,  n), 4),
        "transaction_velocity": np.round(np.random.uniform(0.70, 1.0,  n), 4),
        "account_age_days":     np.random.randint(10, 180, n),
        "govt_alert_flag":      np.random.choice([0, 1], size=n, p=[0.25, 0.75]),
        "label":                np.ones(n, dtype=int),
    })

def gen_normal_accounts(n, start_id):
    return pd.DataFrame({
        "account_id":           [f"ACC{str(i).zfill(5)}" for i in range(start_id, start_id + n)],
        "sweep_ratio":          np.round(np.random.uniform(0.00, 0.45, n), 4),
        "drawdown_score":       np.round(np.random.uniform(0.00, 0.40, n), 4),
        "credit_debit_ratio":   np.round(np.random.uniform(0.10, 0.50, n), 4),
        "community_risk_score": np.round(np.random.uniform(0.00, 0.35, n), 4),
        "transaction_velocity": np.round(np.random.uniform(0.00, 0.40, n), 4),
        "account_age_days":     np.random.randint(180, 3650, n),
        "govt_alert_flag":      np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
        "label":                np.zeros(n, dtype=int),
    })

mule_df   = gen_mule_accounts(n_mule)
normal_df = gen_normal_accounts(n_normal, n_mule + 1)

df = pd.concat([mule_df, normal_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("demo_accounts.csv", index=False)
print(f"Generated demo_accounts.csv — {len(df)} accounts, {df['label'].sum()} mule accounts.")
