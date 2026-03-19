import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# ── 1. CHURN ──────────────────────────────────────────────────────────────────
tenure        = np.random.randint(1, 72, N)
monthly_charge= np.round(np.random.uniform(20, 120, N), 2)
support_calls = np.random.poisson(1.5, N)
num_products  = np.random.randint(1, 5, N)
has_contract  = np.random.choice([0, 1], N, p=[0.4, 0.6])
late_payments = np.random.poisson(0.8, N)

churn_score = (
    -0.04 * tenure
    + 0.015 * monthly_charge
    + 0.30  * support_calls
    - 0.25  * num_products
    - 0.70  * has_contract
    + 0.40  * late_payments
    + np.random.normal(0, 0.5, N)
)
churn = (churn_score > 0.2).astype(int)

pd.DataFrame({
    "tenure": tenure,
    "monthly_charge": monthly_charge,
    "support_calls": support_calls,
    "num_products": num_products,
    "has_contract": has_contract,
    "late_payments": late_payments,
    "churn": churn,
}).to_csv("/home/claude/ml-course/datasets/churn.csv", index=False)

print(f"churn.csv — {N} linhas | churn rate: {churn.mean():.1%}")


# ── 2. FRAUDE ─────────────────────────────────────────────────────────────────
amount         = np.round(np.random.exponential(150, N), 2)
hour           = np.random.randint(0, 24, N)
distance_home  = np.round(np.random.exponential(20, N), 1)
is_new_merchant= np.random.choice([0, 1], N, p=[0.75, 0.25])
repeat_tries   = np.random.poisson(0.3, N)
country_risk   = np.random.choice([0, 1, 2], N, p=[0.7, 0.2, 0.1])

fraud_score = (
    0.003 * amount
    + 0.8  * (hour < 5).astype(int)
    + 0.04 * distance_home
    + 0.9  * is_new_merchant
    + 1.2  * repeat_tries
    + 0.6  * country_risk
    + np.random.normal(0, 1.2, N)
)
fraud = (fraud_score > 4.5).astype(int)

pd.DataFrame({
    "amount": amount,
    "hour": hour,
    "distance_home_km": distance_home,
    "is_new_merchant": is_new_merchant,
    "repeat_tries": repeat_tries,
    "country_risk": country_risk,
    "fraud": fraud,
}).to_csv("/home/claude/ml-course/datasets/fraud.csv", index=False)

print(f"fraud.csv  — {N} linhas | fraud rate: {fraud.mean():.1%}")


# ── 3. LOGÍSTICA ──────────────────────────────────────────────────────────────
distance_km    = np.random.randint(10, 2000, N)
weight_kg      = np.round(np.random.exponential(5, N) + 0.5, 2)
carrier        = np.random.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2])
weather_issue  = np.random.choice([0, 1], N, p=[0.85, 0.15])
peak_season    = np.random.choice([0, 1], N, p=[0.7, 0.3])
warehouse_load = np.random.uniform(0.3, 1.0, N)
carrier_score  = {"A": 0, "B": 0.4, "C": 0.9}

delay_score = (
    0.0005 * distance_km
    + 0.02  * weight_kg
    + np.array([carrier_score[c] for c in carrier])
    + 1.2   * weather_issue
    + 0.6   * peak_season
    + 0.8   * (warehouse_load > 0.85).astype(int)
    + np.random.normal(0, 0.5, N)
)
delayed = (delay_score > 1.0).astype(int)

pd.DataFrame({
    "distance_km": distance_km,
    "weight_kg": weight_kg,
    "carrier": carrier,
    "weather_issue": weather_issue,
    "peak_season": peak_season,
    "warehouse_load": np.round(warehouse_load, 2),
    "delayed": delayed,
}).to_csv("/home/claude/ml-course/datasets/logistics.csv", index=False)

print(f"logistics.csv — {N} linhas | delay rate: {delayed.mean():.1%}")


# ── 4. QUALIDADE ──────────────────────────────────────────────────────────────
fixed_acidity      = np.round(np.random.normal(8.3, 1.7, N).clip(4, 16), 1)
volatile_acidity   = np.round(np.random.normal(0.53, 0.18, N).clip(0.1, 1.5), 2)
citric_acid        = np.round(np.random.uniform(0, 0.8, N), 2)
residual_sugar     = np.round(np.random.exponential(2.5, N) + 1.0, 1)
chlorides          = np.round(np.random.normal(0.087, 0.047, N).clip(0.01, 0.6), 3)
free_sulfur_dioxide= np.round(np.random.normal(15.9, 10.5, N).clip(1, 72), 1)
density            = np.round(np.random.normal(0.9967, 0.002, N), 4)
pH                 = np.round(np.random.normal(3.31, 0.15, N).clip(2.8, 4.0), 2)
sulphates          = np.round(np.random.normal(0.66, 0.17, N).clip(0.3, 2.0), 2)
alcohol            = np.round(np.random.normal(10.4, 1.1, N).clip(8, 15), 1)

quality_score = (
    -0.8 * volatile_acidity
    + 0.5 * citric_acid
    + 0.4 * sulphates
    + 0.3 * (alcohol - 10)
    - 0.3 * chlorides * 10
    + np.random.normal(0, 0.8, N)
)
approved = (quality_score >= 0.0).astype(int)

pd.DataFrame({
    "fixed_acidity": fixed_acidity,
    "volatile_acidity": volatile_acidity,
    "citric_acid": citric_acid,
    "residual_sugar": residual_sugar,
    "chlorides": chlorides,
    "free_sulfur_dioxide": free_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
    "approved": approved,
}).to_csv("/home/claude/ml-course/datasets/quality.csv", index=False)

print(f"quality.csv — {N} linhas | approval rate: {approved.mean():.1%}")
