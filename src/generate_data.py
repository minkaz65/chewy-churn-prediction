"""
Chewy Customer Churn — Synthetic Dataset Generator
==================================================
No public Chewy customer-level dataset exists (customer data is proprietary),
so this script generates a realistic synthetic dataset modeled on Chewy's
publicly known business characteristics:
  - Autoship (subscription) drives ~83% of net sales (FY2025 public filings)
  - ~20M active customers, pet food/supplies/pharmacy categories
  - Churn = customer cancels Autoship / stops ordering within the next 90 days

The generator builds correlated, causally-plausible features so that models
have real signal to learn (autoship users churn less, unresolved support
tickets increase churn, long tenure decreases churn, etc.).

Usage:
    python generate_data.py [--n 50000] [--out ../data/chewy_customers.csv]
"""

import argparse
import numpy as np
import pandas as pd

RNG_SEED = 42

PET_TYPES = ["Dog", "Cat", "Dog+Cat", "Bird", "Fish", "Reptile", "Small Pet", "Horse"]
PET_TYPE_P = [0.46, 0.28, 0.12, 0.035, 0.035, 0.02, 0.04, 0.01]

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
REGION_P = [0.20, 0.25, 0.22, 0.13, 0.20]

PLANS = ["No Autoship", "Autoship-Basic", "Autoship-Premium"]

PRIMARY_CATEGORIES = ["Dog Food", "Cat Food", "Treats", "Toys", "Health & Pharmacy",
                      "Litter & Supplies", "Flea & Tick", "Other"]
CATEGORY_P = [0.30, 0.19, 0.12, 0.09, 0.12, 0.08, 0.06, 0.04]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=50000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    customer_id = np.array([f"CHWY-{i:07d}" for i in range(1, n + 1)])

    # ---- Demographics / account basics -------------------------------------
    pet_type = rng.choice(PET_TYPES, size=n, p=PET_TYPE_P)
    num_pets = np.clip(rng.poisson(1.6, n) + 1, 1, 8)
    region = rng.choice(REGIONS, size=n, p=REGION_P)
    tenure_months = np.clip(rng.gamma(2.0, 14.0, n), 1, 120).round(0)

    # Autoship enrollment correlates with tenure (long-time customers subscribe)
    autoship_logit = -0.9 + 0.025 * tenure_months + rng.normal(0, 1.0, n)
    is_autoship = (sigmoid(autoship_logit) > rng.uniform(0, 1, n)).astype(int)
    plan = np.where(
        is_autoship == 0, PLANS[0],
        np.where(rng.uniform(0, 1, n) < 0.65, PLANS[1], PLANS[2])
    )

    # ---- Purchase behavior --------------------------------------------------
    base_orders = rng.gamma(2.2, 1.1, n) * (1 + 0.9 * is_autoship)
    orders_per_quarter = np.clip(base_orders, 0.3, 18).round(1)

    aov_base = rng.normal(62, 22, n) + 14 * (plan == "Autoship-Premium") + 6 * num_pets
    avg_order_value = np.clip(aov_base, 12, 400).round(2)

    total_spend_12m = np.clip(
        orders_per_quarter * 4 * avg_order_value * rng.normal(1.0, 0.12, n),
        20, 25000
    ).round(2)

    days_since_last_order = np.clip(
        rng.exponential(22, n) * (1.0 - 0.55 * is_autoship) + 1, 1, 365
    ).round(0)

    pct_pharmacy = np.clip(rng.beta(1.4, 6.0, n) + 0.08 * (num_pets > 2), 0, 1).round(3)
    primary_category = rng.choice(PRIMARY_CATEGORIES, size=n, p=CATEGORY_P)

    # ---- Engagement ---------------------------------------------------------
    app_sessions_per_month = np.clip(
        rng.gamma(1.8, 3.0, n) * (1 + 0.4 * is_autoship), 0, 60
    ).round(1)
    email_open_rate = np.clip(rng.beta(2.0, 3.0, n), 0, 1).round(3)
    used_promo_last_90d = (rng.uniform(0, 1, n) < 0.38).astype(int)

    # ---- Service experience -------------------------------------------------
    support_tickets_6m = rng.poisson(0.7, n)
    unresolved_tickets = np.minimum(rng.binomial(support_tickets_6m, 0.25), 5)
    refunds_12m = rng.poisson(0.35, n)
    avg_delivery_days = np.clip(rng.normal(2.6, 1.1, n), 1, 10).round(1)
    price_increase_flag = (rng.uniform(0, 1, n) < 0.30).astype(int)  # saw a price hike

    # ---- Churn label (causally generated) -----------------------------------
    z = (
        -1.55
        - 1.10 * is_autoship
        - 0.016 * tenure_months
        - 0.10 * orders_per_quarter
        + 0.0125 * days_since_last_order
        + 0.55 * unresolved_tickets
        + 0.28 * refunds_12m
        + 0.16 * avg_delivery_days
        + 0.55 * price_increase_flag
        - 0.85 * email_open_rate
        - 0.022 * app_sessions_per_month
        - 0.60 * pct_pharmacy            # pharmacy users are sticky
        - 0.04 * num_pets
        + rng.normal(0, 0.55, n)          # irreducible noise
    )
    churn_prob = sigmoid(z)
    churned = (churn_prob > rng.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "pet_type": pet_type,
        "num_pets": num_pets,
        "region": region,
        "tenure_months": tenure_months.astype(int),
        "plan": plan,
        "is_autoship": is_autoship,
        "orders_per_quarter": orders_per_quarter,
        "avg_order_value": avg_order_value,
        "total_spend_12m": total_spend_12m,
        "days_since_last_order": days_since_last_order.astype(int),
        "pct_pharmacy_spend": pct_pharmacy,
        "primary_category": primary_category,
        "app_sessions_per_month": app_sessions_per_month,
        "email_open_rate": email_open_rate,
        "used_promo_last_90d": used_promo_last_90d,
        "support_tickets_6m": support_tickets_6m,
        "unresolved_tickets": unresolved_tickets,
        "refunds_12m": refunds_12m,
        "avg_delivery_days": avg_delivery_days,
        "price_increase_flag": price_increase_flag,
        "churned": churned,
    })

    # Inject a little real-world mess: ~1.5% missing values in 3 columns
    for col in ["email_open_rate", "avg_delivery_days", "app_sessions_per_month"]:
        mask = rng.uniform(0, 1, n) < 0.015
        df.loc[mask, col] = np.nan

    # ~0.3% duplicate rows (to be cleaned in the PySpark stage)
    dupes = df.sample(frac=0.003, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out", type=str, default="../data/chewy_customers.csv")
    args = ap.parse_args()

    df = generate(args.n)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows -> {args.out}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(df.head())
