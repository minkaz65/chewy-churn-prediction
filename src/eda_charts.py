"""EDA charts for the Chewy churn project — used in the PDF report and slides."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import viz_style as vs

vs.apply()
FIG = "../figures"

df = pd.read_csv("../data/chewy_customers.csv").drop_duplicates(subset="customer_id")
churn_rate = df["churned"].mean()

# ---- 1. Churn rate by Autoship plan ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
order = ["No Autoship", "Autoship-Basic", "Autoship-Premium"]
rates = df.groupby("plan")["churned"].mean().reindex(order) * 100
bars = ax.bar(order, rates, color=vs.BLUE, width=0.55, zorder=3)
for b, v in zip(bars, rates):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%",
            ha="center", fontsize=10.5, fontweight="bold", color=vs.INK)
ax.set_title("Autoship subscribers churn far less", loc="left", pad=14)
ax.set_ylabel("90-day churn rate (%)")
ax.set_ylim(0, rates.max() * 1.2)
vs.despine_y(ax)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/churn_by_plan.png"); plt.close(fig)

# ---- 2. Churn rate by tenure bucket ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
buckets = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 120],
                 labels=["0–6 mo", "7–12 mo", "1–2 yr", "2–4 yr", "4+ yr"])
tr = df.groupby(buckets, observed=True)["churned"].mean() * 100
ax.bar(tr.index.astype(str), tr.values, color=vs.BLUE, width=0.55, zorder=3)
for i, v in enumerate(tr.values):
    ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=vs.INK)
ax.set_title("New customers are the biggest churn risk", loc="left", pad=14)
ax.set_ylabel("90-day churn rate (%)")
ax.set_xlabel("Customer tenure")
vs.despine_y(ax); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/churn_by_tenure.png"); plt.close(fig)

# ---- 3. Recency distribution by churn status ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(0, 181, 10)
for label, sub, color in [("Retained", df[df.churned == 0], vs.BLUE),
                          ("Churned", df[df.churned == 1], vs.ORANGE)]:
    ax.hist(sub["days_since_last_order"].clip(0, 180), bins=bins, density=True,
            histtype="step", linewidth=2.0, color=color, label=label, zorder=3)
ax.set_title("Churners go quiet: days since last order", loc="left", pad=14)
ax.set_xlabel("Days since last order (capped at 180)")
ax.set_ylabel("Density")
ax.legend(loc="upper right")
vs.despine_y(ax); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/recency_distribution.png"); plt.close(fig)

# ---- 4. Churn by unresolved support tickets -------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
t = df.copy(); t["unres"] = t["unresolved_tickets"].clip(0, 3)
tr = t.groupby("unres")["churned"].mean() * 100
labels = ["0", "1", "2", "3+"]
ax.bar(labels, tr.values, color=vs.BLUE, width=0.55, zorder=3)
for i, v in enumerate(tr.values):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10.5, fontweight="bold", color=vs.INK)
ax.set_title("Unresolved support tickets drive customers away", loc="left", pad=14)
ax.set_xlabel("Unresolved tickets (last 6 months)")
ax.set_ylabel("90-day churn rate (%)")
vs.despine_y(ax); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/churn_by_tickets.png"); plt.close(fig)

# ---- 5. Feature correlation with churn ------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))
num = df.select_dtypes("number").drop(columns=["churned"])
corr = num.corrwith(df["churned"]).sort_values()
nice = {
    "is_autoship": "Autoship member", "tenure_months": "Tenure (months)",
    "orders_per_quarter": "Orders per quarter", "email_open_rate": "Email open rate",
    "app_sessions_per_month": "App sessions / month", "pct_pharmacy_spend": "Pharmacy share of spend",
    "num_pets": "Number of pets", "avg_order_value": "Avg order value",
    "total_spend_12m": "Total spend (12m)", "used_promo_last_90d": "Used promo (90d)",
    "support_tickets_6m": "Support tickets (6m)", "refunds_12m": "Refunds (12m)",
    "avg_delivery_days": "Avg delivery days", "price_increase_flag": "Saw price increase",
    "unresolved_tickets": "Unresolved tickets", "days_since_last_order": "Days since last order",
}
corr.index = [nice.get(c, c) for c in corr.index]
colors = [vs.ORANGE if v > 0 else vs.BLUE for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, height=0.62, zorder=3)
ax.axvline(0, color=vs.BASELINE, linewidth=1)
ax.set_title("What correlates with churn", loc="left", pad=14)
ax.set_xlabel("Pearson correlation with churn  (orange = raises risk, blue = lowers risk)")
vs.despine_y(ax); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/feature_correlation.png"); plt.close(fig)

# ---- 6. Dataset snapshot: churn split + pet mix ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
ax = axes[0]
counts = df["churned"].value_counts()
ax.bar(["Retained", "Churned"], [counts[0], counts[1]],
       color=[vs.BLUE, vs.ORANGE], width=0.5, zorder=3)
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 600, f"{v:,}\n({v/len(df):.0%})", ha="center",
            fontsize=10.5, fontweight="bold", color=vs.INK)
ax.set_ylim(0, counts[0] * 1.25)
ax.set_title("Class balance", loc="left", pad=14)
ax.set_ylabel("Customers")
vs.despine_y(ax); ax.grid(axis="x", visible=False)

ax = axes[1]
mix = df["pet_type"].value_counts().head(6)[::-1]
ax.barh(mix.index, mix.values, color=vs.BLUE, height=0.6, zorder=3)
ax.set_title("Customers by pet type (top 6)", loc="left", pad=14)
ax.set_xlabel("Customers")
vs.despine_y(ax); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/dataset_snapshot.png"); plt.close(fig)

print(f"Overall churn rate: {churn_rate:.1%}  |  customers: {len(df):,}")
print("Charts saved to ../figures/")
