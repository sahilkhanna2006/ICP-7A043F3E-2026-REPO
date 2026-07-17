"""
Project 3: Customer Clustering
Week 4 - Data Preparation

Generates a realistic synthetic mall-customer-style dataset (offline, no
internet needed) and preprocesses it for clustering.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(7)

# -----------------------------
# 1. Generate synthetic customer data with 4 natural segments
# -----------------------------
def make_segment(n, age_range, income_range, spend_range):
    return pd.DataFrame({
        "age": np.random.randint(age_range[0], age_range[1], n),
        "annual_income_k": np.random.uniform(income_range[0], income_range[1], n).round(1),
        "spending_score": np.random.uniform(spend_range[0], spend_range[1], n).round(1),
    })

segments = [
    make_segment(80, (18, 30), (15, 40), (60, 95)),   # young, low income, big spenders
    make_segment(80, (30, 50), (60, 100), (55, 90)),  # mid-age, high income, high spenders
    make_segment(80, (35, 60), (60, 100), (5, 35)),   # mid-age, high income, careful spenders
    make_segment(80, (45, 70), (15, 40), (5, 35)),    # older, low income, low spenders
]

df = pd.concat(segments, ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
df.insert(0, "customer_id", range(1, len(df) + 1))

df.to_csv("customer_data.csv", index=False)
print("Dataset shape:", df.shape)
print(df.head())
print("\nSummary stats:\n", df.describe())

# -----------------------------
# 2. EDA
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["age"], bins=20, color="steelblue")
axes[0].set_title("Age Distribution")

axes[1].hist(df["annual_income_k"], bins=20, color="seagreen")
axes[1].set_title("Annual Income (k$) Distribution")

axes[2].scatter(df["annual_income_k"], df["spending_score"], alpha=0.5, color="darkorange")
axes[2].set_xlabel("Annual Income (k$)")
axes[2].set_ylabel("Spending Score")
axes[2].set_title("Income vs Spending Score")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
print("\nSaved eda_overview.png")

# -----------------------------
# 3. Preprocessing - scale features for clustering
# -----------------------------
features = df[["age", "annual_income_k", "spending_score"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled, columns=features.columns)
scaled_df.to_csv("customer_data_scaled.csv", index=False)
print("\nSaved scaled features -> customer_data_scaled.csv")

with open("progress_notes.md", "w") as f:
    f.write("# Week 4 Progress Notes\n\n")
    f.write("## What was done\n")
    f.write("- Generated a synthetic customer dataset (320 customers) with age, "
            "annual income, and spending score\n")
    f.write("- Performed EDA to understand distributions and relationships (see eda_overview.png)\n")
    f.write("- Scaled features with StandardScaler to prepare for clustering "
            "(K-Means/DBSCAN are distance-based, so scaling matters)\n\n")
    f.write("## Next steps (Week 5)\n")
    f.write("- Determine optimal number of clusters (Elbow Method + Silhouette Score)\n")
    f.write("- Apply K-Means and DBSCAN\n")
    f.write("- Reduce dimensions with PCA for visualization\n")
    f.write("- Interpret clusters into business-meaningful segments\n")

print("Saved progress_notes.md")
