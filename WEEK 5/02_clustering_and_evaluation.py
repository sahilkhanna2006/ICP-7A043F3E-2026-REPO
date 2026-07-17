"""
Project 3: Customer Clustering
Week 5 - Clustering, Evaluation & Business Interpretation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# -----------------------------
# 1. Load prepared data
# -----------------------------
df = pd.read_csv("../Week4/customer_data.csv")
scaled_df = pd.read_csv("../Week4/customer_data_scaled.csv")
X = scaled_df.values

# -----------------------------
# 2. Elbow Method - find optimal k
# -----------------------------
inertias = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")

plt.tight_layout()
plt.savefig("cluster_selection.png", dpi=120)

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"Silhouette scores by k: {dict(zip(k_range, [round(s,4) for s in silhouette_scores]))}")
print(f"Best k by silhouette score: {best_k}")

# -----------------------------
# 3. Final K-Means with best k
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X)
kmeans_sil = silhouette_score(X, df["kmeans_cluster"])
print(f"\nK-Means (k={best_k}) Silhouette Score: {kmeans_sil:.4f}")

# -----------------------------
# 4. DBSCAN for comparison
# -----------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(X)
n_dbscan_clusters = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
n_noise = (df["dbscan_cluster"] == -1).sum()
print(f"DBSCAN found {n_dbscan_clusters} clusters and {n_noise} noise points")
if n_dbscan_clusters > 1:
    mask = df["dbscan_cluster"] != -1
    dbscan_sil = silhouette_score(X[mask], df.loc[mask, "dbscan_cluster"])
    print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_sil:.4f}")

# -----------------------------
# 5. PCA for 2D visualization
# -----------------------------
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X)
df["pca_1"] = pca_coords[:, 0]
df["pca_2"] = pca_coords[:, 1]
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter1 = axes[0].scatter(df["pca_1"], df["pca_2"], c=df["kmeans_cluster"], cmap="viridis")
axes[0].set_title(f"K-Means Clusters (k={best_k}) - PCA View")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
plt.colorbar(scatter1, ax=axes[0])

scatter2 = axes[1].scatter(df["pca_1"], df["pca_2"], c=df["dbscan_cluster"], cmap="plasma")
axes[1].set_title("DBSCAN Clusters - PCA View")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig("cluster_visualization.png", dpi=120)

# -----------------------------
# 6. Business interpretation of K-Means clusters
# -----------------------------
cluster_profile = df.groupby("kmeans_cluster")[["age", "annual_income_k", "spending_score"]].mean().round(1)
cluster_counts = df["kmeans_cluster"].value_counts().sort_index()
cluster_profile["count"] = cluster_counts
print("\n=== Cluster Profiles ===")
print(cluster_profile)

def interpret(row):
    if row["annual_income_k"] > 55 and row["spending_score"] > 55:
        return "High-Value Customers (high income, high spending)"
    elif row["annual_income_k"] > 55 and row["spending_score"] <= 55:
        return "Cautious High-Earners (high income, low spending)"
    elif row["annual_income_k"] <= 55 and row["spending_score"] > 55:
        return "Young Enthusiastic Spenders (lower income, high spending)"
    else:
        return "Budget-Conscious Customers (low income, low spending)"

cluster_profile["business_label"] = cluster_profile.apply(interpret, axis=1)
print("\n=== Business Interpretation ===")
print(cluster_profile[["business_label", "count"]])

# -----------------------------
# 7. Save outputs
# -----------------------------
df.to_csv("clustered_customers.csv", index=False)
joblib.dump(kmeans, "kmeans_model.joblib")

with open("model_report.md", "w") as f:
    f.write("# Project 3: Customer Clustering - Final Report\n\n")
    f.write("## Method\n")
    f.write("- Features used: age, annual_income_k, spending_score (scaled with StandardScaler)\n")
    f.write(f"- Optimal k selected via Silhouette Score: **k={best_k}**\n")
    f.write(f"- K-Means Silhouette Score: {kmeans_sil:.4f}\n")
    f.write(f"- DBSCAN found {n_dbscan_clusters} clusters with {n_noise} noise points (eps=0.5, min_samples=5)\n\n")
    f.write("## Cluster Profiles (K-Means)\n\n")
    f.write(cluster_profile.to_markdown())
    f.write("\n\n## Business Interpretation\n")
    for idx, row in cluster_profile.iterrows():
        f.write(f"- **Cluster {idx}** ({int(row['count'])} customers): {row['business_label']}\n")
    f.write("\n## Files in this folder\n")
    f.write("- `02_clustering_and_evaluation.py` - main script\n")
    f.write("- `clustered_customers.csv` - final labeled dataset\n")
    f.write("- `kmeans_model.joblib` - trained K-Means model\n")
    f.write("- `cluster_selection.png` - elbow method & silhouette score plots\n")
    f.write("- `cluster_visualization.png` - PCA-based cluster visualization\n")

print("\nSaved model_report.md, model, and plots.")
