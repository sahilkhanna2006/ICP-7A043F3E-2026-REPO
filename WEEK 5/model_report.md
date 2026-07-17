# Project 3: Customer Clustering - Final Report

## Method
- Features used: age, annual_income_k, spending_score (scaled with StandardScaler)
- Optimal k selected via Silhouette Score: **k=4**
- K-Means Silhouette Score: 0.5816
- DBSCAN found 4 clusters with 1 noise points (eps=0.5, min_samples=5)

## Cluster Profiles (K-Means)

|   kmeans_cluster |   age |   annual_income_k |   spending_score |   count | business_label                                            |
|-----------------:|------:|------------------:|-----------------:|--------:|:----------------------------------------------------------|
|                0 |  23.4 |              27.9 |             77.4 |      80 | Young Enthusiastic Spenders (lower income, high spending) |
|                1 |  47   |              79.6 |             18.9 |      80 | Cautious High-Earners (high income, low spending)         |
|                2 |  38.9 |              78.8 |             74.3 |      80 | High-Value Customers (high income, high spending)         |
|                3 |  56.7 |              27.4 |             19.5 |      80 | Budget-Conscious Customers (low income, low spending)     |

## Business Interpretation
- **Cluster 0** (80 customers): Young Enthusiastic Spenders (lower income, high spending)
- **Cluster 1** (80 customers): Cautious High-Earners (high income, low spending)
- **Cluster 2** (80 customers): High-Value Customers (high income, high spending)
- **Cluster 3** (80 customers): Budget-Conscious Customers (low income, low spending)

## Files in this folder
- `02_clustering_and_evaluation.py` - main script
- `clustered_customers.csv` - final labeled dataset
- `kmeans_model.joblib` - trained K-Means model
- `cluster_selection.png` - elbow method & silhouette score plots
- `cluster_visualization.png` - PCA-based cluster visualization
