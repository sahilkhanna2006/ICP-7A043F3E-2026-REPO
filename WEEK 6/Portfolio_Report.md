# Final Portfolio Report
### Machine Learning (AI) Self-Learning Internship — InternCareerPath
**Portal ID:** ICP-BB250721-2026 | **Repo:** ICP-BB250721-2026-REPO

---

## Overview
This portfolio contains two completed machine learning projects, covering both major
branches of ML: **supervised learning** (regression) and **unsupervised learning**
(clustering). Each project includes data preparation, model development, tuning,
evaluation, and documented results.

---

## Project 1: Regression for Price Prediction
**Location:** `Week2/` (data prep + baseline) and `Week3/` (tuning + final evaluation)

**Problem:** Estimate house prices from property features (area, bedrooms,
bathrooms, age, distance to city, garage spaces, location score).

**Approach:**
- Explored and visualized a 1,000-row housing dataset
- Established a Linear Regression baseline (R² ≈ 0.96 on this synthetic data)
- Compared against Random Forest using 5-fold cross-validation
- Tuned Random Forest hyperparameters with GridSearchCV
- Evaluated final model with RMSE, MAE, and R², plus feature importance and
  actual-vs-predicted error analysis

**Key finding:** `area_sqft` was by far the strongest predictor of price, followed by
`location_score`. Full details in `Week3/model_report.md`.

---

## Project 2: Customer Clustering
**Location:** `Week4/` (data prep) and `Week5/` (clustering + evaluation)

**Problem:** Segment customers into meaningful groups for targeted marketing, using
age, annual income, and spending score.

**Approach:**
- Generated and explored a 320-customer dataset
- Used the Elbow Method and Silhouette Score to select the optimal number of
  clusters (k=4)
- Applied K-Means and compared against DBSCAN
- Visualized clusters in 2D using PCA
- Translated statistical clusters into business-meaningful segments

**Key finding:** Four clear segments emerged — High-Value Customers, Cautious
High-Earners, Young Enthusiastic Spenders, and Budget-Conscious Customers — with a
strong silhouette score of ~0.58. Full details in `Week5/model_report.md`.

---

## Skills Demonstrated
- Data preprocessing & feature scaling
- Exploratory data analysis & visualization
- Supervised learning (Linear Regression, Random Forest)
- Unsupervised learning (K-Means, DBSCAN)
- Cross-validation & hyperparameter tuning (GridSearchCV)
- Dimensionality reduction (PCA)
- Model evaluation (RMSE, MAE, R², Silhouette Score)
- Reproducible, documented ML workflows

## Repository Links
- GitHub Repository: `https://github.com/YOUR-USERNAME/ICP-BB250721-2026-REPO`
  *(replace with your actual repo URL before submitting to the portal)*

## Notes
- All datasets are synthetically generated (fully reproducible via fixed random
  seeds) to keep the projects self-contained and runnable without internet access.
  They can be swapped for real-world datasets (e.g., a Kaggle housing dataset or
  Mall Customer Segmentation dataset) if preferred.
