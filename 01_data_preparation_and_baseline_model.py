"""
Project 2: Regression for Price Prediction
Week 2 - Data Preparation & Initial Model Training

Goal: Load the data, explore it, preprocess it, and train a baseline
Linear Regression model to see where we're starting from.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -----------------------------
# 1. Load the data
# -----------------------------
df = pd.read_csv("housing_data.csv")
print("Dataset shape:", df.shape)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# 2. Quick EDA
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].scatter(df["area_sqft"], df["price"], alpha=0.4)
axes[0, 0].set_xlabel("Area (sqft)"); axes[0, 0].set_ylabel("Price")
axes[0, 0].set_title("Area vs Price")

axes[0, 1].scatter(df["distance_to_city_km"], df["price"], alpha=0.4, color="orange")
axes[0, 1].set_xlabel("Distance to City (km)"); axes[0, 1].set_ylabel("Price")
axes[0, 1].set_title("Distance vs Price")

axes[1, 0].hist(df["price"], bins=30, color="green", alpha=0.7)
axes[1, 0].set_xlabel("Price"); axes[1, 0].set_title("Price Distribution")

corr = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
axes[1, 1].barh(corr.index[1:], corr.values[1:])
axes[1, 1].set_title("Feature Correlation with Price")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
print("\nSaved EDA plot -> eda_overview.png")
print("\nCorrelation with price:\n", corr)

# -----------------------------
# 3. Preprocessing
# -----------------------------
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.joblib")

# -----------------------------
# 4. Baseline model: Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
preds = lr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n=== Baseline Linear Regression Results ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R2:   {r2:.4f}")

joblib.dump(lr, "baseline_linear_model.joblib")

with open("progress_notes.md", "w") as f:
    f.write("# Week 2 Progress Notes\n\n")
    f.write("## What was done\n")
    f.write("- Loaded and explored the housing dataset (1000 rows, 7 features)\n")
    f.write("- Checked for missing values (none found)\n")
    f.write("- Visualized relationships between features and price (see eda_overview.png)\n")
    f.write("- Split data 80/20 train/test\n")
    f.write("- Scaled features using StandardScaler\n")
    f.write("- Trained a baseline Linear Regression model\n\n")
    f.write("## Baseline Results\n")
    f.write(f"- RMSE: {rmse:,.2f}\n")
    f.write(f"- MAE: {mae:,.2f}\n")
    f.write(f"- R2 Score: {r2:.4f}\n\n")
    f.write("## Next steps (Week 3)\n")
    f.write("- Try Random Forest Regressor\n")
    f.write("- Perform cross-validation\n")
    f.write("- Hyperparameter tuning with GridSearchCV\n")
    f.write("- Analyze feature importance and errors\n")

print("\nSaved progress_notes.md")
