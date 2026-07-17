"""
Project 2: Regression for Price Prediction
Week 3 - Advanced Model, Cross-Validation, Hyperparameter Tuning & Final Evaluation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -----------------------------
# 1. Load data (same split logic as Week 2, fixed seed => reproducible)
# -----------------------------
df = pd.read_csv("../Week2/housing_data.csv")
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 2. Cross-validation: compare baseline LR vs Random Forest
# -----------------------------
lr = LinearRegression()
lr_cv_scores = cross_val_score(
    lr, X_train_scaled, y_train, cv=5, scoring="neg_root_mean_squared_error"
)

rf_base = RandomForestRegressor(random_state=42)
rf_cv_scores = cross_val_score(
    rf_base, X_train_scaled, y_train, cv=5, scoring="neg_root_mean_squared_error"
)

print("=== 5-Fold Cross-Validation RMSE ===")
print(f"Linear Regression:  {-lr_cv_scores.mean():,.2f} (+/- {lr_cv_scores.std():,.2f})")
print(f"Random Forest:      {-rf_cv_scores.mean():,.2f} (+/- {rf_cv_scores.std():,.2f})")

# -----------------------------
# 3. Hyperparameter tuning with GridSearchCV
# -----------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

print("\n=== Best Hyperparameters ===")
print(grid_search.best_params_)
print(f"Best CV RMSE: {-grid_search.best_score_:,.2f}")

best_rf = grid_search.best_estimator_

# -----------------------------
# 4. Final evaluation on held-out test set
# -----------------------------
preds = best_rf.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n=== Final Test Set Results (Tuned Random Forest) ===")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R2:   {r2:.4f}")

# -----------------------------
# 5. Feature importance
# -----------------------------
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)

# -----------------------------
# 6. Error analysis plot
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)

# -----------------------------
# 7. Save final model
# -----------------------------
joblib.dump(best_rf, "final_tuned_rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")

with open("model_report.md", "w") as f:
    f.write("# Project 2: Regression for Price Prediction - Final Report\n\n")
    f.write("## Models Compared (5-Fold Cross-Validation RMSE)\n")
    f.write(f"- Linear Regression: {-lr_cv_scores.mean():,.2f}\n")
    f.write(f"- Random Forest (default): {-rf_cv_scores.mean():,.2f}\n\n")
    f.write("## Hyperparameter Tuning\n")
    f.write(f"Best parameters found via GridSearchCV: `{grid_search.best_params_}`\n\n")
    f.write("## Final Test Set Performance\n")
    f.write(f"- RMSE: {rmse:,.2f}\n")
    f.write(f"- MAE: {mae:,.2f}\n")
    f.write(f"- R2 Score: {r2:.4f}\n\n")
    f.write("## Feature Importance\n")
    for feat, val in importances.items():
        f.write(f"- {feat}: {val:.4f}\n")
    f.write("\n## Files in this folder\n")
    f.write("- `02_model_tuning_and_evaluation.py` - main script\n")
    f.write("- `final_tuned_rf_model.joblib` - trained model\n")
    f.write("- `scaler.joblib` - fitted feature scaler\n")
    f.write("- `feature_importance.png` - feature importance chart\n")
    f.write("- `actual_vs_predicted.png` - error analysis plot\n")

print("\nSaved model_report.md, model, and plots.")
