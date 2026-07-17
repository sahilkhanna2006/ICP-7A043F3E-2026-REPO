# Project 2: Regression for Price Prediction - Final Report

## Models Compared (5-Fold Cross-Validation RMSE)
- Linear Regression: 15,469.64
- Random Forest (default): 21,995.44

## Hyperparameter Tuning
Best parameters found via GridSearchCV: `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`

## Final Test Set Performance
- RMSE: 23,602.84
- MAE: 18,822.23
- R2 Score: 0.9090

## Feature Importance
- area_sqft: 0.8190
- location_score: 0.1009
- age_years: 0.0317
- distance_to_city_km: 0.0279
- bedrooms: 0.0126
- bathrooms: 0.0040
- garage_spaces: 0.0039

## Files in this folder
- `02_model_tuning_and_evaluation.py` - main script
- `final_tuned_rf_model.joblib` - trained model
- `scaler.joblib` - fitted feature scaler
- `feature_importance.png` - feature importance chart
- `actual_vs_predicted.png` - error analysis plot
