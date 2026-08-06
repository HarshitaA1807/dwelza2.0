"""
train_model.py
----------------
Trains multiple regression models on the engineered dataset,
compares their performance (RMSE, MAE, R2), and saves the best
model + the label encoders + feature column list to /models
so the Streamlit app can load them later.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_engineering import engineer_features

DATA_PATH = "/home/claude/house-price-prediction/data/cleaned_data.csv"
MODEL_DIR = "/home/claude/house-price-prediction/models"

FEATURE_COLS = [
    "Area_sqft", "BHK", "Age_years",
    "Dist_to_Metro_km", "Dist_to_School_km", "Dist_to_Hospital_km",
    "Num_Amenities", "Parking", "Accessibility_Score",
    "City_enc", "Locality_enc", "Furnishing_enc", "Age_Category_enc",
]
TARGET_COL = "Price"


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:22s} | RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | R2: {r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = pd.read_csv(DATA_PATH)
    df_fe, encoders = engineer_features(df)

    X = df_fe[FEATURE_COLS]
    y = df_fe[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    print("\nModel comparison:")
    print("-" * 70)
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    # Pick best model by R2 score
    best = max(results, key=lambda r: r["r2"])
    print("-" * 70)
    print(f"\nBest model: {best['name']} (R2 = {best['r2']:.4f})")

    # Save best model, encoders, and feature column order
    joblib.dump(best["model"], f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")

    # Save feature importance (works for RF/GB; skipped for Linear Regression)
    if hasattr(best["model"], "feature_importances_"):
        importance = pd.Series(
            best["model"].feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        importance.to_csv(f"{MODEL_DIR}/feature_importance.csv")
        print("\nTop features driving price:")
        print(importance.head(8))

    print(f"\nSaved model + encoders to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
