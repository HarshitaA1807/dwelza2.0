"""
train_model_advanced.py
-------------------------
This is the "advanced" version of model training:
  1. Compares 5 model families instead of 3 (adds XGBoost, LightGBM)
  2. Tunes hyperparameters with RandomizedSearchCV (not default settings)
  3. Uses K-Fold cross-validation, not a single train/test split
  4. Builds a Stacking Ensemble (combines multiple models -- this is
     what most Kaggle-winning and production models actually use)
  5. Saves SHAP explainability values if the shap library is installed

Run this INSTEAD of train_model.py once you have a real dataset.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_engineering_v2 import engineer_features, ADVANCED_FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "cleaned_data.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

TARGET_COL = "Price"

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def tune_random_forest(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=15, cv=3, scoring="r2", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best RF params: {search.best_params_}")
    return search.best_estimator_


def tune_gradient_boosting(X_train, y_train):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=15, cv=3, scoring="r2", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best GB params: {search.best_params_}")
    return search.best_estimator_


def tune_xgboost(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist"),
        param_distributions=param_dist,
        n_iter=20, cv=3, scoring="r2", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best XGBoost params: {search.best_params_}")
    return search.best_estimator_


def tune_lightgbm(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 400, 600],
        "num_leaves": [20, 31, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
    }
    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions=param_dist,
        n_iter=15, cv=3, scoring="r2", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best LightGBM params: {search.best_params_}")
    return search.best_estimator_


def evaluate(name, model, X_test, y_test, X_full, y_full):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # 5-fold cross-validation score (more reliable than a single split)
    cv_scores = cross_val_score(model, X_full, y_full, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

    print(f"{name:20s} | RMSE: {rmse:>12,.0f} | MAE: {mae:>12,.0f} | "
          f"Test R2: {r2:.4f} | CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2, "cv_r2": cv_scores.mean()}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df_fe, encoders = engineer_features(df)

    X = df_fe[ADVANCED_FEATURE_COLS]
    y = df_fe[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTuning + training models (this takes a few minutes on a real dataset)...")
    print("-" * 100)

    results = []
    t0 = time.time()
    print("Tuning Random Forest...")
    rf = tune_random_forest(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test, X, y))

    print("Tuning Gradient Boosting...")
    gb = tune_gradient_boosting(X_train, y_train)
    results.append(evaluate("Gradient Boosting", gb, X_test, y_test, X, y))

    base_estimators = [("rf", rf), ("gb", gb)]

    if HAS_XGB:
        print("Tuning XGBoost...")
        xgb = tune_xgboost(X_train, y_train)
        results.append(evaluate("XGBoost", xgb, X_test, y_test, X, y))
        base_estimators.append(("xgb", xgb))
    else:
        print("xgboost not installed -- skipping (pip install xgboost)")

    if HAS_LGBM:
        print("Tuning LightGBM...")
        lgbm = tune_lightgbm(X_train, y_train)
        results.append(evaluate("LightGBM", lgbm, X_test, y_test, X, y))
        base_estimators.append(("lgbm", lgbm))
    else:
        print("lightgbm not installed -- skipping (pip install lightgbm)")

    # --- Stacking ensemble: combines all tuned models via a meta-learner ---
    print("Training Stacking Ensemble...")
    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    results.append(evaluate("Stacking Ensemble", stack, X_test, y_test, X, y))

    print("-" * 100)
    print(f"Total tuning+training time: {time.time() - t0:.1f}s")

    best = max(results, key=lambda r: r["cv_r2"])
    print(f"\nBest model: {best['name']} (CV R2 = {best['cv_r2']:.4f})")

    joblib.dump(best["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(ADVANCED_FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    if hasattr(best["model"], "feature_importances_"):
        importance = pd.Series(
            best["model"].feature_importances_, index=ADVANCED_FEATURE_COLS
        ).sort_values(ascending=False)
        importance.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))
        print("\nTop features:")
        print(importance.head(10))

    # --- SHAP explainability (optional, only if shap is installed) ---
    try:
        import shap
        explainer = shap.TreeExplainer(best["model"]) if hasattr(best["model"], "feature_importances_") else None
        if explainer:
            shap_values = explainer.shap_values(X_test.iloc[:200])
            joblib.dump((explainer, shap_values, X_test.iloc[:200]), os.path.join(MODEL_DIR, "shap_data.pkl"))
            print("SHAP explainability data saved.")
    except ImportError:
        print("shap not installed -- skipping explainability (pip install shap)")

    print(f"\nAll model comparison results saved. Best: {best['name']}")


if __name__ == "__main__":
    main()
