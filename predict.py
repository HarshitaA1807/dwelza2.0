"""
predict.py
-----------
Loads the saved model + encoders and provides a single function,
predict_price(), that the Streamlit app (or anything else) can call.

Uses paths relative to this file (not hardcoded), and auto-trains
the model on first run if no saved model is found yet -- this makes
the app work on Streamlit Cloud without manually uploading .pkl files.
"""

import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_DIR = os.path.dirname(BASE_DIR)                         # project root
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

_model_path = os.path.join(MODEL_DIR, "best_model.pkl")

if not os.path.exists(_model_path):
    # No trained model found yet (e.g. fresh clone / fresh cloud deploy) ->
    # generate sample data, clean it, and train automatically.
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    import generate_sample_data  # noqa: F401  (running this module generates data/raw_data.csv)
    from data_preprocessing import load_data, clean_data
    import train_model as _train_model

    raw_path = os.path.join(DATA_DIR, "raw_data.csv")
    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_clean.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)

    _train_model.DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
    _train_model.MODEL_DIR = MODEL_DIR
    _train_model.main()

_model = joblib.load(_model_path)
_encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
_feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))


def _safe_encode(encoder, value):
    """Encode a category, falling back to the most common class if unseen."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0  # fallback for unseen categories


def predict_price(
    city, locality, area_sqft, bhk, age_years,
    dist_to_metro_km, dist_to_school_km, dist_to_hospital_km,
    num_amenities, furnishing, parking,
):
    # Age category bucket must match feature_engineering.py logic
    if age_years <= 5:
        age_cat = "New"
    elif age_years <= 15:
        age_cat = "Moderate"
    elif age_years <= 30:
        age_cat = "Old"
    else:
        age_cat = "Very Old"

    accessibility_score = (
        (1 / (1 + dist_to_metro_km)) * 0.4
        + (1 / (1 + dist_to_school_km)) * 0.3
        + (1 / (1 + dist_to_hospital_km)) * 0.3
    )

    row = {
        "Area_sqft": area_sqft,
        "BHK": bhk,
        "Age_years": age_years,
        "Dist_to_Metro_km": dist_to_metro_km,
        "Dist_to_School_km": dist_to_school_km,
        "Dist_to_Hospital_km": dist_to_hospital_km,
        "Num_Amenities": num_amenities,
        "Parking": parking,
        "Accessibility_Score": accessibility_score,
        "City_enc": _safe_encode(_encoders["City"], city),
        "Locality_enc": _safe_encode(_encoders["Locality"], locality),
        "Furnishing_enc": _safe_encode(_encoders["Furnishing"], furnishing),
        "Age_Category_enc": _safe_encode(_encoders["Age_Category"], age_cat),
    }

    X = np.array([[row[col] for col in _feature_cols]])
    predicted_price = _model.predict(X)[0]

    # Confidence range using individual tree predictions (Random Forest only)
    if hasattr(_model, "estimators_"):
        tree_preds = np.array([t.predict(X)[0] for t in _model.estimators_])
        low, high = np.percentile(tree_preds, [10, 90])
    else:
        low, high = predicted_price * 0.9, predicted_price * 1.1

    return {
        "predicted_price": round(predicted_price, -3),
        "range_low": round(low, -3),
        "range_high": round(high, -3),
    }


if __name__ == "__main__":
    result = predict_price(
        city="Bangalore", locality="Whitefield", area_sqft=1200, bhk=3,
        age_years=5, dist_to_metro_km=2.0, dist_to_school_km=1.0,
        dist_to_hospital_km=1.5, num_amenities=6, furnishing="Semi-Furnished",
        parking=1,
    )
    print(result)
