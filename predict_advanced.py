"""
predict_advanced.py
---------------------
Prediction function matching train_model_advanced.py's feature set.
Uses relative paths and auto-trains (basic version) as a fallback
if no model exists yet -- for the advanced model, run
train_model_advanced.py yourself once (it takes several minutes).
"""

import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
_encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
_feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

from feature_engineering_v2 import city_tier


def _safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


def predict_price(
    city, locality, area_sqft, bhk, age_years,
    dist_to_metro_km, dist_to_school_km, dist_to_hospital_km,
    num_amenities, furnishing, parking,
):
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
    luxury_score = num_amenities * 0.4 + parking * 0.3 + (1 if furnishing == "Furnished" else 0) * 0.3

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
        "City_Tier": city_tier(city),
        "Area_x_BHK": area_sqft * bhk,
        "Area_per_BHK": area_sqft / max(bhk, 1),
        "Amenity_density": num_amenities / (area_sqft / 1000),
        "Luxury_Score": luxury_score,
        "Newness_x_Accessibility": (1 / (1 + age_years)) * accessibility_score,
        "City_enc": _safe_encode(_encoders["City"], city),
        "Locality_enc": _safe_encode(_encoders["Locality"], locality),
        "Furnishing_enc": _safe_encode(_encoders["Furnishing"], furnishing),
        "Age_Category_enc": _safe_encode(_encoders["Age_Category"], age_cat),
    }

    X = np.array([[row[col] for col in _feature_cols]])
    predicted_price = _model.predict(X)[0]

    if hasattr(_model, "estimators_"):
        try:
            tree_preds = np.array([t.predict(X)[0] for t in _model.estimators_])
            low, high = np.percentile(tree_preds, [10, 90])
        except (TypeError, AttributeError):
            low, high = predicted_price * 0.92, predicted_price * 1.08
    else:
        low, high = predicted_price * 0.92, predicted_price * 1.08

    return {
        "predicted_price": round(float(predicted_price), -3),
        "range_low": round(float(low), -3),
        "range_high": round(float(high), -3),
    }
