"""
predict.py
-----------
Loads the saved model + encoders and provides a single function,
predict_price(), that the Streamlit app (or anything else) can call.
"""

import numpy as np
import joblib

MODEL_DIR = "/home/claude/house-price-prediction/models"

_model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
_encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
_feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")


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
