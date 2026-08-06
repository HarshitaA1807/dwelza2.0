"""
feature_engineering_v2.py
---------------------------
Advanced feature engineering for the multi-source fusion model.
Adds interaction features, city-tier classification, and a
"luxury score" -- the kind of engineered signals that separate a
basic model from one that actually competes with production systems.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Tier classification -- a real signal Indian real estate models use,
# since Tier-1 cities behave very differently from Tier-2/3 markets.
TIER_1 = {"Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"}
TIER_2 = {"Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh", "Indore", "Nagpur"}


def city_tier(city):
    if city in TIER_1:
        return 1
    elif city in TIER_2:
        return 2
    return 3


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    # --- Basic derived features ---
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]
    df["City_Tier"] = df["City"].apply(city_tier)

    # --- Accessibility composite (multi-source fusion signal) ---
    df["Accessibility_Score"] = (
        (1 / (1 + df["Dist_to_Metro_km"])) * 0.4
        + (1 / (1 + df["Dist_to_School_km"])) * 0.3
        + (1 / (1 + df["Dist_to_Hospital_km"])) * 0.3
    )

    # --- Age buckets ---
    df["Age_Category"] = pd.cut(
        df["Age_years"], bins=[-1, 5, 15, 30, 100],
        labels=["New", "Moderate", "Old", "Very Old"],
    )

    # --- Interaction features (this is what makes a model "advanced" --
    # capturing how features combine, not just treating them independently) ---
    df["Area_x_BHK"] = df["Area_sqft"] * df["BHK"]
    df["Area_per_BHK"] = df["Area_sqft"] / df["BHK"].replace(0, 1)
    df["Amenity_density"] = df["Num_Amenities"] / (df["Area_sqft"] / 1000)
    df["Luxury_Score"] = (
        df["Num_Amenities"] * 0.4
        + df["Parking"] * 0.3
        + (df["Furnishing"] == "Furnished").astype(int) * 0.3
    )
    df["Newness_x_Accessibility"] = (
        (1 / (1 + df["Age_years"])) * df["Accessibility_Score"]
    )

    # --- Encode categoricals ---
    cat_cols = ["City", "Locality", "Furnishing", "Age_Category"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


ADVANCED_FEATURE_COLS = [
    "Area_sqft", "BHK", "Age_years",
    "Dist_to_Metro_km", "Dist_to_School_km", "Dist_to_Hospital_km",
    "Num_Amenities", "Parking", "Accessibility_Score", "City_Tier",
    "Area_x_BHK", "Area_per_BHK", "Amenity_density", "Luxury_Score",
    "Newness_x_Accessibility",
    "City_enc", "Locality_enc", "Furnishing_enc", "Age_Category_enc",
]
