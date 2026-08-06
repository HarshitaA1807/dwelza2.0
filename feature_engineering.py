"""
feature_engineering.py
------------------------
Creates derived features that capture "multi-source" signals
(location, amenities, accessibility) and encodes categorical data
so the model can use it.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    # 1. Price per sqft (useful for analysis, not used as a model input
    #    since it's derived FROM the target — drop before training)
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]

    # 2. Amenity/accessibility composite score (multi-source fusion idea:
    #    combine amenities + proximity to metro/school/hospital into one signal)
    df["Accessibility_Score"] = (
        (1 / (1 + df["Dist_to_Metro_km"])) * 0.4
        + (1 / (1 + df["Dist_to_School_km"])) * 0.3
        + (1 / (1 + df["Dist_to_Hospital_km"])) * 0.3
    )

    # 3. Age buckets
    df["Age_Category"] = pd.cut(
        df["Age_years"],
        bins=[-1, 5, 15, 30, 100],
        labels=["New", "Moderate", "Old", "Very Old"],
    )

    # 4. Encode categorical columns
    cat_cols = ["City", "Locality", "Furnishing", "Age_Category"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/house-price-prediction/data/cleaned_data.csv")
    df_fe, encoders = engineer_features(df)
    df_fe.to_csv("/home/claude/house-price-prediction/data/featured_data.csv", index=False)
    print(f"Feature engineering done. Columns now: {list(df_fe.columns)}")
