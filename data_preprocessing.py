"""
data_preprocessing.py
----------------------
Loads the raw dataset, cleans it, removes duplicates/outliers,
and saves a cleaned CSV ready for feature engineering.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    # 2. Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Remove obvious invalid rows
    df = df[df["Area_sqft"] > 100]
    df = df[df["Price"] > 0]

    # 4. Remove outliers using IQR method on Price
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]
    print(f"Removed {before - len(df)} outlier rows (Price IQR method).")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

    df = load_data(os.path.join(DATA_DIR, "raw_data.csv"))
    print(f"Loaded {len(df)} rows.")

    df_clean = clean_data(df)
    df_clean.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)
    print(f"Saved cleaned data: {len(df_clean)} rows.")
