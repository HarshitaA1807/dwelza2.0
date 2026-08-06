"""
generate_sample_data.py
------------------------
Creates a realistic sample housing dataset and saves it as both
Excel (.xlsx) and CSV so you can practice cleaning/feature engineering
in Excel before automating with Python.

Replace this with your REAL dataset (Kaggle / government / scraped data)
once you have one. This is only to get your pipeline working end-to-end.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of sample transactions

cities = ["Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chennai"]
localities = {
    "Delhi": ["Dwarka", "Rohini", "Saket", "Karol Bagh"],
    "Mumbai": ["Andheri", "Borivali", "Powai", "Thane"],
    "Bangalore": ["Whitefield", "Koramangala", "Indiranagar", "Electronic City"],
    "Pune": ["Hinjewadi", "Kothrud", "Baner", "Viman Nagar"],
    "Hyderabad": ["Gachibowli", "Madhapur", "Kukatpally", "Banjara Hills"],
    "Chennai": ["Anna Nagar", "OMR", "Velachery", "T Nagar"],
}

rows = []
for _ in range(N):
    city = np.random.choice(cities)
    locality = np.random.choice(localities[city])
    area_sqft = np.random.randint(400, 3000)
    bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
    age_years = np.random.randint(0, 30)
    dist_to_metro_km = round(np.random.uniform(0.2, 15), 2)
    dist_to_school_km = round(np.random.uniform(0.1, 5), 2)
    dist_to_hospital_km = round(np.random.uniform(0.2, 8), 2)
    num_amenities = np.random.randint(0, 12)  # gym, pool, park, security, etc.
    furnishing = np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"])
    parking = np.random.choice([0, 1, 2])

    # base price per sqft depends on city (rough real-world proxy)
    base_rate = {
        "Mumbai": 18000, "Delhi": 12000, "Bangalore": 9000,
        "Pune": 7000, "Hyderabad": 6500, "Chennai": 7500,
    }[city]

    price = (
        area_sqft * base_rate
        + bhk * 150000
        - age_years * 15000
        - dist_to_metro_km * 20000
        - dist_to_school_km * 8000
        - dist_to_hospital_km * 5000
        + num_amenities * 40000
        + parking * 100000
        + np.random.normal(0, 250000)  # noise
    )
    price = max(price, 500000)  # floor

    rows.append([
        city, locality, area_sqft, bhk, age_years, dist_to_metro_km,
        dist_to_school_km, dist_to_hospital_km, num_amenities,
        furnishing, parking, round(price, -3)
    ])

df = pd.DataFrame(rows, columns=[
    "City", "Locality", "Area_sqft", "BHK", "Age_years",
    "Dist_to_Metro_km", "Dist_to_School_km", "Dist_to_Hospital_km",
    "Num_Amenities", "Furnishing", "Parking", "Price"
])

df.to_excel("/home/claude/house-price-prediction/data/raw_data.xlsx", index=False)
df.to_csv("/home/claude/house-price-prediction/data/raw_data.csv", index=False)

print(f"Generated {len(df)} rows.")
print(df.head())
