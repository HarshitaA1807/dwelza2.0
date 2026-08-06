"""
streamlit_app.py
------------------
Run with:  streamlit run app/streamlit_app.py
This is the actual website UI for your house price prediction system.
"""

import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 India House Price Predictor")
st.write(
    "An ML-powered tool that predicts house prices using property details, "
    "location, and nearby amenities — with a transparent price range instead "
    "of a single black-box number."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chennai"])
    locality_options = {
        "Delhi": ["Dwarka", "Rohini", "Saket", "Karol Bagh"],
        "Mumbai": ["Andheri", "Borivali", "Powai", "Thane"],
        "Bangalore": ["Whitefield", "Koramangala", "Indiranagar", "Electronic City"],
        "Pune": ["Hinjewadi", "Kothrud", "Baner", "Viman Nagar"],
        "Hyderabad": ["Gachibowli", "Madhapur", "Kukatpally", "Banjara Hills"],
        "Chennai": ["Anna Nagar", "OMR", "Velachery", "T Nagar"],
    }
    locality = st.selectbox("Locality", locality_options[city])
    area_sqft = st.number_input("Area (sq. ft.)", min_value=200, max_value=10000, value=1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
    age_years = st.slider("Property Age (years)", 0, 40, 5)
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])

with col2:
    dist_to_metro_km = st.number_input("Distance to Metro/Transit (km)", 0.0, 20.0, 2.0)
    dist_to_school_km = st.number_input("Distance to School (km)", 0.0, 10.0, 1.0)
    dist_to_hospital_km = st.number_input("Distance to Hospital (km)", 0.0, 15.0, 1.5)
    num_amenities = st.slider("Number of Amenities (gym, pool, security, etc.)", 0, 15, 6)
    parking = st.selectbox("Parking Spaces", [0, 1, 2], index=1)

st.divider()

if st.button("🔍 Predict Price", use_container_width=True):
    result = predict_price(
        city=city, locality=locality, area_sqft=area_sqft, bhk=bhk,
        age_years=age_years, dist_to_metro_km=dist_to_metro_km,
        dist_to_school_km=dist_to_school_km, dist_to_hospital_km=dist_to_hospital_km,
        num_amenities=num_amenities, furnishing=furnishing, parking=parking,
    )

    st.success(f"### Estimated Price: ₹{result['predicted_price']:,.0f}")
    st.write(f"**Likely Range:** ₹{result['range_low']:,.0f} – ₹{result['range_high']:,.0f}")

    st.caption(
        "This range reflects model uncertainty across multiple decision trees — "
        "unlike single-number estimates from typical listing sites."
    )

    # Show feature importance if available
    try:
        importance = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "models", "feature_importance.csv"),
            index_col=0,
        )
        st.subheader("What influenced this prediction most")
        st.bar_chart(importance.head(8))
    except FileNotFoundError:
        pass

st.divider()
st.caption("Built with Python, scikit-learn & Streamlit | Multi-source data fusion model")
