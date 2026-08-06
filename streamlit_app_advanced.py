"""
streamlit_app_advanced.py
----------------------------
Run with: streamlit run app/streamlit_app_advanced.py

Advanced version of the website with:
- Custom styling
- Price prediction with confidence range + feature importance chart
- EMI (home loan) calculator with adjustable interest rate
- Side-by-side property comparison (2 properties)
- Simple price appreciation projection (5-year outlook)
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict_advanced import predict_price

st.set_page_config(page_title="India House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .big-price { font-size: 42px; font-weight: 700; color: #22c55e; }
    .price-range { font-size: 16px; color: #94a3b8; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

CITIES = ["Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chennai"]
LOCALITIES = {
    "Delhi": ["Dwarka", "Rohini", "Saket", "Karol Bagh"],
    "Mumbai": ["Andheri", "Borivali", "Powai", "Thane"],
    "Bangalore": ["Whitefield", "Koramangala", "Indiranagar", "Electronic City"],
    "Pune": ["Hinjewadi", "Kothrud", "Baner", "Viman Nagar"],
    "Hyderabad": ["Gachibowli", "Madhapur", "Kukatpally", "Banjara Hills"],
    "Chennai": ["Anna Nagar", "OMR", "Velachery", "T Nagar"],
}

st.title("🏠 India House Price Predictor")
st.caption("Multi-source ML model — property details + location + amenities + accessibility")

tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "🏘️ Compare Properties", "📊 EMI Calculator"])


def property_inputs(key_prefix=""):
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", CITIES, key=f"{key_prefix}city")
        locality = st.selectbox("Locality", LOCALITIES[city], key=f"{key_prefix}locality")
        area_sqft = st.number_input("Area (sq. ft.)", 200, 10000, 1200, key=f"{key_prefix}area")
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2, key=f"{key_prefix}bhk")
        age_years = st.slider("Property Age (years)", 0, 40, 5, key=f"{key_prefix}age")
        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"], key=f"{key_prefix}furn")
    with col2:
        dist_metro = st.number_input("Distance to Metro/Transit (km)", 0.0, 20.0, 2.0, key=f"{key_prefix}metro")
        dist_school = st.number_input("Distance to School (km)", 0.0, 10.0, 1.0, key=f"{key_prefix}school")
        dist_hospital = st.number_input("Distance to Hospital (km)", 0.0, 15.0, 1.5, key=f"{key_prefix}hosp")
        amenities = st.slider("Number of Amenities", 0, 15, 6, key=f"{key_prefix}amen")
        parking = st.selectbox("Parking Spaces", [0, 1, 2], index=1, key=f"{key_prefix}park")
    return dict(city=city, locality=locality, area_sqft=area_sqft, bhk=bhk, age_years=age_years,
                dist_to_metro_km=dist_metro, dist_to_school_km=dist_school, dist_to_hospital_km=dist_hospital,
                num_amenities=amenities, furnishing=furnishing, parking=parking)


with tab1:
    inputs = property_inputs()
    if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
        result = predict_price(**inputs)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='big-price'>₹{result['predicted_price']:,.0f}</div>", unsafe_allow_html=True)
        c1.caption("Estimated Price")
        c2.metric("Lower Bound", f"₹{result['range_low']:,.0f}")
        c3.metric("Upper Bound", f"₹{result['range_high']:,.0f}")

        st.divider()

        # 5-year price appreciation projection (simple compound growth assumption)
        st.subheader("📈 5-Year Price Outlook (projected)")
        growth_rate = st.slider("Assumed annual appreciation rate (%)", 3.0, 12.0, 6.5, 0.5)
        years = list(range(6))
        projected = [result["predicted_price"] * ((1 + growth_rate / 100) ** y) for y in years]
        proj_df = pd.DataFrame({"Year": [f"Year {y}" for y in years], "Projected Value (₹)": projected})
        st.line_chart(proj_df.set_index("Year"))
        st.caption("This is a simple compound-growth projection based on your assumed rate — not a guarantee.")

        # Feature importance
        try:
            importance = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "..", "models", "feature_importance.csv"), index_col=0
            )
            st.subheader("🔍 What influenced this prediction most")
            st.bar_chart(importance.head(8))
        except FileNotFoundError:
            pass

with tab2:
    st.subheader("Compare two properties side by side")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Property A**")
        inputs_a = property_inputs(key_prefix="a_")
    with colB:
        st.markdown("**Property B**")
        inputs_b = property_inputs(key_prefix="b_")

    if st.button("Compare", use_container_width=True, type="primary"):
        result_a = predict_price(**inputs_a)
        result_b = predict_price(**inputs_b)

        colA, colB = st.columns(2)
        colA.metric("Property A — Estimated Price", f"₹{result_a['predicted_price']:,.0f}")
        colB.metric("Property B — Estimated Price", f"₹{result_b['predicted_price']:,.0f}",
                     delta=f"₹{result_b['predicted_price'] - result_a['predicted_price']:,.0f}")

        price_per_sqft_a = result_a["predicted_price"] / inputs_a["area_sqft"]
        price_per_sqft_b = result_b["predicted_price"] / inputs_b["area_sqft"]
        st.write(f"**Price per sq.ft.** — A: ₹{price_per_sqft_a:,.0f} | B: ₹{price_per_sqft_b:,.0f}")

        if result_a["predicted_price"] < result_b["predicted_price"]:
            st.success("Property A offers better value at this price point.")
        else:
            st.success("Property B offers better value at this price point.")

with tab3:
    st.subheader("🏦 Home Loan EMI Calculator")
    st.write("Use this alongside your predicted price to estimate monthly payments.")

    loan_amount = st.number_input("Loan Amount (₹)", 100000, 100000000, 8000000, step=100000)
    interest_rate = st.slider("Annual Interest Rate (%)", 6.0, 12.0, 8.5, 0.1)
    tenure_years = st.slider("Loan Tenure (years)", 5, 30, 20)

    r = interest_rate / (12 * 100)
    n = tenure_years * 12
    if r > 0:
        emi = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        emi = loan_amount / n

    total_payment = emi * n
    total_interest = total_payment - loan_amount

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly EMI", f"₹{emi:,.0f}")
    c2.metric("Total Interest", f"₹{total_interest:,.0f}")
    c3.metric("Total Payment", f"₹{total_payment:,.0f}")

    st.caption(
        "Note: interest rates shown are for calculation purposes — check current rates with your bank/NBFC "
        "for an actual loan offer, as rates change with RBI policy and lender terms."
    )

st.divider()
st.caption("Built with Python, scikit-learn, XGBoost/LightGBM (stacked ensemble) & Streamlit")
