# 🏠 House Price Prediction — Multi-Source Data Fusion Approach

A machine learning system that predicts house prices using property details **combined with** location, amenity, and accessibility data — instead of relying on basic attributes alone. Built with Python, scikit-learn, and Streamlit.

## Problem

Existing house price prediction systems and Indian real estate apps (MagicBricks, 99acres, Housing.com, etc.) suffer from:
- Low accuracy due to limited features (only size/location/rooms)
- Black-box predictions with no explanation
- Inconsistent, listing-based (not transaction-based) pricing
- No multi-source data fusion (amenities, transit access, etc. ignored)

## Solution

This project builds a **lightweight, explainable, multi-source ML pipeline**:
1. Cleans and preprocesses raw property data
2. Engineers accessibility/amenity-based features
3. Trains and compares multiple models (Linear Regression, Random Forest, Gradient Boosting)
4. Selects the best-performing model automatically
5. Serves predictions through an interactive Streamlit web app with a **price range**, not just a single number

## Project Structure

```
house-price-prediction/
├── data/                       # raw and cleaned datasets
├── notebooks/                  # EDA and experimentation
├── src/
│   ├── generate_sample_data.py # creates sample dataset (swap with real data)
│   ├── data_preprocessing.py   # cleaning, dedup, outlier removal
│   ├── feature_engineering.py  # derived features + encoding
│   ├── train_model.py          # trains & compares models, saves best
│   └── predict.py              # reusable prediction function
├── app/
│   └── streamlit_app.py        # the website (Streamlit UI)
├── models/                     # saved model, encoders, feature importance
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data (skip this if you add your own dataset to /data)
python src/generate_sample_data.py

# 4. Clean the data
python src/data_preprocessing.py

# 5. Train the model
python src/train_model.py

# 6. Launch the website
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Using Your Own Dataset

Replace `data/raw_data.csv` with your real dataset. Required columns:

| Column | Description |
|---|---|
| City, Locality | Location |
| Area_sqft | Built-up area |
| BHK | Number of bedrooms |
| Age_years | Property age |
| Dist_to_Metro_km, Dist_to_School_km, Dist_to_Hospital_km | Accessibility |
| Num_Amenities | Count of amenities |
| Furnishing | Unfurnished / Semi-Furnished / Furnished |
| Parking | Number of parking spots |
| Price | Target variable |

## Model Performance (on sample data)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~5.9M | ~4.5M | 0.47 |
| **Random Forest** | **~0.51M** | **~0.39M** | **0.996** |
| Gradient Boosting | ~0.77M | ~0.54M | 0.991 |

*(Results will vary with a real dataset — this sample data is synthetically generated for pipeline testing.)*

## Deployment

Deploy the Streamlit app for free on [Streamlit Community Cloud](https://streamlit.io/cloud), Render, or Hugging Face Spaces, and link the live URL here.

## Tech Stack

Python · pandas · scikit-learn · Streamlit · matplotlib/seaborn · joblib

## License

MIT
