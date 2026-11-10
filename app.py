"""
app.py - Web interface for the crop yield prediction system.
Run with:  streamlit run app.py
Requires models/best_model.joblib produced by train_models.py
"""
import joblib
import pandas as pd
import streamlit as st

STATES = ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"]
CROPS = ["Cassava", "Yam", "Maize", "Cocoyam", "Plantain"]  # keep in step with training data

st.set_page_config(page_title="SW Nigeria Crop Yield Predictor", layout="centered")
st.title("Crop Yield Prediction - South-West, Nigeria")
st.caption("Machine learning-driven estimate of yield (kg/ha) and total production (tonnes)")


@st.cache_resource
def load_bundle():
    return joblib.load("models/best_model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("Model file not found. Run train_models.py first.")
    st.stop()

col1, col2 = st.columns(2)
state = col1.selectbox("State", STATES)
crop = col2.selectbox("Crop", CROPS)
year = st.number_input("Forecast year", min_value=2000, max_value=2100, value=2026, step=1)
area = st.number_input("Expected area to be harvested (ha)", min_value=1.0, value=10000.0, step=100.0)

st.subheader("Weather (growing-season totals / means)")
c1, c2, c3 = st.columns(3)
rain = c1.number_input("Rainfall (mm)", 0.0, 4000.0, 1300.0)
tmax = c2.number_input("Max temperature (deg C)", 15.0, 45.0, 31.0)
tmin = c3.number_input("Min temperature (deg C)", 10.0, 35.0, 22.0)
c4, c5 = st.columns(2)
hum = c4.number_input("Relative humidity (%)", 0.0, 100.0, 75.0)
sol = c5.number_input("Solar radiation (MJ/m2/day)", 0.0, 40.0, 18.0)

st.subheader("Recent history for this state and crop")
h1, h2 = st.columns(2)
y_lag1 = h1.number_input("Last season's yield (kg/ha)", 0.0, 100000.0, 10000.0)
y_roll3 = h2.number_input("3-season average yield (kg/ha)", 0.0, 100000.0, 10000.0)
r1, r2 = st.columns(2)
rain_lag1 = r1.number_input("Last season's rainfall (mm)", 0.0, 4000.0, 1300.0)
rain_roll3 = r2.number_input("3-season average rainfall (mm)", 0.0, 4000.0, 1300.0)

if st.button("Predict yield"):
    row = pd.DataFrame([{
        "state": state, "crop": crop, "year": year, "area_harvested_ha": area,
        "rainfall_mm": rain, "tmax_c": tmax, "tmin_c": tmin, "humidity_pct": hum,
        "solar_mj_m2": sol, "rain_lag1": rain_lag1, "rain_roll3": rain_roll3,
        "yield_lag1": y_lag1, "yield_roll3": y_roll3}])[bundle["features"]]
    y_hat = float(bundle["model"].predict(row)[0])
    st.success(f"Predicted yield: {y_hat:,.0f} kg/ha")
    st.metric("Estimated production", f"{y_hat * area / 1000:,.1f} tonnes")
    st.caption(f"Model used: {bundle['name']}. Estimates are indicative and should "
               "support, not replace, expert judgement.")
