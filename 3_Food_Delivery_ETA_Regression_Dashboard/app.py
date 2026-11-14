import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Food Delivery ETA Predictor", layout="wide")
st.title("End-to-End Food Delivery Time Prediction Dashboard")
st.caption("Developed by Vishal Pandey | Supervised Regression (Scikit-Learn) & Streamlit")

@st.cache_data
def train_eta_model():
    np.random.seed(101)
    n = 400
    distance_km = np.round(np.random.uniform(1.0, 18.0, n), 1)
    prep_time_min = np.random.randint(8, 30, n)
    traffic_level = np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.4, 0.25, 0.1]) # 1: Low, 4: Jam
    weather_mult = np.random.choice([1.0, 1.25, 1.4], n, p=[0.7, 0.2, 0.1]) # Clear, Rainy, Stormy
    rider_rating = np.round(np.random.uniform(3.8, 5.0, n), 1)
    
    eta = prep_time_min + (distance_km * 2.8 * weather_mult * (0.8 + 0.25 * traffic_level)) - ((rider_rating - 4.0) * 2.5) + np.random.normal(0, 1.8, n)
    eta = np.round(np.clip(eta, 12, 95), 1)
    
    X = pd.DataFrame({
        "Distance_km": distance_km,
        "Prep_Time_min": prep_time_min,
        "Traffic_Level": traffic_level,
        "Weather_Multiplier": weather_mult,
        "Rider_Rating": rider_rating
    })
    y = eta
    model = RandomForestRegressor(n_estimators=120, random_state=101)
    model.fit(X, y)
    preds = model.predict(X)
    return model, mean_absolute_error(y, preds), r2_score(y, preds), X

model, mae, r2, X_df = train_eta_model()

st.markdown(f"**Model Metrics:** R² Score = `{r2:.3f}` | Mean Absolute Error (MAE) = `{mae:.2f} mins`")

col1, col2 = st.columns(2)
with col1:
    dist = st.slider("Delivery Distance (km)", 1.0, 20.0, 5.5, 0.5)
    prep = st.slider("Restaurant Order Preparation Time (mins)", 5, 35, 15)
    rating = st.slider("Delivery Partner Rating", 3.5, 5.0, 4.7, 0.1)
with col2:
    traffic_label = st.selectbox("Real-Time Traffic Density", ["Low (1)", "Medium (2)", "High (3)", "Traffic Jam (4)"], index=1)
    weather_label = st.selectbox("Weather Condition", ["Clear / Sunny (1.0x)", "Rainy (1.25x)", "Heavy Rain / Fog (1.4x)"], index=0)

traffic_map = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3, "Traffic Jam (4)": 4}
weather_map = {"Clear / Sunny (1.0x)": 1.0, "Rainy (1.25x)": 1.25, "Heavy Rain / Fog (1.4x)": 1.4}

input_df = pd.DataFrame([[dist, prep, traffic_map[traffic_label], weather_map[weather_label], rating]],
                        columns=["Distance_km", "Prep_Time_min", "Traffic_Level", "Weather_Multiplier", "Rider_Rating"])
pred_eta = model.predict(input_df)[0]

st.metric("Predicted Delivery ETA", f"{pred_eta:.0f} Minutes")
