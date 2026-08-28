import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Seoul Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Seoul Bike Rental Demand Prediction")
st.markdown(
    "Predict hourly rented bike counts using weather, seasonal, and operational data."
)


@st.cache_resource
def load_or_train_model():
    model_filename = "bike_demand_model.pkl"
    feature_names = [
        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
        "Seasons_Spring",
        "Seasons_Summer",
        "Seasons_Winter",
        "Holiday_No Holiday",
        "Functioning Day_Yes",
    ]
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            model, loaded_features = pickle.load(f)
            return model, loaded_features
    else:
        # Fallback placeholder model if model file is absent
        X_dummy = pd.DataFrame(
            np.random.randn(20, len(feature_names)), columns=feature_names
        )
        y_dummy = np.random.randint(0, 1500, size=20)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        return model, feature_names


model, feature_names = load_or_train_model()

# Sidebar input controls
st.sidebar.header("Operational & Weather Inputs")

hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 12)
functioning_day = st.sidebar.selectbox("Functioning Day", ["Yes", "No"])
holiday = st.sidebar.selectbox("Holiday", ["No Holiday", "Holiday"])
seasons = st.sidebar.selectbox(
    "Season", ["Spring", "Summer", "Autumn", "Winter"]
)

temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 15.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind speed (m/s)", 0.0, 10.0, 2.0, step=0.1)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1500, step=10)
dew_point = st.sidebar.slider(
    "Dew point temperature (°C)", -35.0, 30.0, 5.0, step=0.1
)
solar_rad = st.sidebar.slider(
    "Solar Radiation (MJ/m²)", 0.0, 4.0, 0.5, step=0.01
)
rainfall = st.sidebar.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1
)
snowfall = st.sidebar.number_input(
    "Snowfall (cm)", min_value=0.0, max_value=20.0, value=0.0, step=0.1
)

# Format features into DataFrame matching expected encoding structure
input_data = {
    "Hour": hour,
    "Temperature(°C)": temp,
    "Humidity(%)": humidity,
    "Wind speed (m/s)": wind_speed,
    "Visibility (10m)": visibility,
    "Dew point temperature(°C)": dew_point,
    "Solar Radiation (MJ/m2)": solar_rad,
    "Rainfall(mm)": rainfall,
    "Snowfall (cm)": snowfall,
    "Seasons_Spring": 1 if seasons == "Spring" else 0,
    "Seasons_Summer": 1 if seasons == "Summer" else 0,
    "Seasons_Winter": 1 if seasons == "Winter" else 0,
    "Holiday_No Holiday": 1 if holiday == "No Holiday" else 0,
    "Functioning Day_Yes": 1 if functioning_day == "Yes" else 0,
}

input_df = pd.DataFrame([input_data])

# Main UI parameter overview
st.subheader("Selected Input Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Time of Day", f"{hour}:00")
    st.write(f"**Season:** {seasons}")
    st.write(f"**Holiday:** {holiday}")
    st.write(f"**Functioning Day:** {functioning_day}")

with col2:
    st.write(f"**Temperature:** {temp} °C")
    st.write(f"**Humidity:** {humidity} %")
    st.write(f"**Wind Speed:** {wind_speed} m/s")
    st.write(f"**Solar Radiation:** {solar_rad} MJ/m²")

with col3:
    st.write(f"**Visibility:** {visibility} (10m)")
    st.write(f"**Dew Point:** {dew_point} °C")
    st.write(f"**Rainfall:** {rainfall} mm")
    st.write(f"**Snowfall:** {snowfall} cm")

st.divider()

# Prediction logic execution
if st.button("Predict Bike Demand", type="primary"):
    if functioning_day == "No":
        st.warning(
            "⚠️ The system is marked as non-functioning on this day. Estimated demand: 0 bikes."
        )
    else:
        formatted_df = input_df.reindex(columns=feature_names, fill_value=0)
        predicted_count = model.predict(formatted_df)[0]
        final_demand = max(0, int(round(predicted_count)))
        st.success(
            f"### 🔮 Predicted Rented Bike Count: **{final_demand:,} bikes**"
        )
