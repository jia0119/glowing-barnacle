import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Page setup
st.set_page_config(
    page_title="Seoul Bike Demand - Multi-Model Predictor",
    page_icon="🚲",
    layout="wide",
)

FEATURE_NAMES = [
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


@st.cache_resource
def load_or_train_models():
    """Load pre-trained models from disk or create fallback trained models."""
    model_files = {
        "Random Forest": "model_rf.pkl",
        "Gradient Boosting": "model_gb.pkl",
        "Decision Tree": "model_dt.pkl",
        "Linear Regression": "model_lr.pkl",
    }

    models = {}
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                models[name] = pickle.load(f)

    # Train fallback models if local pickles are missing
    if not models:
        np.random.seed(42)
        X_dummy = pd.DataFrame(
            np.random.randn(200, len(FEATURE_NAMES)), columns=FEATURE_NAMES
        )
        # Synthetic demand relation for realistic demo behavior
        y_dummy = (
            np.maximum(
                0,
                (X_dummy["Temperature(°C)"] * 25)
                + (X_dummy["Hour"] * 40)
                - (X_dummy["Humidity(%)"] * 5)
                + 500,
            )
            + np.random.randint(0, 200, size=200)
        )

        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=50, random_state=42
            ).fit(X_dummy, y_dummy),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ).fit(X_dummy, y_dummy),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=8, random_state=42
            ).fit(X_dummy, y_dummy),
            "Linear Regression": LinearRegression().fit(X_dummy, y_dummy),
        }

    return models


models = load_or_train_models()

# App Header
st.title("🚲 Seoul Bike Rental Demand: Multi-Model Predictor")
st.markdown(
    "Compare bike demand predictions across multiple Machine Learning models based on real-time weather and temporal conditions."
)

# Sidebar Inputs
st.sidebar.header("⚙️ Model & Input Controls")

# Model Selection Control
selected_model_name = st.sidebar.selectbox(
    "Primary Model for Detail View", list(models.keys())
)

st.sidebar.divider()
st.sidebar.subheader("📅 Temporal & Operating Info")
hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 17)
seasons = st.sidebar.selectbox(
    "Season", ["Spring", "Summer", "Autumn", "Winter"]
)
functioning_day = st.sidebar.selectbox("Functioning Day", ["Yes", "No"])
holiday = st.sidebar.selectbox("Holiday Status", ["No Holiday", "Holiday"])

st.sidebar.divider()
st.sidebar.subheader("🌡️ Atmospheric Parameters")
temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 24.0, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, step=0.1)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1700, step=50)
dew_point = st.sidebar.slider(
    "Dew Point Temp (°C)", -35.0, 30.0, 11.0, step=0.5
)
solar_rad = st.sidebar.slider(
    "Solar Radiation (MJ/m²)", 0.0, 4.0, 1.5, step=0.05
)
rainfall = st.sidebar.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
)
snowfall = st.sidebar.number_input(
    "Snowfall (cm)", min_value=0.0, max_value=20.0, value=0.0, step=0.5
)

# Prepare Feature Matrix
raw_input = {
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

input_df = pd.DataFrame([raw_input]).reindex(
    columns=FEATURE_NAMES, fill_value=0
)

# Tabbed Interface
tab1, tab2 = st.tabs(
    ["📊 Model Comparison & Predictions", "📈 24-Hour Forecast by Model"]
)

# TAB 1: Single Input Predictions Across All Models
with tab1:
    st.subheader("Model Predictions for Current Inputs")

    if functioning_day == "No":
        st.warning(
            "⚠️ The bike system is marked as Non-Functioning. Demand across all models defaults to 0."
        )
    else:
        # Generate predictions for all models
        results = {}
        for name, model in models.items():
            pred = model.predict(input_df)[0]
            results[name] = max(0, int(round(pred)))

        # Highlight Primary Model Selection
        col_primary, col_stats = st.columns([1, 2])

        with col_primary:
            st.metric(
                label=f"Primary ({selected_model_name})",
                value=f"{results[selected_model_name]:,} bikes",
            )

        with col_stats:
            avg_pred = int(np.mean(list(results.values())))
            min_pred = min(results.values())
            max_pred = max(results.values())

            m1, m2, m3 = st.columns(3)
            m1.metric("Ensemble Average", f"{avg_pred:,} bikes")
            m2.metric("Lowest Model Estimate", f"{min_pred:,} bikes")
            m3.metric("Highest Model Estimate", f"{max_pred:,} bikes")

        st.divider()

        # Side-by-Side Comparison Table and Chart
        col_chart, col_table = st.columns([2, 1])

        results_df = pd.DataFrame(
            list(results.items()), columns=["Model", "Predicted Bike Count"]
        )

        with col_chart:
            st.subheader("Side-by-Side Model Comparison")
            st.bar_chart(
                results_df.set_index("Model"), color="#29b5e8"
            )

        with col_table:
            st.subheader("Summary Data")
            st.dataframe(
                results_df.style.highlight_max(
                    subset=["Predicted Bike Count"], color="#d4edda"
                ),
                use_container_width=True,
                hide_index=True,
            )

# TAB 2: 24-Hour Trend Comparison Across All Models
with tab2:
    st.subheader("24-Hour Demand Forecast Curves")
    st.caption(
        "Comparing hourly forecast trajectories across models under fixed weather parameters."
    )

    if functioning_day == "No":
        st.warning(
            "System non-functioning. Demand remains 0 across all hours."
        )
    else:
        hours_list = list(range(24))
        hourly_df_list = []

        for h in hours_list:
            row = raw_input.copy()
            row["Hour"] = h
            hourly_df_list.append(row)

        full_day_df = pd.DataFrame(hourly_df_list).reindex(
            columns=FEATURE_NAMES, fill_value=0
        )

        # Build combined 24-hour prediction dataframe
        trend_data = {"Hour": hours_list}
        for name, model in models.items():
            preds = model.predict(full_day_df)
            trend_data[name] = np.maximum(0, np.round(preds)).astype(int)

        trend_df = pd.DataFrame(trend_data).set_index("Hour")
        st.line_chart(trend_df)
