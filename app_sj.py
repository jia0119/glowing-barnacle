import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(
    page_title="Seoul Bike Analytics & Demand Forecaster",
    page_icon="🚲",
    layout="wide",
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
        # Generate lightweight placeholder model if pkl doesn't exist yet
        np.random.seed(42)
        X_dummy = pd.DataFrame(
            np.random.randn(100, len(feature_names)), columns=feature_names
        )
        y_dummy = np.random.randint(10, 1500, size=100)
        model = RandomForestRegressor(n_estimators=30, random_state=42)
        model.fit(X_dummy, y_dummy)
        return model, feature_names


model, feature_names = load_or_train_model()

# Header
st.title("🚲 Seoul Bike Rental Forecasting Hub")
st.markdown(
    "Interactive dashboard featuring **single-hour predictions**, **24-hour daily trend simulations**, **batch processing**, and **feature importance insights**."
)

# Sidebar Input Controls
st.sidebar.header("⚙️ Environment & Time Controls")

seasons = st.sidebar.selectbox(
    "Season", ["Spring", "Summer", "Autumn", "Winter"]
)
functioning_day = st.sidebar.selectbox("Functioning Day", ["Yes", "No"])
holiday = st.sidebar.selectbox("Holiday Status", ["No Holiday", "Holiday"])
hour = st.sidebar.slider("Hour of Day (Target)", 0, 23, 17)

st.sidebar.divider()
st.sidebar.header("🌡️ Weather Conditions")
temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 22.0, step=0.5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 45)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 1.8, step=0.1)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1800, step=50)
dew_point = st.sidebar.slider(
    "Dew Point Temp (°C)", -35.0, 30.0, 9.5, step=0.5
)
solar_rad = st.sidebar.slider(
    "Solar Radiation (MJ/m²)", 0.0, 4.0, 1.2, step=0.05
)
rainfall = st.sidebar.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
)
snowfall = st.sidebar.number_input(
    "Snowfall (cm)", min_value=0.0, max_value=20.0, value=0.0, step=0.5
)


# Utility to build encoded feature row
def build_feature_row(target_hour):
    return {
        "Hour": target_hour,
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


# Multi-Tab Layout
tab1, tab2, tab3 = st.tabs(
    ["🎯 Single Prediction & Trend", "📁 Batch CSV Prediction", "📊 Model Insights"]
)

# TAB 1: Single Prediction & 24h Trend Simulation
with tab1:
    col_pred, col_metrics = st.columns([1, 2])

    current_data = pd.DataFrame([build_feature_row(hour)]).reindex(
        columns=feature_names, fill_value=0
    )

    if functioning_day == "No":
        single_pred = 0
    else:
        single_pred = max(0, int(round(model.predict(current_data)[0])))

    with col_pred:
        st.subheader("Target Hour Result")
        st.metric(
            label=f"Predicted Demand at {hour}:00",
            value=f"{single_pred:,} bikes",
        )

        # Weather Warnings / Indicators
        if rainfall > 0:
            st.warning(f"🌧️ Rain Alert: {rainfall} mm precipitation")
        if snowfall > 0:
            st.info(f"❄️ Snow Alert: {snowfall} cm accumulation")
        if temp > 32:
            st.error("🔥 Extreme Heat Warning")
        elif temp < -5:
            st.error("🥶 Freezing Condition Warning")

    with col_metrics:
        st.subheader("24-Hour Daily Simulation")
        st.caption(
            "Simulating rental demand across all hours under the current weather parameters."
        )

        # Generate 24-hour curve
        daily_hours = list(range(24))
        daily_records = [build_feature_row(h) for h in daily_hours]
        daily_df = pd.DataFrame(daily_records).reindex(
            columns=feature_names, fill_value=0
        )

        if functioning_day == "No":
            daily_preds = [0] * 24
        else:
            daily_preds = np.maximum(
                0, np.round(model.predict(daily_df))
            ).astype(int)

        chart_data = pd.DataFrame(
            {"Hour": daily_hours, "Predicted Bikes": daily_preds}
        ).set_index("Hour")
        st.line_chart(chart_data)

# TAB 2: Batch CSV Prediction
with tab2:
    st.subheader("Batch Prediction via CSV File")
    st.markdown(
        "Upload a `.csv` file containing weather features to generate bulk predictions."
    )

    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

    if uploaded_file is not None:
        try:
            user_csv = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", user_csv.head(3))

            # Encode and realign missing columns
            processed_csv = user_csv.reindex(
                columns=feature_names, fill_value=0
            )
            raw_predictions = model.predict(processed_csv)
            user_csv["Predicted_Bike_Count"] = np.maximum(
                0, np.round(raw_predictions)
            ).astype(int)

            st.success("Batch prediction complete!")
            st.write(
                user_csv[["Hour", "Temperature(°C)", "Predicted_Bike_Count"]]
            )

            # File Download Button
            csv_export = user_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_export,
                file_name="seoul_bike_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(
                f"Error processing file. Ensure feature names match expected schema. Details: {e}"
            )
    else:
        # Sample Download Template
        sample_df = pd.DataFrame(
            [build_feature_row(8), build_feature_row(18)]
        ).reindex(columns=feature_names, fill_value=0)
        sample_csv = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_csv,
            file_name="sample_bike_input.csv",
            mime="text/csv",
        )

# TAB 3: Model Insights
with tab3:
    st.subheader("Feature Importance Analysis")
    st.markdown(
        "Importance score of each feature used by the underlying decision trees."
    )

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=True)

        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("The loaded model does not expose feature importances.")
