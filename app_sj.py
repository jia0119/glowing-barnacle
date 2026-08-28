import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Seoul Bike Rental Analytics Hub",
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

    # Train synthetic fallback models if local pkl files are absent
    if not models:
        np.random.seed(42)
        X_dummy = pd.DataFrame(
            np.random.randn(200, len(FEATURE_NAMES)), columns=FEATURE_NAMES
        )
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

# Dashboard Title
st.title("🚲 Seoul Bike Rental Analytics Hub")
st.markdown(
    "All-in-one forecasting tool featuring **Multi-Model Comparisons**, **24-Hour Simulation**, **Batch Processing**, and **Feature Importance**."
)

# Sidebar Inputs
st.sidebar.header("⚙️ Configuration & Inputs")

selected_model_name = st.sidebar.selectbox(
    "Primary Model Selection", list(models.keys())
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
st.sidebar.subheader("🌡️ Weather Conditions")
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


def build_feature_dict(target_hour):
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


current_input_df = pd.DataFrame([build_feature_dict(hour)]).reindex(
    columns=FEATURE_NAMES, fill_value=0
)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Multi-Model Comparison",
        "📈 24-Hour Forecast Curve",
        "📁 Batch Prediction (CSV)",
        "🔍 Feature Importance",
    ]
)

# TAB 1: Single Input Multi-Model Comparison
with tab1:
    st.subheader("Single-Hour Predictions Across Models")

    if functioning_day == "No":
        st.warning(
            "⚠️ The system is marked as Non-Functioning. Demand defaults to 0 across all models."
        )
    else:
        results = {}
        for name, model in models.items():
            pred = model.predict(current_input_df)[0]
            results[name] = max(0, int(round(pred)))

        col_primary, col_ensemble = st.columns([1, 2])

        with col_primary:
            st.metric(
                label=f"Primary ({selected_model_name})",
                value=f"{results[selected_model_name]:,} bikes",
            )

        with col_ensemble:
            avg_val = int(np.mean(list(results.values())))
            min_val = min(results.values())
            max_val = max(results.values())

            m1, m2, m3 = st.columns(3)
            m1.metric("Ensemble Average", f"{avg_val:,} bikes")
            m2.metric("Minimum Prediction", f"{min_val:,} bikes")
            m3.metric("Maximum Prediction", f"{max_val:,} bikes")

        # Weather Alerts
        if rainfall > 0 or snowfall > 0 or temp > 32 or temp < -5:
            st.markdown("**Weather Impact Alerts:**")
            if rainfall > 0:
                st.warning(f"🌧️ Rain Alert: {rainfall} mm")
            if snowfall > 0:
                st.info(f"❄️ Snow Alert: {snowfall} cm")
            if temp > 32:
                st.error("🔥 Extreme Heat Warning")
            elif temp < -5:
                st.error("🥶 Freezing Condition Warning")

        st.divider()

        col_chart, col_table = st.columns([2, 1])

        results_df = pd.DataFrame(
            list(results.items()), columns=["Model", "Predicted Bike Count"]
        )

        with col_chart:
            st.subheader("Model Comparison Bar Chart")
            st.bar_chart(
                results_df.set_index("Model"), color="#29b5e8"
            )

        with col_table:
            st.subheader("Detailed Data")
            st.dataframe(
                results_df.style.highlight_max(
                    subset=["Predicted Bike Count"], color="#d4edda"
                ),
                use_container_width=True,
                hide_index=True,
            )

# TAB 2: 24-Hour Multi-Model Forecast Curves
with tab2:
    st.subheader("24-Hour Demand Forecast Comparison")
    st.caption(
        "Simulates hourly rental demand curves for each model under current weather conditions."
    )

    if functioning_day == "No":
        st.warning(
            "System is Non-Functioning. Demand remains 0 across all 24 hours."
        )
    else:
        hours_list = list(range(24))
        daily_records = [build_feature_dict(h) for h in hours_list]
        full_day_df = pd.DataFrame(daily_records).reindex(
            columns=FEATURE_NAMES, fill_value=0
        )

        trend_data = {"Hour": hours_list}
        for name, model in models.items():
            preds = model.predict(full_day_df)
            trend_data[name] = np.maximum(0, np.round(preds)).astype(int)

        trend_df = pd.DataFrame(trend_data).set_index("Hour")
        st.line_chart(trend_df)

# TAB 3: Batch CSV Prediction
with tab3:
    st.subheader("Batch Prediction via CSV File")
    st.markdown(
        "Upload a `.csv` file containing weather and temporal variables to generate batch predictions."
    )

    batch_model_choice = st.selectbox(
        "Select Model for Batch Scoring", list(models.keys())
    )
    uploaded_file = st.file_uploader("Upload Input CSV Data", type=["csv"])

    if uploaded_file is not None:
        try:
            user_csv = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data Preview:", user_csv.head(3))

            processed_csv = user_csv.reindex(
                columns=FEATURE_NAMES, fill_value=0
            )

            # Generate batch predictions using chosen model
            selected_model = models[batch_model_choice]
            raw_preds = selected_model.predict(processed_csv)
            user_csv[f"Predicted_Bikes_{batch_model_choice}"] = np.maximum(
                0, np.round(raw_preds)
            ).astype(int)

            st.success(
                f"Successfully generated predictions using {batch_model_choice}!"
            )
            st.dataframe(user_csv.head(10))

            csv_data = user_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv_data,
                file_name="seoul_bike_batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as err:
            st.error(
                f"Failed to process CSV file. Ensure standard column naming. Details: {err}"
            )
    else:
        # Generate sample template
        sample_df = pd.DataFrame(
            [build_feature_dict(8), build_feature_dict(18)]
        ).reindex(columns=FEATURE_NAMES, fill_value=0)
        sample_csv = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_csv,
            file_name="sample_bike_input.csv",
            mime="text/csv",
        )

# TAB 4: Feature Importance
with tab4:
    st.subheader("Model Feature Importance Analysis")

    importance_model_choice = st.selectbox(
        "Select Model to Inspect",
        [m for m in models.keys() if m != "Linear Regression"],
    )
    target_model = models[importance_model_choice]

    if hasattr(target_model, "feature_importances_"):
        fi_df = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Importance Score": target_model.feature_importances_,
            }
        ).sort_values(by="Importance Score", ascending=True)

        st.bar_chart(fi_df.set_index("Feature"))
    else:
        st.info("Selected model does not support feature importance metrics.")
