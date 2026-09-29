import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# MOBILE APP SESSION LENGTH PREDICTION
# --------------------------------------------------

st.set_page_config(
    page_title="Mobile App Session Length Prediction",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Mobile App Session Length Prediction")
st.write("Predict the mobile app session duration using Machine Learning.")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

linear_model = joblib.load(
    "dataset/models/linear_regression.pkl"
)

random_forest_model = joblib.load(
    "dataset/models/random_forest_regressor.pkl"
)

gradient_boosting_model = joblib.load(
    "dataset/models/gradient_boosting_regressor.pkl"
)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.header("Enter Session Details")

battery_level = st.number_input(
    "Battery Level",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

memory_usage_mb = st.number_input(
    "Memory Usage (MB)",
    min_value=0.0,
    value=500.0
)

event_value = st.number_input(
    "Event Value",
    min_value=0.0,
    value=10.0
)

user_age = st.number_input(
    "User Age",
    min_value=1.0,
    max_value=100.0,
    value=25.0
)

device_os = st.selectbox(
    "Device OS",
    ["Android", "iOS"]
)

device_os_version = st.text_input(
    "Device OS Version",
    value="Android 13"
)

device_model = st.text_input(
    "Device Model",
    value="Google Pixel 6"
)

screen_resolution = st.text_input(
    "Screen Resolution",
    value="1080x1920"
)

location_country = st.text_input(
    "Location Country",
    value="India"
)

location_city = st.text_input(
    "Location City",
    value="Bengaluru"
)

app_language = st.text_input(
    "App Language",
    value="English"
)

network_type = st.selectbox(
    "Network Type",
    ["WiFi", "4G", "5G", "3G"]
)

event_type = st.text_input(
    "Event Type",
    value="click"
)

event_target = st.text_input(
    "Event Target",
    value="button"
)

app_version = st.text_input(
    "App Version",
    value="1.0.0"
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Session Length"):

    input_data = pd.DataFrame({
        "battery_level": [battery_level],
        "memory_usage_mb": [memory_usage_mb],
        "event_value": [event_value],
        "user_age": [user_age],

        "device_os": [device_os],
        "device_os_version": [device_os_version],
        "device_model": [device_model],
        "screen_resolution": [screen_resolution],
        "location_country": [location_country],
        "location_city": [location_city],
        "app_language": [app_language],
        "network_type": [network_type],
        "event_type": [event_type],
        "event_target": [event_target],
        "app_version": [app_version]
    })

    try:

        linear_prediction = linear_model.predict(input_data)[0]

        random_forest_prediction = (
            random_forest_model.predict(input_data)[0]
        )

        gradient_boosting_prediction = (
            gradient_boosting_model.predict(input_data)[0]
        )

        st.success("Prediction completed successfully! 🎉")

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Linear Regression",
                f"{max(0, linear_prediction):.2f} sec"
            )

        with col2:
            st.metric(
                "Random Forest",
                f"{max(0, random_forest_prediction):.2f} sec"
            )

        with col3:
            st.metric(
                "Gradient Boosting",
                f"{max(0, gradient_boosting_prediction):.2f} sec"
            )

        average_prediction = (
            linear_prediction
            + random_forest_prediction
            + gradient_boosting_prediction
        ) / 3

        st.subheader("Final Prediction")

        st.success(
            f"Estimated Session Duration: "
            f"{max(0, average_prediction):.2f} seconds"
        )

        st.info(
            f"Approximately "
            f"{max(0, average_prediction) / 60:.2f} minutes"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.error(str(e))