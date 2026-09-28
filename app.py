import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

#=========================
# Title of the Streamlit app
#=========================
st.title("Phone Addiction Prediction")

#=========================
# User Input Form
#=========================
st.write("Enter your smartphone usage and lifestyle details to predict addiction.")
st.header("User Details")
age = st.number_input("Age", min_value=10, max_value=100, value=20)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

daily_screen_time_hours = st.number_input(
    "Daily Screen Time (hours)",
    min_value=0.0,
    max_value=24.0,
    value=4.0
)

social_media_hours = st.number_input(
    "Social Media Hours",
    min_value=0.0,
    max_value=24.0,
    value=2.0
)

gaming_hours = st.number_input(
    "Gaming Hours",
    min_value=0.0,
    max_value=24.0,
    value=1.0
)

work_study_hours = st.number_input(
    "Work/Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

notifications_per_day = st.number_input(
    "Notifications Per Day",
    min_value=0,
    max_value=1000,
    value=100
)

app_opens_per_day = st.number_input(
    "App Opens Per Day",
    min_value=0,
    max_value=1000,
    value=50
)

weekend_screen_time = st.number_input(
    "Weekend Screen Time (hours)",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

stress_level = st.selectbox(
    "Stress Level",
    ["Low", "Medium", "High"]
)

academic_work_impact = st.selectbox(
    "Academic Work Impact",
    ["Yes", "No"]
)

#=========================
# Prediction Button
#==========================
st.header("Prediction")
if st.button("Predict Addiction"):
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "daily_screen_time_hours": [daily_screen_time_hours],
        "social_media_hours": [social_media_hours],
        "gaming_hours": [gaming_hours],
        "work_study_hours": [work_study_hours],
        "sleep_hours": [sleep_hours],
        "notifications_per_day": [notifications_per_day],
        "app_opens_per_day": [app_opens_per_day],
        "weekend_screen_time": [weekend_screen_time],
        "stress_level": [stress_level],
        "academic_work_impact": [academic_work_impact]
    })

    st.write(input_data)

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)

    if prediction[0] == 1:
        st.success("Predicted Result: Addicted")
    else:
        st.info("Predicted Result: Not Addicted")