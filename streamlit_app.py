import streamlit as st
import pandas as pd
from src.predict import load_model, predict_final_score

st.title("Student Performance Predictor")
st.write("Enter student study and behavior data to predict the final exam score.")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=7.0)
attendance_pct = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
previous_score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, value=80.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=20, value=10)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
internet_usage_hours = st.number_input("Internet Usage Hours", min_value=0.0, max_value=24.0, value=4.0)
family_support = st.selectbox("Family Support", ["yes", "no"])

if st.button("Predict Score"):
    input_data = {
        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours,
        "internet_usage_hours": internet_usage_hours,
        "family_support": 1 if family_support == "yes" else 0
    }
    model = load_model()
    score = predict_final_score(input_data, model)
    st.success(f"Predicted Final Exam Score: {score:.2f}")
