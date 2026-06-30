import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Predictor")
st.write("Enter the student's details below to predict their final exam score.")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Please ensure 'student_model.pkl' exists. Details: {e}")
    st.stop()

# Input form
with st.form("prediction_form"):
    st.subheader("Student Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        
    with col2:
        assignments = st.number_input("Assignments Completed", min_value=0.0, value=10.0, step=1.0)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    
    # Optional features (required by model but maybe not requested explicitly in UI)
    st.markdown("---")
    with st.expander("Additional Features"):
        internet_usage = st.number_input("Internet Usage Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        family_support = st.number_input("Family Support (0=Low, 1=Medium, 2=High etc.)", min_value=0.0, value=1.0, step=1.0)

    # Predict button
    submit_button = st.form_submit_button(label="Predict Final Score")

if submit_button:
    # Create DataFrame for prediction
    features = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_pct": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments,
        "sleep_hours": sleep_hours,
        "internet_usage_hours": internet_usage,
        "family_support": family_support
    }])
    
    try:
        prediction = model.predict(features)[0]
        st.success(f"### Predicted Final Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
