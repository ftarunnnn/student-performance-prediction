import joblib
import pandas as pd

def main():
    print("--- Student Performance Predictor ---")
    
    # Load the model
    try:
        model = joblib.load("student_model.pkl")
    except FileNotFoundError:
        print("Error: Model file 'student_model.pkl' not found. Please train the model first.")
        return

    try:
        study_hours = float(input("Enter Study Hours: "))
        attendance = float(input("Enter Attendance (percentage): "))
        previous_score = float(input("Enter Previous Score: "))
        assignments = float(input("Enter Assignments Completed: "))
        sleep_hours = float(input("Enter Sleep Hours: "))
        internet_usage = float(input("Enter Internet Usage Hours (default 0 if unknown): ") or 0)
        family_support = float(input("Enter Family Support (default 0 if unknown): ") or 0)
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Create a DataFrame for prediction
    features = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_pct": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments,
        "sleep_hours": sleep_hours,
        "internet_usage_hours": internet_usage,
        "family_support": family_support
    }])

    # Predict
    predicted_score = model.predict(features)[0]
    
    print(f"\nPredicted Final Score: {predicted_score:.2f}")

if __name__ == "__main__":
    main()
