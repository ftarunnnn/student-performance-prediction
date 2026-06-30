import joblib
import pandas as pd
import numpy as np

def run_tests():
    print("--- Running Model Tests ---")
    
    try:
        model = joblib.load("student_model.pkl")
    except Exception as e:
        print("Failed to load model:", e)
        return

    features_list = [
        "study_hours", "attendance_pct", "previous_score",
        "assignments_completed", "sleep_hours", "internet_usage_hours", "family_support"
    ]

    # Test 1: Normal Values
    print("\n[Test 1] Normal Values")
    data_normal = {
        "study_hours": [5.0], "attendance_pct": [80.0], "previous_score": [75.0],
        "assignments_completed": [10.0], "sleep_hours": [7.0], "internet_usage_hours": [2.0], "family_support": [1.0]
    }
    df_normal = pd.DataFrame(data_normal)
    try:
        pred = model.predict(df_normal)[0]
        print(f"Passed! Predicted Score: {pred:.2f}")
    except Exception as e:
        print(f"Failed: {e}")

    # Test 2: High Values (Should predict a very high score, possibly > 100)
    print("\n[Test 2] High Values")
    data_high = {
        "study_hours": [24.0], "attendance_pct": [100.0], "previous_score": [100.0],
        "assignments_completed": [50.0], "sleep_hours": [12.0], "internet_usage_hours": [0.0], "family_support": [3.0]
    }
    df_high = pd.DataFrame(data_high)
    try:
        pred = model.predict(df_high)[0]
        print(f"Passed! Predicted Score: {pred:.2f} (Notice it may exceed 100 since it's a linear model without bounds)")
    except Exception as e:
        print(f"Failed: {e}")

    # Test 3: Low Values (Should predict a very low score, possibly < 0)
    print("\n[Test 3] Low Values")
    data_low = {
        "study_hours": [0.0], "attendance_pct": [0.0], "previous_score": [0.0],
        "assignments_completed": [0.0], "sleep_hours": [0.0], "internet_usage_hours": [24.0], "family_support": [0.0]
    }
    df_low = pd.DataFrame(data_low)
    try:
        pred = model.predict(df_low)[0]
        print(f"Passed! Predicted Score: {pred:.2f} (Notice it may be less than 0)")
    except Exception as e:
        print(f"Failed: {e}")

    # Test 4: Missing Values (Should throw an error as sklearn LinearRegression doesn't support NaNs)
    print("\n[Test 4] Missing Values (NaN)")
    data_missing = {
        "study_hours": [np.nan], "attendance_pct": [80.0], "previous_score": [75.0],
        "assignments_completed": [10.0], "sleep_hours": [7.0], "internet_usage_hours": [2.0], "family_support": [1.0]
    }
    df_missing = pd.DataFrame(data_missing)
    try:
        pred = model.predict(df_missing)[0]
        print(f"Failed to catch error! Predicted: {pred}")
    except Exception as e:
        print(f"Passed! Caught expected missing value error: {type(e).__name__} - {e}")

    # Test 5: Invalid Input Type (String instead of float)
    print("\n[Test 5] Invalid Input Type")
    data_invalid = {
        "study_hours": ["five"], "attendance_pct": [80.0], "previous_score": [75.0],
        "assignments_completed": [10.0], "sleep_hours": [7.0], "internet_usage_hours": [2.0], "family_support": [1.0]
    }
    df_invalid = pd.DataFrame(data_invalid)
    try:
        pred = model.predict(df_invalid)[0]
        print(f"Failed to catch error! Predicted: {pred}")
    except Exception as e:
        print(f"Passed! Caught expected invalid input error: {type(e).__name__}")

if __name__ == "__main__":
    run_tests()
