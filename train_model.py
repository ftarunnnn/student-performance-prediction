import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_features


def train_and_save_model() -> None:
    df = load_data("data/student_performance.csv")
    df = preprocess_data(df)
    X = select_features(df)
    y = df["final_exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "models/student_model.joblib")

    print("Model trained and saved to models/student_model.joblib")
