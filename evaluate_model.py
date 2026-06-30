import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_features


def evaluate_model(model_path: str = "models/student_model.joblib") -> None:
    df = load_data("data/student_performance.csv")
    df = preprocess_data(df)
    X = select_features(df)
    y = df["final_exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print("Model evaluation")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"R^2: {r2_score(y_test, y_pred):.2f}")
