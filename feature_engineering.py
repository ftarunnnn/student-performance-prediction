import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def encode_family_support(df: pd.DataFrame, column: str = "family_support") -> pd.DataFrame:
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].map({"yes": 1, "no": 0})
    return df


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def create_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "internet_usage_hours" in df.columns and "study_hours" in df.columns:
        df["study_efficiency"] = df["study_hours"] / df["internet_usage_hours"].replace({0: 1})
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "study_hours",
        "attendance_pct",
        "previous_score",
        "assignments_completed",
        "sleep_hours",
        "internet_usage_hours",
        "family_support",
        "study_efficiency"
    ]
    return df[feature_cols]


def compute_feature_correlation(df: pd.DataFrame, target: str = "final_exam_score") -> pd.Series:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    corr = df.corr()[target].drop(labels=[target])
    return corr.abs().sort_values(ascending=False)


def compute_feature_importance(df: pd.DataFrame, target: str = "final_exam_score", random_state: int = 42) -> pd.Series:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestRegressor(random_state=random_state, n_estimators=100)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)
