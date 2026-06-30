import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    df = df.copy()
    if strategy == "mean":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "drop":
        df = df.dropna()
    else:
        raise ValueError("Unsupported missing value strategy: {}".format(strategy))
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def fix_data_types(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    df = df.copy()
    return df.astype(dtype_map)


def handle_outliers(df: pd.DataFrame, cols=None, method: str = "iqr") -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    if method == "iqr":
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    else:
        raise ValueError("Unsupported outlier handling method: {}".format(method))
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "family_support" in df.columns:
        df["family_support"] = df["family_support"].map({"yes": 1, "no": 0}).fillna(0)
    return df
