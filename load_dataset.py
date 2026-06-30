import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Read the dataset from CSV and return a pandas DataFrame."""
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame) -> None:
    """Print basic dataset structure information."""
    print("First rows:\n", df.head(), "\n")
    print("Shape:", df.shape, "\n")
    print("Info:\n")
    print(df.info())
