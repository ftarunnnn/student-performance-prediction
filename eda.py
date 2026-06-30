import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def eda_summary(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nStatistical summary:")
    print(df.describe(include="all"))


def plot_histograms(df: pd.DataFrame, cols=None) -> None:
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[cols].hist(figsize=(12, 8), bins=10)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, cols=None) -> None:
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    plt.figure(figsize=(12, len(cols) * 3))
    for i, col in enumerate(cols, start=1):
        plt.subplot(len(cols), 1, i)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_pairplot(df: pd.DataFrame, cols=None) -> None:
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
    sns.pairplot(df[cols])
    plt.show()


def plot_countplot(df: pd.DataFrame, col: str) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.show()
