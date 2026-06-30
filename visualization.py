import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")


def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=15)
    plt.xlabel("Residual")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.show()


def plot_regression_line(X, y, model, feature):
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X")
    plt.figure(figsize=(8, 6))
    sns.regplot(x=X[feature], y=y, lowess=True, line_kws={"color": "red"})
    plt.xlabel(feature)
    plt.ylabel("Final Exam Score")
    plt.title(f"Regression Line: {feature} vs Final Exam Score")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importances, feature_names):
    if len(feature_importances) != len(feature_names):
        raise ValueError("Feature importances and feature names must match lengths")
    importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
