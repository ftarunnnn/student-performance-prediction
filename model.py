import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style="whitegrid")

# Load dataset
url = "data/student_performance.csv"

data = pd.read_csv(url)

# Feature columns and target
features = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "internet_usage_hours",
    "family_support"
]

target = "final_exam_score"

X = data[features]
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Coefficients
coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
print("\nFeature Coefficients:")
print(coeff_df.sort_values(by="Coefficient", ascending=False))

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title("Actual vs Predicted Exam Scores")
plt.plot([0, 100], [0, 100], color="red", linestyle="--")
plt.tight_layout()
plt.savefig("student_performance_prediction.png")
# plt.show()

# Save the model
joblib.dump(model, "student_model.pkl")
