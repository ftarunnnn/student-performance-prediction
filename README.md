# Student Performance Prediction

## Project Overview
This project builds a machine learning model to predict a student's final exam score based on several features, including study hours, attendance, previous grades, sleep hours, assignments completed, internet usage, and family support. The goal is to follow a complete end-to-end machine learning workflow, from data collection to deploying an interactive web application.

## Features
- **Data Preprocessing & EDA:** Cleans missing values, handles outliers, and explores data through visualizations.
- **Machine Learning Model:** Uses a Linear Regression model to predict scores.
- **Interactive Dashboard:** Includes a Streamlit web app allowing users to input student details and get instant predictions.
- **Model Evaluation:** Provides metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score to measure performance.
- **Testing Script:** A dedicated script to test the model with edge cases (missing values, high/low inputs).

## Dataset
The dataset used contains the following columns:
- `study_hours` (numerical)
- `attendance_pct` (numerical)
- `previous_score` (numerical)
- `assignments_completed` (numerical)
- `sleep_hours` (numerical)
- `internet_usage_hours` (numerical)
- `family_support` (categorical/numerical)
- `final_exam_score` (numerical, Target)

## Installation

1. Clone the repository or download the project files.
2. Install the required dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. (Optional) Run the model training script if you want to retrain the model:
   ```powershell
   python model.py
   ```
4. Run the Streamlit Dashboard:
   ```powershell
   streamlit run dashboard.py
   ```

## Project Structure
```text
Student-Performance-Prediction/
├── data/
│   └── student_performance.csv
├── models/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
├── model.py                        # Script for training and saving the model
├── student_model.pkl               # Saved trained model
├── dashboard.py                    # Streamlit web interface
├── predict.py                      # CLI prediction script
├── test_predictions.py             # Script testing model edge cases
└── student_performance_prediction.png # Actual vs Predicted scatter plot
```

## Model Used
- **Algorithm:** Linear Regression
- **Library:** Scikit-learn
- **Inputs:** Study Hours, Attendance (%), Previous Score, Assignments Completed, Sleep Hours, Internet Usage Hours, Family Support.
- **Output:** Predicted Final Exam Score.

## Results
Based on the current trained model:
- **Mean Absolute Error (MAE):** ~4.04
- **Root Mean Squared Error (RMSE):** ~4.34
- **R² Score:** ~0.91

*(Metrics might vary slightly based on data splits or updates.)*

## Screenshots
*(Add your screenshots here! Example:)*
- `student_performance_prediction.png` - Visualizes Actual vs Predicted Final Exam Scores.

## Future Improvements
- **Advanced Models:** Try Random Forest or Gradient Boosting models for potentially better accuracy.
- **More Features:** Include more diverse data points (e.g., commute time, extracurriculars).
- **Deployment:** Host the Streamlit app online using Streamlit Community Cloud, Render, or Heroku.

## Final Project Workflow

```text
Problem Definition
        │
        ▼
Collect Dataset
        │
        ▼
Load Dataset
        │
        ▼
Explore Data (EDA)
        │
        ▼
Clean Data
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Train Linear Regression Model
        │
        ▼
Predict Scores
        │
        ▼
Evaluate Model
        │
        ▼
Save Model
        │
        ▼
Build Web App
        │
        ▼
Deploy Project
```

## Technologies Used

| Component | Technology |
| --- | --- |
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model | Linear Regression |
| Model Saving | Joblib or Pickle |
| Web App (Optional) | Flask or Streamlit |
| IDE | VS Code or Jupyter Notebook |
| Version Control | Git & GitHub |

This roadmap is suitable for a beginner and follows a complete end-to-end machine learning workflow, from collecting data to deploying a regression model that predicts student performance.
