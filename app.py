from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/student_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"predicted_final_exam_score": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
