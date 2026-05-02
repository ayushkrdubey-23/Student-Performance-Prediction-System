from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# Initialize app
app = FastAPI(title="Student Performance Prediction API")

# Load model (make sure path is correct)
MODEL_PATH = "models/student_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found! Please train model first.")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Input Schema
# -----------------------------
class Student(BaseModel):
    prior_gpa: float
    attendance_pct: float
    quiz_avg: float
    assign_avg: float
    midterm: float
    study_hours_wk: float
    on_time_submit_pct: float
    lms_logins_wk: float
    forum_posts: float
    commute_min: float
    gender: str
    school_type: str
    parent_edu: str

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running 🚀"}

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "OK"}

# -----------------------------
# Single Prediction
# -----------------------------
@app.post("/predict")
def predict(student: Student):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([student.model_dump()])

        # Prediction
        prob = float(model.predict_proba(data)[0][1])
        pred = int(model.predict(data)[0])

        return {
            "risk_probability": round(prob, 4),
            "prediction": "Pass" if pred == 1 else "Fail"
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Batch Prediction Endpoint (Optional 🔥)
# -----------------------------
@app.post("/predict-batch")
def predict_batch():
    try:
        df = pd.read_csv("data/students.csv")

        X = df.drop("passed", axis=1)

        df["risk_probability"] = model.predict_proba(X)[:, 1]
        df["prediction"] = model.predict(X)
        df["prediction_label"] = df["prediction"].map({1: "Pass", 0: "Fail"})

        # Save output
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/api_predictions.csv", index=False)

        return {
            "message": "Batch prediction completed",
            "file_saved": "outputs/api_predictions.csv"
        }

    except Exception as e:
        return {"error": str(e)}
    