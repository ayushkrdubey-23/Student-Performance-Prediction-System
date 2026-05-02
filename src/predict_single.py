import joblib
import pandas as pd

# Load model
model = joblib.load("models/student_model.joblib")

# Example student data
new_student = pd.DataFrame([{
    "prior_gpa": 2.5,
    "attendance_pct": 60,
    "quiz_avg": 55,
    "assign_avg": 58,
    "midterm": 50,
    "study_hours_wk": 4,
    "on_time_submit_pct": 65,
    "lms_logins_wk": 3,
    "forum_posts": 1,
    "commute_min": 40,
    "gender": "Male",
    "school_type": "Govt",
    "parent_edu": "School"
## Another Data to predict
# "prior_gpa": 3.8,
# "attendance_pct": 90,
# "quiz_avg": 85,
# "assign_avg": 88,
# "midterm": 80,
# "study_hours_wk": 10,
# "on_time_submit_pct": 95,
# "lms_logins_wk": 8,
# "forum_posts": 5,
# "commute_min": 10,
}])

# Prediction
prob = model.predict_proba(new_student)[0][1]
pred = model.predict(new_student)[0]

with open("outputs/single_predictions.txt", "w") as f:
    f.write(f"Risk Probability: {prob}\n")
    f.write(f"Prediction: {'Pass' if pred == 1 else 'Fail'}\n")


# Output
print("Risk Probability:", prob)
print("Prediction:", "Pass" if pred == 1 else "Fail")

