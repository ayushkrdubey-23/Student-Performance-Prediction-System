import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load model
model = joblib.load("models/student_model.joblib")

# Load dataset
df = pd.read_csv("data/students.csv")

# Features only
X = df.drop("passed", axis=1)

# Predict
df["risk_probability"] = model.predict_proba(X)[:, 1]
df["prediction"] = model.predict(X)

# Convert prediction to label
df["prediction_label"] = df["prediction"].map({1: "Pass", 0: "Fail"})

# Save results
df.to_csv("outputs/predictions.csv", index=False)

# Print sample
print("Batch prediction completed!")
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# ---------------- GRAPH 1: Pass vs Fail ----------------
plt.figure(figsize=(6,4))
sns.countplot(x="prediction_label", data=df, palette="Set2")
plt.title("Student Performance Distribution", fontsize=14)
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/pass_fail_distribution.png")
plt.close()

# ---------------- GRAPH 2: Risk Distribution ----------------
plt.figure(figsize=(6,4))
sns.histplot(df["risk_probability"], bins=20, kde=True, color="blue")
plt.title("Risk Probability Distribution", fontsize=14)
plt.xlabel("Risk Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/risk_distribution.png")
plt.close()

# ---------------- GRAPH 3: Attendance vs Risk ----------------
plt.figure(figsize=(6,4))
sns.scatterplot(
    x="attendance_pct",
    y="risk_probability",
    hue="prediction_label",
    data=df,
    palette="coolwarm"
)
plt.title("Attendance vs Risk Analysis", fontsize=14)
plt.xlabel("Attendance %")
plt.ylabel("Risk Probability")
plt.tight_layout()
plt.savefig("images/attendance_vs_risk.png")
plt.close()

print("graphs saved in images/")