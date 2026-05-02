import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pipeline import preprocessor

# Create folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/students.csv")

X = df.drop("passed", axis=1)
y = df["passed"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost Model
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluation
report = classification_report(y_test, xgb_pred)
f1 = f1_score(y_test, xgb_pred)

# Print
print("=== XGBoost Model ===")
print(report)
print("F1 Score:", f1)

# Save outputs
with open("outputs/xgb_report.txt", "w") as f:
    f.write("=== XGBoost Model ===\n")
    f.write(report)
    f.write(f"\nF1 Score: {f1}\n")

# Save model
joblib.dump(xgb_model, "models/student_model.joblib")

print("\nXGBoost model saved in models/")

