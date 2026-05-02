import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pipeline import preprocessor

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Load data
df = pd.read_csv("data/students.csv")

# Split features & target
X = df.drop("passed", axis=1)
y = df["passed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=200))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train models
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluation
log_report = classification_report(y_test, log_pred)
rf_report = classification_report(y_test, rf_pred)

log_f1 = f1_score(y_test, log_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Print to terminal
print("=== Logistic Regression ===")
print(log_report)
print("F1 Score:", log_f1)

print("\n=== Random Forest ===")
print(rf_report)
print("F1 Score:", rf_f1)

# Save to file
with open("outputs/classification_report.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(log_report)
    f.write(f"\nF1 Score: {log_f1}\n\n")

    f.write("=== Random Forest ===\n")
    f.write(rf_report)
    f.write(f"\nF1 Score: {rf_f1}\n")

# Save accuracy summary
with open("outputs/accuracy.txt", "w") as f:
    f.write(f"Logistic Regression F1: {log_f1}\n")
    f.write(f"Random Forest F1: {rf_f1}\n")

print("\n Results saved in outputs/ folder")

