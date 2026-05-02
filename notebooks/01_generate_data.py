import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "prior_gpa": np.round(np.random.uniform(2.0, 4.0, n), 2),
    "attendance_pct": np.random.randint(50, 100, n),
    "quiz_avg": np.random.randint(40, 100, n),
    "assign_avg": np.random.randint(40, 100, n),
    "midterm": np.random.randint(30, 100, n),
    "study_hours_wk": np.random.randint(1, 15, n),
    "on_time_submit_pct": np.random.randint(50, 100, n),
    "lms_logins_wk": np.random.randint(1, 10, n),
    "forum_posts": np.random.randint(0, 10, n),
    "commute_min": np.random.randint(5, 60, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "school_type": np.random.choice(["Private", "Govt"], n),
    "parent_edu": np.random.choice(["School", "UG", "PG"], n)
})

# Create target (passed)
score = (
    data["prior_gpa"] * 10 +
    data["attendance_pct"] * 0.2 +
    data["quiz_avg"] * 0.2 +
    data["assign_avg"] * 0.2 +
    data["midterm"] * 0.2 +
    data["study_hours_wk"] * 1.5 +
    data["on_time_submit_pct"] * 0.1 +
    data["lms_logins_wk"] * 1
)

data["passed"] = (score > 120).astype(int)

# Save dataset
data.to_csv("data/students.csv", index=False)

print("Dataset created successfully!")
print(data.head())
