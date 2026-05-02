from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Numerical features
NUM = [
    "prior_gpa",
    "attendance_pct",
    "quiz_avg",
    "assign_avg",
    "midterm",
    "study_hours_wk",
    "on_time_submit_pct",
    "lms_logins_wk",
    "forum_posts",
    "commute_min"
]

# Categorical features
CAT = [
    "gender",
    "school_type",
    "parent_edu"
]

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Final preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUM),
    ("cat", cat_pipeline, CAT)
])
