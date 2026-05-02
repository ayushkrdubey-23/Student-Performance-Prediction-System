#  Student Performance Prediction System

---

##  Project Overview

The **Student Performance Prediction System** is a Machine Learning-based project designed to predict whether a student will pass or fail based on academic and behavioral features such as attendance, study hours, quiz scores, and engagement metrics.

This project demonstrates a complete **end-to-end ML pipeline**, including data generation, preprocessing, model training, evaluation, deployment, and visualization.

---

##  Objective

* Identify at-risk students early
* Improve academic performance through data insights
* Enable data-driven decision-making in education systems

---

##  Acknowledgment

This project was developed under the guidance of **Mr. Umesh Yadav Sir** as part of the **IIP (Internship Program) by EDC in collaboration with IIT Delhi**.

---

##  Development Approach

This project follows a **prompt-driven development approach**, where structured prompts were used to:

* Design dataset schema and simulation
* Build machine learning pipeline
* Train and evaluate models
* Develop API for deployment
* Generate visual insights

This reflects modern AI-assisted development practices used in industry.

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Matplotlib, Seaborn
* Joblib

---

##  Project Structure

```
Student-Performance-Prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── images/
├── serving/
├── main.py
├── requirements.txt
└── README.md
```

---

##  Features

* Synthetic dataset generation (real-world simulation)
* Data preprocessing pipeline using ColumnTransformer
* Model training (Logistic Regression, Random Forest, XGBoost)
* Model evaluation using F1-score and classification report
* Batch prediction system (CSV output)
* REST API deployment using FastAPI
* Professional data visualization (Seaborn & Matplotlib)
* Output logging (reports, predictions, metrics)

---

##  Outputs

* `outputs/predictions.csv`
* `outputs/classification_report.txt`
* `outputs/accuracy.txt`
* `images/*.png` (visual graphs)

---

##  Visualizations

* Pass vs Fail Distribution
* Risk Probability Distribution
* Attendance vs Risk Analysis

All graphs are saved in the `images/` folder.

---

##  How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Generate dataset

```
python notebooks/01_generate_data.py
```

### 3. Train model

```
python src/train_xgboost.py
```

### 4. Run batch prediction + visualization

```
python src/predict.py
```

### 5. Run API

```
python -m uvicorn serving.app:app --reload
```

---

##  API Endpoints

* `/` → API status
* `/health` → health check
* `/predict` → single prediction
* `/predict-batch` → batch prediction

---

##  Example API Input

```json
{
  "prior_gpa": 3.5,
  "attendance_pct": 85,
  "quiz_avg": 80,
  "assign_avg": 78,
  "midterm": 75,
  "study_hours_wk": 8,
  "on_time_submit_pct": 90,
  "lms_logins_wk": 6,
  "forum_posts": 2,
  "commute_min": 20,
  "gender": "Male",
  "school_type": "Private",
  "parent_edu": "UG"
}
```

---

##  Key Highlights

* End-to-End Machine Learning Pipeline
* Synthetic Data Simulation
* Feature Engineering & Preprocessing
* Model Comparison & Optimization
* Batch & Real-time Prediction
* FastAPI Deployment
* Professional Visualization
* Industry-level Project Structure

---

##  Project Capabilities

* Predict student performance (Pass/Fail)
* Identify at-risk students using probability scores
* Perform batch predictions on large datasets
* Provide insights through visual analytics
* Serve predictions via REST API

---

##  Industry Relevance

This system is highly relevant in:

* EdTech platforms
* Universities and colleges
* Learning analytics systems
* Corporate training & development

It helps in:

* Early risk detection
* Personalized learning
* Dropout prevention

---

##  Future Enhancements

* Add frontend dashboard (React / Next.js)
* Deploy using Docker & Cloud platforms
* Add model monitoring & drift detection
* Integrate real-world datasets

---

##  Key Learnings

* End-to-end ML pipeline development
* Model deployment using FastAPI
* Feature engineering & preprocessing
* Data visualization for insights
* Batch vs real-time prediction systems

---

##  Author

**Ayush Kumar Dubey**
**(Student Btech-CSE)**
---
