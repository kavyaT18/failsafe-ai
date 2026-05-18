# FAILSAFE

## AI-Powered Early Student Risk Detection and Intervention System

FAILSAFE is an AI-driven academic analytics platform designed to help educational institutions identify students who are at risk of academic failure before final semester results.

The system combines machine learning, explainable AI, and large language models to provide not only risk predictions, but also transparent explanations and personalized intervention recommendations for faculty members.

Instead of relying only on final grades, FAILSAFE analyzes academic performance, attendance, behavioral patterns, support systems, and study habits to detect warning signs early and help institutions take proactive action.

---

# Problem Statement

In many educational institutions, student failure is detected too late — often only after end-semester evaluations. Faculty members typically lack intelligent tools that can:

- identify at-risk students early,
- explain why a student is struggling,
- suggest actionable intervention strategies,
- monitor academic risk trends over time.

FAILSAFE addresses this problem using an explainable AI-based risk prediction system.

---

# Key Features

## Early Risk Prediction

Predicts student failure risk using:

- grades,
- attendance,
- study time,
- previous failures,
- educational support indicators,
- behavioral and socio-academic features.

---

## Explainable AI using SHAP

The system explains each prediction transparently.

Example explanations:

- Low G2 increased risk
- High absences increased risk
- Educational support reduced risk

This makes the model understandable and actionable for non-technical faculty members.

---

## AI-Generated Intervention Plans

FAILSAFE uses Groq-hosted LLMs to generate personalized intervention recommendations such as:

- extra academic mentoring,
- counseling referrals,
- attendance monitoring,
- structured study plans,
- regular faculty follow-ups.

---

## Feature Engineering

Custom engineered features include:

- grade_drop
- failure_x_nosupport
- support_deficit
- parent_edu_index
- disadvantage_score

These features improve the model’s ability to capture hidden academic risk patterns.

---

# Tech Stack

## Machine Learning

- Python
- Scikit-learn
- XGBoost
- SHAP
- Pandas
- NumPy
- Imbalanced-learn (SMOTE)

---

## Backend

- FastAPI
- Uvicorn
- Python-dotenv

---

## Frontend

- HTML
- CSS
- JavaScript

---

## LLM Integration

- Groq API
- Llama 3.3 70B Versatile

---

# Dataset

The project uses the UCI Student Performance Dataset.

Dataset files:

- student-mat.csv
- student-por.csv

The dataset contains:

- academic records,
- attendance information,
- support indicators,
- family background,
- behavioral patterns,
- demographic features.

---

# Machine Learning Pipeline

## Data Preprocessing

The preprocessing pipeline includes:

- one-hot encoding,
- standard scaling,
- log transformation for skewed features,
- feature engineering,
- imbalance handling using SMOTE.

---

## Models Evaluated

Several machine learning models were tested:

- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- XGBoost

---

## Final Model

The final deployed model uses:

- XGBoost
- SMOTE
- SHAP Explainability

Hyperparameter tuning was performed using:

- RandomizedSearchCV
- StratifiedKFold Cross Validation

---

# Project Structure

```bash
failsafe/
│
├── backend/
│   ├── main.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── .env
│   ├── failsafe_model3.pkl
│   └── venv/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── .gitignore
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/failsafe-ai.git

cd failsafe-ai
```

---

# Backend Setup

## 2. Navigate to Backend

```bash
cd backend
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 5. Configure Environment Variables

Create a `.env` file inside the backend folder.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 6. Add Trained Model

Place the trained model file:

```bash
failsafe_model3.pkl
```

inside the backend directory.

---

## 7. Run Backend Server

```bash
uvicorn main:app --reload --port 8000
```

Backend runs at:

```bash
http://localhost:8000
```

Swagger API documentation:

```bash
http://localhost:8000/docs
```

---

# Frontend Setup

## 8. Open New Terminal

Navigate to frontend directory:

```bash
cd frontend
```

---

## 9. Run Frontend Server

```bash
python -m http.server 5500
```

Frontend runs at:

```bash
http://localhost:5500
```

---




# API Endpoint

## POST /predict

Predicts student academic risk.

### Example Request

```json
{
  "G1": 10,
  "G2": 8,
  "absences": 12,
  "failures": 2,
  "studytime": 2
}
```

---

### Example Response

```json
{
  "risk_probability": 0.82,
  "risk_category": "High Risk",
  "reasons": [
    "Low G2 increased risk",
    "High absences increased risk"
  ],
  "intervention": "Schedule weekly mentoring sessions and monitor attendance closely."
}
```

---

# Future Improvements

Possible future extensions include:

- faculty authentication system,
- bulk CSV prediction,
- analytics dashboard,
- student progress tracking,
- attendance monitoring integration,
- LMS integration,
- assignment analytics,
- longitudinal academic tracking,
- real-time alerts,
- multi-user institutional deployment.

---

# License

This project is intended for educational and research purposes.
