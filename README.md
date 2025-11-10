# EMI Risk Assessment Platform

**Repository:** EMI Risk Assessment Platform — Dual ML (Classification + Regression) with MLflow tracking and Streamlit UI

---

## 🚀 Project Overview

This project builds a production-ready financial risk assessment platform that helps lenders, fintechs, banks, and loan officers make fast, data-driven decisions about EMI eligibility and recommended EMI amounts. It combines two supervised learning objectives:

* **Classification** — Predict whether an applicant is *eligible* for EMI (binary: Eligible / Not Eligible).
* **Regression** — Predict the **maximum EMI amount** that an applicant can sustainably afford.

Key capabilities:

* Trains and evaluates multiple models for both tasks (baseline + advanced ensembles).
* Tracks experiments, metrics, artifacts, and model versions using **MLflow** (tracking server + model registry).
* Real-time inference and interactive analytics through a **multi-page Streamlit** application.
* Full **CRUD** operations for financial records (admin/Dev API + UI) to support onboarding and live dataset updates.
* Scales to and demonstrates with **~400,000 records** and 22 core financial/demographic features.

---

## 📌 Business Use Cases

* **Financial Institutions**: Automate underwriting, risk-based pricing and reduce manual review time.
* **FinTech**: Provide instant pre-qualification checks in onboarding and mobile apps.
* **Banks & Credit Agencies**: Data-driven loan amount recommendations and regulatory audit trails.
* **Loan Officers**: AI-assisted decision support with explainability and historical performance tracking.

---

## 🔬 Dataset

* **Records**: 400,000 realistic financial records across multiple EMI scenarios.
* **Features**: 22 financial & demographic variables (income, existing liabilities, credit score, employment length, age, dependents, housing status, monthly expenses breakdown, etc.).
* **Targets**:

  * `emi_eligible` (binary)
  * `max_emi_amount` (numeric)

> Note: This repo expects a `data/` folder with `emi_dataset.csv` (or similar). A sample data generator script `scripts/generate_sample_data.py` is included for testing without the full dataset.

---

## 🛠️ Architecture / Components

1. **Data Layer**

   * Raw and cleaned datasets in `data/`.
   * CRUD API backed by SQLite/Postgres (configurable) for record management.
2. **ML Pipeline**

   * Data preprocessing & feature engineering module
   * Model training & evaluation scripts (classification + regression)
   * Hyperparameter search (GridSearch / Randomized / Optuna - optional)
3. **Experiment Tracking**

   * MLflow tracking server + artifact store (local or S3-compatible)
   * Model registry for promoted models (staging/production)
4. **Serving & UI**

   * Streamlit multi-page app with prediction UI, admin CRUD, EDA, and MLflow dashboard integration
   * REST API endpoints for CRUD and inference (FastAPI optional)
5. **Deployment**

   * Streamlit Cloud deployment guide included
   * MLflow server + backend store instructions (docker-compose / k8s manifests)


## ✅ Feature Highlights

### Data & Preprocessing

* Missing value imputation (domain-aware strategies).
* Duplicate and consistency checks; data quality reporting.
* Train / validation / test splits with stratification for the classification label.

### Feature Engineering

* Derived ratios: **Debt-to-Income (DTI)**, **Expense-to-Income**, **Affordability Ratio**, **Savings Buffer**.
* Employment stability and credit-history-based risk scoring features.
* Categorical encoding (target encoding, one-hot, or ordinal where appropriate).
* Scalers: RobustScaler / StandardScaler for numeric stability.
* Interaction features between income, existing EMIs, and expenses.

### Models (Minimum 3 per task)

**Classification**

* Logistic Regression (baseline)
* Random Forest Classifier
* XGBoost Classifier
* Optional: SVC, LightGBM

**Regression**

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost Regressor
* Optional: SVR, GradientBoostingRegressor

Evaluation metrics logged to MLflow:

* Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
* Regression: RMSE, MAE, R², MAPE, Residual plots

### MLflow Integration

* Track experiments: params, metrics, artifacts (plots, confusion matrices, feature importance CSVs).
* Automated logging from training scripts using `mlflow.sklearn.log_model`.
* Model registry with model versioning (staging -> production workflows).

### Streamlit Application

* Multi-page layout:  **Predict (Real-time) EMI  based o yput input**, 

