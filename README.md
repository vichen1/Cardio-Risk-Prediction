# Cardiovascular Risk Prediction Pipeline 🫀 

## Overview

This project builds an end-to-end machine learning pipeline to predict cardiovascular disease risk using structured clinical features.

The system includes:

* PostgreSQL feature engineering (training_view)
* Logistic Regression baseline
* XGBoost with hyperparameter tuning
* 5-fold Stratified Cross-Validation
* Threshold optimization to reduce false negatives
* SHAP model interpretability
* Versioned artifact storage in AWS S3
* Cloud-based model retrieval for inference

The goal is to reduce false negatives (missed high-risk patients) while maintaining strong overall classification performance.

# Dataset Description

This project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.  
The dataset contains clinical measurements used to predict the presence of cardiovascular disease.

* Number of observations: ~303 patients  
* Task: Binary classification (heart disease present vs not present)

---

## Feature Definitions

### Demographic Features

age  
Age of the patient (years)

sex  
0 = female  
1 = male  

---

### Chest Pain Characteristics

cp (chest pain type)  
1 = typical angina  
2 = atypical angina  
3 = non-anginal pain  
4 = asymptomatic  

Chest pain type is one of the strongest predictors of cardiovascular risk.

---

### Resting Clinical Measurements

trestbps  
Resting blood pressure (mm Hg)

chol  
Serum cholesterol level (mg/dL)

fbs  
Fasting blood sugar > 120 mg/dL  
1 = true  
0 = false  

restecg  
Resting electrocardiogram results  
0 = normal  
1 = ST-T wave abnormality  
2 = left ventricular hypertrophy  

---

### Exercise-Based Measurements

thalach  
Maximum heart rate achieved during exercise

exang  
Exercise-induced angina  
1 = yes  
0 = no  

oldpeak  
ST depression induced by exercise relative to rest  
Higher values indicate greater abnormal heart stress response

slope  
Slope of the peak exercise ST segment  
1 = upsloping  
2 = flat  
3 = downsloping  

---

### Imaging / Diagnostic Features

ca  
Number of major vessels colored by fluoroscopy (0–3)  
Higher values indicate greater arterial blockage

thal  
Thalassemia test result  
3 = normal  
6 = fixed defect  
7 = reversible defect  

---

### Target Variable

target (or y)  
1 = presence of heart disease  
0 = no heart disease  

---

## Dataset Considerations

The Cleveland dataset contains approximately 300 observations, which is relatively small for modern machine learning standards. To mitigate variance and improve generalization reliability:

* 5-fold Stratified Cross-Validation was used  
* Hyperparameter tuning was performed via RandomizedSearchCV  
* Threshold optimization was applied to reduce false negatives  
* Model interpretability was analyzed using SHAP  

Despite its size, the dataset remains a widely used benchmark in medical risk prediction research.


## Architecture 

PostgreSQL (training_view)
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Model Training
   - Logistic Regression
   - XGBoost + RandomizedSearchCV
        ↓
Threshold Optimization
        ↓
SHAP Interpretability
        ↓
Artifacts Saved (run_id versioned)
        ↓
AWS S3 Storage
        ↓
Inference (local or cloud download)


## Model Performance 
* Logistic Regression (Baseline)
* AUC: 0.93
* F1: 0.79
* Recall: 0.75
* False Negatives: 7

## XGBoost (Tuned)

* Hyperparameter tuning:
* 5-fold StratifiedKFold
* RandomizedSearchCV (30 iterations)
* ROC-AUC optimization

### Results:

* AUC: 0.94
* F1 (0.5 threshold): 0.81
* F1 (optimized threshold 0.35): 0.88
* False Negatives reduced from 7 → 2
* Threshold tuning significantly reduced missed high-risk patients.

---

## Interpretability

SHAP (SHapley Additive exPlanations) was used to analyze global and local feature importance for the tuned XGBoost model.

Top predictive drivers identified:

* ca (number of major vessels)
* cp (chest pain type)
* chol (cholesterol)
* age
* oldpeak (ST depression)
* thal (thalassemia defect type)

Two visualizations are generated per training run:

* SHAP summary bar plot (global importance)
* SHAP beeswarm plot (feature impact distribution)

These plots are saved within each run_id artifact folder and uploaded to AWS S3.

---

## Cloud Integration (AWS S3)

Each training run automatically:

* Saves artifacts locally under a unique run_id
* Uploads artifacts to versioned S3 storage:
  s3://<bucket>/cardio-risk/models/run_id=<timestamp>/
* Updates a production pointer:
  s3://<bucket>/cardio-risk/latest/

Saved artifacts include:

* model.joblib
* threshold.json
* best_params.json
* metadata.json
* SHAP plots

This enables:

* Reproducibility
* Model version control
* Lightweight MLOps workflow
* Cloud-based inference retrieval

---

## How to Run

### 1) Database Setup

Ensure PostgreSQL contains a table or view named:

training_view

This view should contain all engineered features and the target variable.

---

### 2) Train the Model

Run:

python train.py

This will:

* Train Logistic Regression baseline
* Perform hyperparameter tuning on XGBoost
* Optimize decision threshold
* Generate SHAP visualizations
* Save artifacts locally
* Upload artifacts to AWS S3

---

### 3) Run Inference

Run:

python inference.py

This script:

* Downloads the latest model from S3 (or loads locally)
* Loads the optimized threshold
* Predicts cardiovascular risk for new patients

## Tech Stack

* Python
* PostgreSQL
* scikit-learn
* XGBoost
* SHAP
* AWS S3
* boto3

---

## Key Design Decisions

* ROC-AUC was selected as the primary tuning metric due to its robustness under class imbalance.
* Threshold optimization was performed to prioritize reducing false negatives, which is critical in healthcare risk prediction.
* SHAP was used to provide transparent, clinically interpretable explanations.
* AWS S3 was integrated to provide scalable, versioned artifact storage.

## Limitations

* The dataset contains ~300 observations, which limits model generalization to broader populations.
* The dataset represents a specific cohort (Cleveland Clinic), which may introduce sampling bias.
* External validation on larger, more diverse datasets would improve reliability.



## Author

Victor Chen  
Data Science @UCSD
