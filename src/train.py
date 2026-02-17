# train.py

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

from xgboost import XGBClassifier

# For sparse handling in SHAP
from scipy import sparse

# AWS
import boto3


# -----------------------------
# Config
# -----------------------------
DB_USER = "victorchen"
DB_PASS = ""  # leave blank if no password
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "cardio_ml"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

S3_BUCKET = "victor-cardio-ml-artifacts"
S3_PREFIX = "cardio-risk"


def upload_directory_to_s3(local_dir: str, bucket: str, prefix: str):
    """
    Upload all files in local_dir to s3://bucket/prefix/<relative_path>
    """
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, local_dir)
            s3_key = f"{prefix.rstrip('/')}/{rel_path}"
            s3.upload_file(full_path, bucket, s3_key)
            print(f"Uploaded: s3://{bucket}/{s3_key}")


def upload_file_to_s3(local_path: str, bucket: str, key: str):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded: s3://{bucket}/{key}")


def report(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print("AUC:", roc_auc_score(y_true, proba))
    print("F1:", f1_score(y_true, pred))
    print("Precision:", precision_score(y_true, pred, zero_division=0))
    print("Recall:", recall_score(y_true, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))


# DB connection
if DB_PASS:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

df = pd.read_sql("SELECT * FROM training_view;", engine)

y = df["y"].astype(int)
X = df.drop(columns=["y"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Columns
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "chol_bp_ratio"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "age_group", "bp_group"]

# Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# Logistic Regression baseline

logreg = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000))
    ]
)

logreg.fit(X_train, y_train)
proba_lr = logreg.predict_proba(X_test)[:, 1]
report("Logistic Regression", y_test, proba_lr)

# XGBoost + RandomizedSearchCV
xgb = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_distributions = {
    "clf__n_estimators": [200, 400, 800],
    "clf__max_depth": [2, 3, 4, 5, 6],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
    "clf__min_child_weight": [1, 3, 5, 10],
    "clf__reg_lambda": [0.0, 0.5, 1.0, 5.0, 10.0],
    "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0],
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("\nBest params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

proba_xgb = best_model.predict_proba(X_test)[:, 1]
report("XGBoost (Tuned)", y_test, proba_xgb)



# Threshold tuning
best_t = 0.5
best_f1 = -1.0

for t in np.linspace(0.1, 0.9, 81):
    preds = (proba_xgb >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = float(f1)
        best_t = float(t)

print("\nBest threshold:", best_t)
print("Best F1:", best_f1)

results = X_test.copy()
results["y_true"] = y_test
results["y_pred"] = (proba_xgb >= best_t).astype(int)

false_negatives = results[(results["y_true"] == 1) & (results["y_pred"] == 0)]
print("False negatives:", len(false_negatives))

if "age_group" in false_negatives.columns:
    print("\nFalse negatives by age_group:")
    print(false_negatives["age_group"].value_counts())


# Run folder

run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
run_dir = os.path.join(ARTIFACT_DIR, f"run_id={run_id}")
os.makedirs(run_dir, exist_ok=True)


# SHAP (use fitted best_model)
prep = best_model.named_steps["prep"]
model = best_model.named_steps["clf"]

X_test_enc = prep.transform(X_test)

if sparse.issparse(X_test_enc):
    X_test_enc_dense = X_test_enc.toarray()
else:
    X_test_enc_dense = X_test_enc

feature_names = prep.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_enc_dense)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Save SHAP plots INTO run_dir
bar_path = os.path.join(run_dir, "shap_summary_bar.png")
bee_path = os.path.join(run_dir, "shap_summary_beeswarm.png")

plt.figure()
shap.summary_plot(
    shap_values,
    X_test_enc_dense,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(bar_path, dpi=200)
plt.close()

plt.figure()
shap.summary_plot(
    shap_values,
    X_test_enc_dense,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(bee_path, dpi=200)
plt.close()

print("Saved SHAP plots:", bar_path, bee_path)

# Save artifacts (INTO run_dir)

model_path = os.path.join(run_dir, "model.joblib")
joblib.dump(best_model, model_path)

threshold_path = os.path.join(run_dir, "threshold.json")
with open(threshold_path, "w") as f:
    json.dump({"best_threshold": best_t, "best_f1": best_f1}, f, indent=2)

best_params_path = os.path.join(run_dir, "best_params.json")
with open(best_params_path, "w") as f:
    json.dump(search.best_params_, f, indent=2)

metadata_path = os.path.join(run_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(
        {
            "run_id": run_id,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv": "StratifiedKFold(n_splits=5)",
            "search": "RandomizedSearchCV(n_iter=30, scoring=roc_auc)",
            "test_auc": float(roc_auc_score(y_test, proba_xgb)),
            "threshold_best": best_t,
            "threshold_best_f1": best_f1,
            "false_negatives": int(len(false_negatives)),
        },
        f,
        indent=2,
    )

print(f"\nSaved artifacts to: {run_dir}")
run_prefix = f"{S3_PREFIX}/models/run_id={run_id}"
upload_directory_to_s3(run_dir, S3_BUCKET, run_prefix)

latest_prefix = f"{S3_PREFIX}/latest"
upload_file_to_s3(model_path, S3_BUCKET, f"{latest_prefix}/model.joblib")
upload_file_to_s3(threshold_path, S3_BUCKET, f"{latest_prefix}/threshold.json")
upload_file_to_s3(metadata_path, S3_BUCKET, f"{latest_prefix}/metadata.json")

print(f"\nUploaded run to s3://{S3_BUCKET}/{run_prefix}/ and updated latest/")
