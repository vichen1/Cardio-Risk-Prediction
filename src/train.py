import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.metrics import f1_score
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from xgboost import XGBClassifier

DB_USER = "victorchen"
DB_PASS = ""          
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "cardio_ml"

if DB_PASS:
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
else:
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

df = pd.read_sql("SELECT * FROM training_view;", engine)

y = df["y"].astype(int)
X = df.drop(columns=["y"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Columns
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "chol_bp_ratio"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "age_group", "bp_group"]

# Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

def report(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    print(f"\n=== {name} (threshold={threshold}) ===")
    print("AUC:", roc_auc_score(y_true, proba))
    print("F1:", f1_score(y_true, pred))
    print("Precision:", precision_score(y_true, pred))
    print("Recall:", recall_score(y_true, pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))

# ---- Logistic Regression baseline ----
logreg = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000))
    ]
)

logreg.fit(X_train, y_train)
proba_lr = logreg.predict_proba(X_test)[:, 1]
report("Logistic Regression", y_test, proba_lr)

# ---- XGBoost ----
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
            random_state=42
        ))
    ]
)

xgb.fit(X_train, y_train)
proba_xgb = xgb.predict_proba(X_test)[:, 1]
report("XGBoost", y_test, proba_xgb)

best_t = 0.5
best_f1 = 0

for t in np.linspace(0.1, 0.9, 81):
    preds = (proba_xgb >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest threshold:", best_t)
print("Best F1:", best_f1)

results = X_test.copy()
results["y_true"] = y_test.values
results["y_pred"] = (proba_xgb >= best_t).astype(int)

false_negatives = results[(results["y_true"] == 1) & (results["y_pred"] == 0)]
print("False negatives:", len(false_negatives))

print("\nFalse negatives by age_group:")
print(false_negatives["age_group"].value_counts())


# 1) Get fitted preprocessor + fitted model out of the pipeline
prep = xgb.named_steps["prep"]
model = xgb.named_steps["clf"]

# 2) Transform X_test using the SAME preprocessing
X_test_enc = prep.transform(X_test)

# 3) Get feature names after one-hot encoding (important!)
feature_names = prep.get_feature_names_out()

# 4) SHAP on the XGBoost model (tree explainer)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_enc)

# 5) Summary plot (bar = global importance)
plt.figure()
shap.summary_plot(shap_values, X_test_enc, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=200)
plt.close()

# 6) Summary plot (beeswarm = distribution of impact)
plt.figure()
shap.summary_plot(shap_values, X_test_enc, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=200)
plt.close()

print("Saved SHAP plots: shap_summary_bar.png, shap_summary_beeswarm.png")
