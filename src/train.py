import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from xgboost import XGBClassifier

# ---- UPDATE THIS if needed ----
DB_USER = "victorchen"
DB_PASS = ""          # leave blank if you didn't set one
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
