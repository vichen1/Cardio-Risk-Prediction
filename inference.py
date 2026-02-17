import joblib
import json
import pandas as pd

# Load model
model = joblib.load("model.joblib")

# Load threshold
with open("threshold.json") as f:
    threshold = json.load(f)["best_threshold"]

print("Loaded threshold:", threshold)

# Example patient
sample = pd.DataFrame([{
    "age": 60,
    "trestbps": 140,
    "chol": 240,
    "thalach": 150,
    "oldpeak": 1.5,
    "chol_bp_ratio": 240/140,
    "sex": 1,
    "cp": 2,
    "fbs": 0,
    "restecg": 1,
    "exang": 0,
    "slope": 2,
    "ca": 1,
    "thal": 2,
    "age_group": "senior",
    "bp_group": "high"
}])

proba = model.predict_proba(sample)[:, 1][0]
pred = int(proba >= threshold)

print("Predicted probability:", proba)
print("Predicted class:", pred)
