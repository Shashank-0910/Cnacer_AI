# predict.py
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model/cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_stage(features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    stage = model.predict(X_scaled)[0]
    confidence = max(model.predict_proba(X_scaled)[0])
    return {"stage": int(stage), "confidence": round(confidence, 2)}
