# test_model.py
import joblib
import numpy as np
import pandas as pd

# Load trained model
model_path = "model/cancer_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Create a sample test input (use same features as training, except 'stage')
sample_data = {
    "patient_id": ["P00001"], 
    "age": [55],
    "sex": ["Male"],
    "bp_systolic": [130],
    "bp_diastolic": [85],
    "hr": [78],
    "temp_c": [36.8],
    "SpO2": [97],
    "cancer_type": ["Lung"],
    "histology": ["Adenocarcinoma"],
    "tnm_t": [2],
    "tnm_n": [1],
    "tnm_m": [0],
    "EGFR": ["Positive"],
    "ALK": ["Negative"],
    "HER2": ["Negative"],
    "BRCA": ["Negative"],
    "hb": [13.2],
    "wbc": [6.5],
    "platelets": [220],
    "creatinine": [1.0],
    "alt": [35],
    "ast": [30],
    "diabetes": [0],
    "hypertension": [1],
    "renal_disease": [0],
    "num_prior_treatments": [2],
    "last_treatment_type": ["Chemotherapy"],
    "time_since_last_treatment_days": [120],
    "ECOG_status": [1],
    "progression": [0],
    "recommended_action": ["Monitor"],
    "response": ["Stable"]
}

# Convert dict to DataFrame
test_df = pd.DataFrame(sample_data)

# Make prediction
prediction = model.predict(test_df)
print(f"\n🎯 Predicted Cancer Stage: {prediction[0]}")
