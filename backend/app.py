from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Load trained cancer model
MODEL_PATH = os.path.join("model", "cancer_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# ✅ Home route to check API
@app.route('/')
def home():
    return jsonify({"message": "Cancer Stage Prediction API is running ✅"})


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📩 Received data:", data)

        if not data:
            return jsonify({"error": "No input data received"}), 400

        input_df = pd.DataFrame([data])

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Model prediction
        prediction = model.predict(input_df)[0]
        print("✅ Prediction:", prediction)

        # ✅ Make sure this matches your frontend key
        return jsonify({"predicted_stage": str(prediction)})

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
