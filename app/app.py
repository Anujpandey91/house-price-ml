# Import required libraries
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

# Create Flask app instance
app = Flask(__name__)

# -------------------------------------------------------
# 🔹 Load Trained Model
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "house_price_model.pkl")

# Load full pipeline
model = joblib.load(MODEL_PATH)

# -------------------------------------------------------
# 🔹 Extract Categories for Dropdown Menus
# -------------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
encoder = preprocessor.named_transformers_["cat"]

location_categories = encoder.categories_[0]
area_type_categories = encoder.categories_[1]

# -------------------------------------------------------
# 🔹 Model Metadata (Displayed in UI)
# -------------------------------------------------------

MODEL_TYPE = "Random Forest Regressor"
TARGET_TRANSFORMATION = "Log(price)"
R2_SCORE = "0.77"
DATASET_SIZE = "13,000+ houses"

# -------------------------------------------------------
# 🔹 Home Route
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=location_categories,
        area_types=area_type_categories,
        model_type=MODEL_TYPE,
        target_transformation=TARGET_TRANSFORMATION,
        r2_score=R2_SCORE,
        dataset_size=DATASET_SIZE
    )

# -------------------------------------------------------
# 🔹 Prediction Route
# -------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "location": request.form["location"],
            "area_type": request.form["area_type"],
            "total_sqft": float(request.form["total_sqft"]),
            "bath": float(request.form["bath"]),
            "balcony": float(request.form["balcony"]),
            "bhk": int(request.form["bhk"]),
        }

        # -------------------------------------------------------
        # 🔹 Backend Validation
        # -------------------------------------------------------

        if data["total_sqft"] < 300 or data["total_sqft"] > 20000:
            raise ValueError("Total Sqft must be between 300 and 20000.")

        if data["bhk"] < 1 or data["bhk"] > 10:
            raise ValueError("BHK must be between 1 and 10.")

        if data["bath"] < 1 or data["bath"] > 10:
            raise ValueError("Bathrooms must be between 1 and 10.")

        if data["balcony"] < 0 or data["balcony"] > 5:
            raise ValueError("Balconies must be between 0 and 5.")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict log price
        log_price = model.predict(input_df)[0]

        # Convert back to actual price
        price = np.exp(log_price)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: {price:.2f} Lakhs",
            locations=location_categories,
            area_types=area_type_categories,
            model_type=MODEL_TYPE,
            target_transformation=TARGET_TRANSFORMATION,
            r2_score=R2_SCORE,
            dataset_size=DATASET_SIZE
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            locations=location_categories,
            area_types=area_type_categories,
            model_type=MODEL_TYPE,
            target_transformation=TARGET_TRANSFORMATION,
            r2_score=R2_SCORE,
            dataset_size=DATASET_SIZE
        )

# -------------------------------------------------------
# 🔹 Run App
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)