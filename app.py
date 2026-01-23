from fastapi import FastAPI
import joblib
import pandas as pd

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="Customer Churn Prediction API")

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model/churn_model.joblib")

# Business threshold from your analysis
THRESHOLD = 0.12

# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
def predict(data: dict):
    """
    Expects raw customer data in JSON format.
    Returns churn probability and prediction.
    """

    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])

    # ----------------------------
    # Apply SAME feature engineering as training
    # ----------------------------

    # Fix TotalCharges if needed
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Feature engineering
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["is_long_term_customer"] = (df["tenure"] > 24).astype(int)

    # ----------------------------
    # Predict
    # ----------------------------
    proba = model.predict_proba(df)[0][1]

    # Apply business threshold
    prediction = int(proba >= THRESHOLD)

    # Risk label
    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": float(proba),
        "churn_prediction": prediction,
        "risk_level": risk
    }
