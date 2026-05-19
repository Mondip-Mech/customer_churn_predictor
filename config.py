from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "final_churn_model.pkl"
BACKGROUND_DATA_PATH = BASE_DIR / "background_data.pkl"
METRICS_PATH = BASE_DIR / "metrics" / "model_performance.json"

# Feature groups matching the training pipeline
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
OHE_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
ALL_FEATURES = NUM_FEATURES + ["SeniorCitizen"] + BINARY_FEATURES + OHE_FEATURES

# Business cost assumptions (USD) — adjust to match your company's economics
COST_FALSE_NEGATIVE = 200   # Revenue lost per missed churner (avg monthly charge × ~3 months)
COST_FALSE_POSITIVE = 25    # Cost of unnecessary retention offer sent to a loyal customer

DECISION_THRESHOLD = 0.5    # Default classification threshold
