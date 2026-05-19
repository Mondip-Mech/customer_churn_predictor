"""FastAPI prediction service for the Customer Churn model.

Endpoints
---------
GET  /health                  → liveness probe
GET  /metrics                 → aggregate prediction stats
GET  /predictions             → recent prediction log
POST /predict                 → single customer prediction
POST /predict/batch           → batch prediction (up to 500 customers)

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.logger import get_recent_predictions, get_summary_stats, log_prediction
from api.schema import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from config import DECISION_THRESHOLD, MODEL_PATH

# ── Globals ───────────────────────────────────────────────────────────────────
_model = None
MODEL_VERSION = "VotingClassifier-v1.0"


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH} ✓")
    except FileNotFoundError:
        print(f"WARNING: model file not found at {MODEL_PATH}")
    except Exception as e:
        # Handles sklearn version mismatches or corrupt pickle files
        print(f"WARNING: model could not be loaded ({type(e).__name__}: {e})")
    yield
    _model = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "MLOps-grade REST API for telecom customer churn prediction. "
        "Built with FastAPI · Pydantic · MLflow · scikit-learn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def _recommendation(prob: float, features: dict) -> str:
    if prob < 0.40:
        return "No action needed — customer is stable."
    if features.get("Contract") == "Month-to-month":
        return "Offer a discounted annual contract to lock in retention."
    if features.get("InternetService") == "Fiber optic":
        return "Proactively check service quality and offer a loyalty discount."
    if prob >= 0.70:
        return "High churn risk — escalate to the retention team immediately."
    return "Send a personalised retention offer within 48 hours."


def _run_inference(customer: CustomerFeatures) -> tuple[bool, float, float]:
    """Returns (prediction, probability, latency_ms)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([customer.model_dump()])
    t0 = time.perf_counter()
    prob = float(_model.predict_proba(df)[0][1])
    latency = (time.perf_counter() - t0) * 1000
    prediction = prob >= DECISION_THRESHOLD
    return prediction, prob, latency


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    """Liveness probe — use this for Docker HEALTHCHECK and load-balancer pings."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures):
    """Predict churn probability for a single customer.

    Returns the probability, binary prediction, risk tier, and a tailored
    retention recommendation. Every request is logged to predictions.db.
    """
    prediction, prob, latency = _run_inference(customer)
    risk = _risk_level(prob)
    rec  = _recommendation(prob, customer.model_dump())

    log_prediction(
        features=customer.model_dump(),
        prediction=prediction,
        probability=prob,
        risk_level=risk,
        latency_ms=latency,
    )

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        risk_level=risk,
        recommendation=rec,
        latency_ms=round(latency, 2),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(request: BatchRequest):
    """Predict churn for up to 500 customers in a single request.

    Useful for nightly batch scoring of the entire customer base.
    """
    t0 = time.perf_counter()
    predictions = []

    for customer in request.customers:
        prediction, prob, lat = _run_inference(customer)
        risk = _risk_level(prob)
        rec  = _recommendation(prob, customer.model_dump())
        log_prediction(customer.model_dump(), prediction, prob, risk, lat)
        predictions.append(
            PredictionResponse(
                churn_probability=round(prob, 4),
                churn_prediction=prediction,
                risk_level=risk,
                recommendation=rec,
                latency_ms=round(lat, 2),
                model_version=MODEL_VERSION,
            )
        )

    total_latency = (time.perf_counter() - t0) * 1000
    high_risk = sum(1 for p in predictions if p.risk_level == "High")

    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
        latency_ms=round(total_latency, 2),
    )


@app.get("/predictions", tags=["Monitoring"])
def recent_predictions(limit: int = 20):
    """Return the last N predictions from the log database."""
    if limit > 200:
        raise HTTPException(status_code=400, detail="limit must be ≤ 200")
    return get_recent_predictions(limit)


@app.get("/metrics", tags=["Monitoring"])
def prediction_metrics():
    """Aggregate stats across all logged predictions — use for drift monitoring."""
    return get_summary_stats()
