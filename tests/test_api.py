"""API integration tests.

Run with:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient that triggers the app lifespan (loads model on startup)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def model_available(client):
    """Skip inference tests if model file is not present in CI."""
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        pytest.skip("Model file not available — skipping inference tests")


# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_CUSTOMER = {
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

HIGH_TENURE_CUSTOMER = {**VALID_CUSTOMER, "tenure": 60, "Contract": "Two year",
                        "MonthlyCharges": 45.0, "TotalCharges": 2700.0}


# ── Health ────────────────────────────────────────────────────────────────────
def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_response_schema(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


# ── Single prediction ─────────────────────────────────────────────────────────
def test_predict_valid_customer(client, model_available):
    assert client.post("/predict", json=VALID_CUSTOMER).status_code == 200


def test_predict_response_fields(client, model_available):
    data = client.post("/predict", json=VALID_CUSTOMER).json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert "latency_ms" in data


def test_predict_probability_range(client, model_available):
    data = client.post("/predict", json=VALID_CUSTOMER).json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_risk_level_valid(client, model_available):
    data = client.post("/predict", json=VALID_CUSTOMER).json()
    assert data["risk_level"] in ("High", "Medium", "Low")


def test_predict_high_tenure_low_risk(client, model_available):
    """Long-tenure Two-year contract customers should have lower churn probability."""
    low  = client.post("/predict", json=HIGH_TENURE_CUSTOMER).json()
    high = client.post("/predict", json=VALID_CUSTOMER).json()
    assert low["churn_probability"] <= high["churn_probability"]


# ── Validation errors (no model needed — Pydantic rejects before inference) ───
def test_predict_missing_field_returns_422(client):
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_invalid_tenure_returns_422(client):
    assert client.post("/predict", json={**VALID_CUSTOMER, "tenure": -5}).status_code == 422


def test_predict_invalid_contract_returns_422(client):
    assert client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Weekly"}).status_code == 422


def test_predict_invalid_gender_returns_422(client):
    assert client.post("/predict", json={**VALID_CUSTOMER, "gender": "Unknown"}).status_code == 422


# ── Batch prediction ──────────────────────────────────────────────────────────
def test_batch_predict_two_customers(client, model_available):
    payload = {"customers": [VALID_CUSTOMER, HIGH_TENURE_CUSTOMER]}
    assert client.post("/predict/batch", json=payload).status_code == 200


def test_batch_response_count(client, model_available):
    payload = {"customers": [VALID_CUSTOMER, HIGH_TENURE_CUSTOMER]}
    data = client.post("/predict/batch", json=payload).json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_batch_empty_returns_422(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


# ── Monitoring endpoints ──────────────────────────────────────────────────────
def test_predictions_log_returns_200(client):
    assert client.get("/predictions?limit=5").status_code == 200


def test_metrics_returns_200(client):
    assert client.get("/metrics").status_code == 200


def test_predictions_limit_too_large_returns_400(client):
    assert client.get("/predictions?limit=999").status_code == 400
