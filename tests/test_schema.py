"""Unit tests for Pydantic input/output schemas."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schema import CustomerFeatures, PredictionResponse

VALID = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
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


def test_valid_customer_parses():
    c = CustomerFeatures(**VALID)
    assert c.tenure == 12
    assert c.gender == "Female"


def test_tenure_below_zero_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "tenure": -1})


def test_tenure_above_max_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "tenure": 200})


def test_monthly_charges_negative_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "MonthlyCharges": -10.0})


def test_invalid_gender_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "gender": "Other"})


def test_invalid_contract_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "Contract": "Quarterly"})


def test_invalid_senior_citizen_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "SeniorCitizen": 2})


def test_invalid_payment_method_raises():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "PaymentMethod": "Crypto"})


def test_model_dump_returns_all_fields():
    c = CustomerFeatures(**VALID)
    d = c.model_dump()
    assert len(d) == len(VALID)


def test_prediction_response_valid():
    r = PredictionResponse(
        churn_probability=0.72,
        churn_prediction=True,
        risk_level="High",
        recommendation="Contact customer immediately.",
        latency_ms=12.5,
    )
    assert r.risk_level == "High"
    assert r.model_version == "VotingClassifier-v1.0"
