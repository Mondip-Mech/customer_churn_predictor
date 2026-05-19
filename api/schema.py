"""Pydantic schemas for request validation and response serialisation.

Every field mirrors the exact column name and allowed values from the
Telco Customer Churn dataset so the model pipeline receives a clean
DataFrame with zero preprocessing surprises.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input schema — one telecom customer record."""

    # ── Numerical ────────────────────────────────────────────────────────────
    tenure: int = Field(..., ge=0, le=100,
                        description="Months the customer has been with the company")
    MonthlyCharges: float = Field(..., ge=0.0, le=200.0,
                                  description="Current monthly bill (USD)")
    TotalCharges: float = Field(..., ge=0.0,
                                description="Total amount billed to date (USD)")

    # ── Demographics ─────────────────────────────────────────────────────────
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(...,
                                         description="1 if customer is 65+, else 0")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # ── Phone services ───────────────────────────────────────────────────────
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    # ── Internet services ────────────────────────────────────────────────────
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # ── Billing ──────────────────────────────────────────────────────────────
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    """Output schema — churn prediction with metadata."""

    churn_probability: float = Field(..., description="Probability of churning [0–1]")
    churn_prediction: bool = Field(..., description="True = predicted to churn")
    risk_level: Literal["High", "Medium", "Low"] = Field(
        ..., description="High ≥ 0.70 · Medium ≥ 0.40 · Low < 0.40"
    )
    recommendation: str = Field(..., description="Suggested retention action")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(default="VotingClassifier-v1.0")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    version: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    latency_ms: float
