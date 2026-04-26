"""
api/schemas.py
--------------
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any


class ApplicantInput(BaseModel):
    """Input data for a credit risk scoring request."""

    revolving_utilization: float = Field(
        ..., ge=0.0, le=2.0,
        description="Revolving utilization of unsecured lines (0.0–1.0, capped at 2.0)",
        example=0.35,
    )
    age: int = Field(..., ge=21, le=100, description="Applicant age", example=42)

    num_30_59_days_late: int = Field(
        0, ge=0, le=20,
        description="Times 30–59 days past due (not worse)",
        example=0,
    )
    debt_ratio: float = Field(
        ..., ge=0.0, le=5.0,
        description="Debt ratio (monthly debt payments / gross monthly income)",
        example=0.38,
    )
    monthly_income: Optional[float] = Field(
        None, ge=0.0,
        description="Gross monthly income (USD). Leave null if unknown.",
        example=4500.0,
    )
    num_open_credit_lines: int = Field(
        0, ge=0, le=50,
        description="Number of open credit lines and loans",
        example=7,
    )
    num_90_days_late: int = Field(
        0, ge=0, le=20,
        description="Number of times 90+ days past due",
        example=0,
    )
    num_real_estate_loans: int = Field(
        0, ge=0, le=10,
        description="Number of real estate loans or lines",
        example=1,
    )
    num_60_89_days_late: int = Field(
        0, ge=0, le=20,
        description="Times 60–89 days past due (not worse)",
        example=0,
    )
    num_dependents: Optional[int] = Field(
        None, ge=0, le=20,
        description="Number of dependents. Leave null if unknown.",
        example=2,
    )
    fico_score: int = Field(
        ..., ge=300, le=850,
        description="Credit (FICO) score of the applicant",
        example=685,
    )

    @field_validator("revolving_utilization")
    @classmethod
    def cap_utilization(cls, v):
        return min(v, 1.5)


class RiskTier(str):
    LOW       = "Low Risk"
    MEDIUM    = "Medium Risk"
    HIGH      = "High Risk"
    VERY_HIGH = "Very High Risk"


class FeatureContribution(BaseModel):
    feature:     str
    value:       float
    shap_value:  float
    direction:   str  # "increases_risk" | "decreases_risk"


class CreditScoreResponse(BaseModel):
    """Complete credit risk scoring response."""

    # Core outputs
    probability_of_default: float = Field(..., description="Probability of default (0.0–1.0)")
    risk_score:             int   = Field(..., description="Risk score 0–1000 (lower = safer)")
    risk_tier:              str   = Field(..., description="Risk tier label")
    risk_tier_color:        str   = Field(..., description="Hex color for UI display")
    decision:               str   = Field(..., description="Suggested credit decision")

    # FICO bucket
    fico_band:  str
    fico_label: str

    # Explainability
    top_risk_factors:    List[FeatureContribution]
    top_protective_factors: List[FeatureContribution]

    # Metadata
    model_version: str = "1.0.0"
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "probability_of_default": 0.073,
                "risk_score": 730,
                "risk_tier": "Low Risk",
                "risk_tier_color": "#1A7A4A",
                "decision": "Approve",
                "fico_band": "Good",
                "fico_label": "670–739",
                "top_risk_factors": [],
                "top_protective_factors": [],
                "model_version": "1.0.0",
            }
        },
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    explainer_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    feature_count: int
    features: List[str]
    thresholds: Dict[str, Any]
    training_notes: str
