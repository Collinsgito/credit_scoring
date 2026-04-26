"""
api/main.py
-----------
FastAPI application for Credit Risk Scoring.

Start: uvicorn api.main:app --reload --port 8000
Docs:  http://127.0.0.1:8000/docs
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception as shap_import_error:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = shap_import_error

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ApplicantInput,
    CreditScoreResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfoResponse,
)

# ── Paths ──────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
MODELS_DIR  = ROOT / "models"

MODEL_PATH     = MODELS_DIR / "credit_model.pkl"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"
FEATURES_PATH  = MODELS_DIR / "feature_names.pkl"

# ── Global state ───────────────────────────────────────────────
state = {
    "model": None,
    "explainer": None,
    "feature_names": None,
}

# ── FICO bands ─────────────────────────────────────────────────
FICO_BANDS = [
    ("Exceptional",  "800–850", 800, 850),
    ("Very Good",    "740–799", 740, 799),
    ("Good",         "670–739", 670, 739),
    ("Fair",         "580–669", 580, 669),
    ("Poor",         "300–579", 300, 579),
]
FICO_ENC = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Exceptional": 4}

# ── Risk tiers ─────────────────────────────────────────────────
RISK_TIERS = [
    (0.00, 0.05, "Low Risk",       "#1A7A4A", "Approve"),
    (0.05, 0.12, "Medium Risk",    "#C9A84C", "Approve with conditions"),
    (0.12, 0.25, "High Risk",      "#D05538", "Manual review required"),
    (0.25, 1.00, "Very High Risk", "#8B1A1A", "Decline"),
]

# ── Feature display names ──────────────────────────────────────
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Credit Utilization",
    "age":                                  "Applicant Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 Days Late",
    "DebtRatio":                            "Debt Ratio",
    "MonthlyIncome":                        "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans":      "Open Credit Lines",
    "NumberOfTimes90DaysLate":              "Times 90+ Days Late",
    "NumberRealEstateLoansOrLines":         "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 Days Late",
    "NumberOfDependents":                   "Number of Dependents",
    "FICOScore":                            "FICO Score",
    "FICO_Band_enc":                        "FICO Band (encoded)",
    "DebtToIncomeRatio":                    "Debt-to-Income Ratio",
    "TotalLatePayments":                    "Total Late Payment Count",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup."""
    print("Loading model artifacts...")
    if not SHAP_AVAILABLE:
        print(f"  ! SHAP disabled: {SHAP_IMPORT_ERROR}")

    if MODEL_PATH.exists():
        state["model"]        = joblib.load(MODEL_PATH)
        print(f"  ✓ Model loaded: {MODEL_PATH.name}")
    else:
        print(f"  ✗ Model not found at {MODEL_PATH}")
        print("    Run: python train_model.py")

    if SHAP_AVAILABLE and EXPLAINER_PATH.exists():
        state["explainer"]    = joblib.load(EXPLAINER_PATH)
        print(f"  ✓ SHAP explainer loaded")
    elif not SHAP_AVAILABLE:
        print("  ! SHAP explainer skipped because SHAP is unavailable")
    else:
        print("  ✗ SHAP explainer not found. Run train_model.py")

    if FEATURES_PATH.exists():
        state["feature_names"] = joblib.load(FEATURES_PATH)
        print(f"  ✓ Feature names loaded: {len(state['feature_names'])} features")
    else:
        print("  ✗ Feature names not found. Run train_model.py")

    explainability_status = "enabled" if state["explainer"] is not None else "disabled"
    model_status = "ready" if state["model"] is not None else "not ready"
    print("API startup summary:")
    print(f"  • URL: http://127.0.0.1:8000")
    print(f"  • Model: {model_status}")
    print(f"  • Explainability (SHAP): {explainability_status}")
    print("  • OpenAPI docs: http://127.0.0.1:8000/docs")

    yield
    print("Shutting down...")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Scoring API",
    description="""
## AI-Powered Credit Risk Scoring

Built by **Collins Gitonga Mutembei** | Data Scientist & ML Engineer

This API scores loan applicants using an XGBoost model trained on credit bureau data,
with SHAP-based explainability for every decision.

### Features
- 🎯 Probability of default (0–1)
- 📊 Risk score (0–1000)
- 🏷️ Risk tier classification
- 🔍 SHAP-powered explanations
- 💳 FICO band classification
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────
def get_fico_band(score: int):
    for band, label, lo, hi in FICO_BANDS:
        if lo <= score <= hi:
            return band, label
    return "Unknown", "N/A"


def get_risk_tier(prob: float):
    for lo, hi, label, color, decision in RISK_TIERS:
        if lo <= prob < hi:
            return label, color, decision
    return "Very High Risk", "#8B1A1A", "Decline"


def build_feature_vector(data: ApplicantInput, feature_names: list) -> pd.DataFrame:
    """Convert API input to model feature vector."""
    monthly_income = data.monthly_income if data.monthly_income is not None else 3500.0
    num_dependents = data.num_dependents if data.num_dependents is not None else 0

    fico_band, _ = get_fico_band(data.fico_score)
    fico_enc     = FICO_ENC.get(fico_band, 2)

    safe_income = monthly_income if monthly_income > 0 else 1
    debt_to_income = (data.debt_ratio * safe_income) / safe_income

    total_late = (
        data.num_30_59_days_late
        + data.num_60_89_days_late
        + data.num_90_days_late
    )

    raw = {
        "RevolvingUtilizationOfUnsecuredLines": data.revolving_utilization,
        "age":                                  data.age,
        "NumberOfTime30-59DaysPastDueNotWorse": data.num_30_59_days_late,
        "DebtRatio":                            data.debt_ratio,
        "MonthlyIncome":                        monthly_income,
        "NumberOfOpenCreditLinesAndLoans":      data.num_open_credit_lines,
        "NumberOfTimes90DaysLate":              data.num_90_days_late,
        "NumberRealEstateLoansOrLines":         data.num_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": data.num_60_89_days_late,
        "NumberOfDependents":                   float(num_dependents),
        "FICOScore":                            data.fico_score,
        "FICO_Band_enc":                        fico_enc,
        "DebtToIncomeRatio":                    debt_to_income,
        "TotalLatePayments":                    total_late,
    }

    available = [f for f in feature_names if f in raw]
    return pd.DataFrame([{k: raw[k] for k in available}])


def get_shap_contributions(
    explainer, df: pd.DataFrame, feature_names: list
) -> tuple[list, list]:
    """Return top risk and protective factors from SHAP."""
    try:
        sv = explainer.shap_values(df)
        if isinstance(sv, list):
            sv = sv[1]  # XGBoost binary classification sometimes returns list
        sv = sv[0]  # first (only) row

        contribs = []
        for i, fname in enumerate(feature_names):
            if i < len(sv):
                val = float(df.iloc[0][fname]) if fname in df.columns else 0.0
                contribs.append(FeatureContribution(
                    feature=FEATURE_LABELS.get(fname, fname),
                    value=round(val, 4),
                    shap_value=round(float(sv[i]), 4),
                    direction="increases_risk" if sv[i] > 0 else "decreases_risk",
                ))

        risk_factors       = sorted([c for c in contribs if c.shap_value > 0],
                                    key=lambda x: x.shap_value, reverse=True)[:4]
        protective_factors = sorted([c for c in contribs if c.shap_value < 0],
                                    key=lambda x: x.shap_value)[:4]
        return risk_factors, protective_factors

    except Exception as e:
        print(f"SHAP error: {e}")
        return [], []


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """API health check — verify model is loaded."""
    return HealthResponse(
        status="healthy" if state["model"] else "degraded",
        model_loaded=state["model"] is not None,
        explainer_loaded=state["explainer"] is not None,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model metadata."""
    if not state["model"]:
        raise HTTPException(503, "Model not loaded. Run train_model.py first.")
    return ModelInfoResponse(
        model_type="XGBoost (XGBClassifier)",
        feature_count=len(state["feature_names"] or []),
        features=[FEATURE_LABELS.get(f, f) for f in (state["feature_names"] or [])],
        thresholds={
            "low":       "< 5% probability of default",
            "medium":    "5–12% probability of default",
            "high":      "12–25% probability of default",
            "very_high": "> 25% probability of default",
        },
        training_notes="XGBoost with SMOTE oversampling. SHAP explainability enabled.",
    )


@app.post("/score", response_model=CreditScoreResponse, tags=["Scoring"])
async def score_applicant(data: ApplicantInput):
    """
    Score a credit applicant.

    Returns:
    - Probability of default
    - Risk score (0–1000)
    - Risk tier and recommended decision
    - FICO band classification
    - SHAP-based top risk and protective factors
    """
    if not state["model"]:
        raise HTTPException(503, "Model not loaded. Run: python train_model.py")

    try:
        df = build_feature_vector(data, state["feature_names"])
        prob = float(state["model"].predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

    risk_tier, color, decision = get_risk_tier(prob)
    fico_band, fico_label      = get_fico_band(data.fico_score)

    # Risk score: 0 = max risk, 1000 = safest
    risk_score = int((1 - prob) * 1000)

    # SHAP explanations
    risk_factors, protective_factors = [], []
    if state["explainer"] is not None:
        risk_factors, protective_factors = get_shap_contributions(
            state["explainer"], df, state["feature_names"]
        )

    return CreditScoreResponse(
        probability_of_default=round(prob, 4),
        risk_score=risk_score,
        risk_tier=risk_tier,
        risk_tier_color=color,
        decision=decision,
        fico_band=fico_band,
        fico_label=fico_label,
        top_risk_factors=risk_factors,
        top_protective_factors=protective_factors,
    )


@app.post("/score/batch", tags=["Scoring"])
async def score_batch(applicants: list[ApplicantInput]):
    """Score multiple applicants in one request (max 100)."""
    if len(applicants) > 100:
        raise HTTPException(400, "Maximum 100 applicants per batch request.")
    results = []
    for applicant in applicants:
        result = await score_applicant(applicant)
        results.append(result)
    return {"count": len(results), "results": results}
