# 💳 Credit Risk Scoring API

> AI-powered credit risk assessment with XGBoost + SHAP explainability + FastAPI + Streamlit

Built by **Collins Gitonga Mutembei** | Data Scientist & ML Engineer

---

## 🎯 What It Does

- **Predicts** probability of loan default for any applicant
- **Classifies** into risk tiers: Low / Medium / High / Very High
- **Explains** every decision using SHAP feature importance
- **Classifies** FICO scores into standard credit bands (Poor → Exceptional)
- **Exposes** a production-ready REST API with Swagger docs

---

## 🏗️ Architecture

```
credit-risk-api/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── credit_data.csv      # Training data (generated)
├── models/
│   ├── credit_model.pkl     # Trained XGBoost model
│   ├── shap_explainer.pkl   # SHAP TreeExplainer
│   └── feature_names.pkl    # Feature list
├── api/
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic models
├── dashboard/
│   └── app.py               # Streamlit frontend
├── train_model.py           # Full training pipeline
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-api.git
cd credit-risk-api
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```bash
python data/generate_data.py   # Create synthetic dataset
python train_model.py          # Train XGBoost + save artifacts
```

Expected output:
```
XGBoost AUC:  0.8700+   ← Final model
Saved: credit_model.pkl, shap_explainer.pkl, feature_names.pkl
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://127.0.0.1:8000/docs** for interactive Swagger UI.

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Visit **http://localhost:8501**

---

## 📡 API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/health`      | Health check + model status        |
| GET    | `/model-info`  | Model metadata and feature list    |
| POST   | `/score`       | Score a single applicant           |
| POST   | `/score/batch` | Score up to 100 applicants at once |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "revolving_utilization": 0.35,
    "age": 42,
    "num_30_59_days_late": 0,
    "debt_ratio": 0.38,
    "monthly_income": 4500,
    "num_open_credit_lines": 7,
    "num_90_days_late": 0,
    "num_real_estate_loans": 1,
    "num_60_89_days_late": 0,
    "num_dependents": 2,
    "fico_score": 685
  }'
```

### Example Response

```json
{
  "probability_of_default": 0.0731,
  "risk_score": 927,
  "risk_tier": "Low Risk",
  "risk_tier_color": "#1A7A4A",
  "decision": "Approve",
  "fico_band": "Good",
  "fico_label": "670–739",
  "top_risk_factors": [
    {
      "feature": "Revolving Credit Utilization",
      "value": 0.35,
      "shap_value": 0.042,
      "direction": "increases_risk"
    }
  ],
  "top_protective_factors": [
    {
      "feature": "FICO Score",
      "value": 685.0,
      "shap_value": -0.089,
      "direction": "decreases_risk"
    }
  ],
  "model_version": "1.0.0"
}
```

---

## 🐳 Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

---

## 📊 Model Performance

| Metric            | Logistic Regression | XGBoost (Final) |
|-------------------|---------------------|-----------------|
| ROC-AUC           | ~0.78               | ~0.87           |
| Precision (Default)| 0.62               | 0.74            |
| Recall (Default)  | 0.68                | 0.71            |

---

## 🔑 Key Features

- **SMOTE** balancing for imbalanced credit data
- **SHAP** TreeExplainer for decision transparency
- **Pydantic v2** validation on all inputs
- **FICO bucketing** (Poor / Fair / Good / Very Good / Exceptional)
- **Docker** ready for production deployment
- **Batch scoring** endpoint for portfolio-level assessment

---

## 🧠 Skills Demonstrated

- Machine Learning (XGBoost, Scikit-learn)
- Credit Risk Analysis & FICO Scoring
- Model Explainability (SHAP)
- REST API Development (FastAPI)
- Data Visualization (Streamlit + Plotly)
- ML Engineering (Docker, model serialization)

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built with: Python · XGBoost · SHAP · FastAPI · Streamlit · Plotly · Docker*
