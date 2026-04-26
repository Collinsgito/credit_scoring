"""
train_model.py
--------------
Full pipeline: load data → clean → feature engineering → SMOTE →
train XGBoost → evaluate → SHAP → save model.

Run: python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_PATH   = ROOT / "data" / "credit_data.csv"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "SeriousDeliquency2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "FICOScore",
    # engineered below
    "FICO_Band_enc",
    "DebtToIncomeRatio",
    "TotalLatePayments",
]

FICO_BANDS = {
    "Exceptional":  (800, 850),
    "Very Good":    (740, 799),
    "Good":         (670, 739),
    "Fair":         (580, 669),
    "Poor":         (300, 579),
}


def fico_bucket(score):
    for label, (lo, hi) in FICO_BANDS.items():
        if lo <= score <= hi:
            return label
    return "Unknown"


FICO_ENC = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Exceptional": 4, "Unknown": 2}


def load_and_clean(path: Path) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,} | Columns: {df.shape[1]}")

    # Impute missing values
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df["NumberOfDependents"].fillna(df["NumberOfDependents"].median(), inplace=True)

    # Cap extreme revolving utilization at 1.5
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1.5)

    # Remove nonsensical late payment values (> 90 is data error)
    for col in ["NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"]:
        df[col] = df[col].clip(0, 20)

    print(f"  Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    # FICO bucket (categorical → encoded)
    df["FICO_Band"] = df["FICOScore"].apply(fico_bucket)
    df["FICO_Band_enc"] = df["FICO_Band"].map(FICO_ENC)

    # Debt-to-income ratio
    safe_income = df["MonthlyIncome"].replace(0, 1)
    df["DebtToIncomeRatio"] = (df["DebtRatio"] * safe_income) / safe_income

    # Total late payment incidents
    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    return df


def split_and_balance(df: pd.DataFrame):
    print("Splitting and balancing classes...")
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]
    y = df[TARGET]

    print(f"  Class distribution: {dict(y.value_counts())}")
    print(f"  Default rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=5)
    # Keep explicit indexing to satisfy type checkers that model alternate return shapes.
    resampled = smote.fit_resample(X_train, y_train)
    X_res = resampled[0]
    y_res = resampled[1]
    print(f"  After SMOTE → X_train: {X_res.shape}, positives: {y_res.sum():,}")
    return X_res, X_test, y_res, y_test, available


def train_baseline(X_train, y_train):
    print("\nTraining Logistic Regression baseline...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  LR CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost classifier...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_test, y_test, model_name="XGBoost", feature_names=None):
    print(f"\nEvaluating {model_name}...")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    preds = (proba >= 0.5).astype(int)

    print(f"  ROC-AUC: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=["Non-Default", "Default"]))

    # Save ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=model_name)
    axes[0].set_title(f"ROC Curve — {model_name}")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name=model_name)
    axes[1].set_title(f"Precision-Recall — {model_name}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "evaluation_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved evaluation curves → models/evaluation_curves.png")

    return auc


def generate_shap(model, X_test, feature_names):
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])  # sample 500 for speed

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved SHAP plot → models/shap_importance.png")

    # Save explainer
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    print("  Saved SHAP explainer → models/shap_explainer.pkl")
    return explainer


def save_artifacts(model, feature_names, fico_enc):
    print("\nSaving artifacts...")
    joblib.dump(model,        MODELS_DIR / "credit_model.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(fico_enc,     MODELS_DIR / "fico_enc.pkl")
    print("  Saved: credit_model.pkl, feature_names.pkl, fico_enc.pkl")


def main():
    print("=" * 60)
    print("  CREDIT RISK MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Generate data if missing
    if not DATA_PATH.exists():
        print("Dataset not found. Generating synthetic data...")
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "data" / "generate_data.py")])

    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_balance(df)

    lr_model  = train_baseline(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    lr_auc  = evaluate_model(lr_model,  X_test, y_test, "Logistic Regression", feature_names)
    xgb_auc = evaluate_model(xgb_model, X_test, y_test, "XGBoost",             feature_names)

    print(f"\n{'='*40}")
    print(f"  Logistic Regression AUC: {lr_auc:.4f}")
    print(f"  XGBoost AUC:             {xgb_auc:.4f}  ← FINAL MODEL")
    print(f"{'='*40}")

    generate_shap(xgb_model, X_test, feature_names)
    save_artifacts(xgb_model, feature_names, FICO_ENC)

    print("\n✅ Training complete. All artifacts saved to /models/")


if __name__ == "__main__":
    main()
