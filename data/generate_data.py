"""
generate_data.py
----------------
Generates a realistic synthetic credit dataset (10,000 rows).
Based on the 'Give Me Some Credit' feature structure from Kaggle.
Run this if you don't have the Kaggle dataset:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10_000

def generate_dataset(n=N):
    # --- Base features ---
    age = np.random.normal(loc=45, scale=12, size=n).clip(21, 90).astype(int)

    # Monthly income — log-normal, some NaN
    monthly_income = np.random.lognormal(mean=8.5, sigma=0.7, size=n).clip(500, 50_000)
    monthly_income[np.random.rand(n) < 0.08] = np.nan  # 8% missing

    # Number of dependents
    num_dependents = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.35, 0.25, 0.20, 0.10, 0.07, 0.03])
    num_dependents = num_dependents.astype(float)
    num_dependents[np.random.rand(n) < 0.02] = np.nan  # 2% missing

    # Revolving utilization — between 0 and 1+
    revolving_utilization = np.random.beta(a=2, b=5, size=n)
    revolving_utilization[np.random.rand(n) < 0.05] *= 10  # outliers above 1

    # Number of open credit lines
    num_open_credit_lines = np.random.poisson(lam=8, size=n).clip(0, 30)

    # Number of real estate loans
    num_real_estate_loans = np.random.choice([0, 1, 2, 3], size=n, p=[0.55, 0.30, 0.12, 0.03])

    # Past due entries
    num_30_59_days_late = np.random.choice(range(6), size=n, p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01])
    num_60_89_days_late = np.random.choice(range(5), size=n, p=[0.82, 0.10, 0.05, 0.02, 0.01])
    num_90_days_late    = np.random.choice(range(5), size=n, p=[0.88, 0.07, 0.03, 0.01, 0.01])

    # Debt ratio
    debt_ratio = np.random.beta(a=1.5, b=3, size=n)

    # FICO-like score (300–850)
    # Correlated with utilization, late payments
    fico_base = 700 - (revolving_utilization * 120) - (num_90_days_late * 40) \
                    - (num_60_89_days_late * 25) - (num_30_59_days_late * 15)
    fico_score = (fico_base + np.random.normal(0, 30, n)).clip(300, 850).astype(int)

    # --- Target variable (SeriousDeliquency) ---
    # Logistic model to create realistic default rates (~6.7%)
    log_odds = (
        -4.5
        + revolving_utilization * 2.0
        + num_90_days_late * 1.5
        + num_60_89_days_late * 1.0
        + num_30_59_days_late * 0.7
        + debt_ratio * 1.2
        - (fico_score - 600) / 200
        - (age - 40) / 80
        + np.random.normal(0, 0.5, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    serious_deliquency = (np.random.rand(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "SeriousDeliquency2yrs":       serious_deliquency,
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age":                         age,
        "NumberOfTime30-59DaysPastDueNotWorse": num_30_59_days_late,
        "DebtRatio":                   debt_ratio,
        "MonthlyIncome":               monthly_income,
        "NumberOfOpenCreditLinesAndLoans": num_open_credit_lines,
        "NumberOfTimes90DaysLate":     num_90_days_late,
        "NumberRealEstateLoansOrLines": num_real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": num_60_89_days_late,
        "NumberOfDependents":          num_dependents,
        "FICOScore":                   fico_score,
    })

    return df


def main():
    out_path = Path(__file__).parent / "credit_data.csv"
    print(f"Generating {N:,} synthetic credit records...")
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    print(f"\nClass distribution:")
    print(df["SeriousDeliquency2yrs"].value_counts())
    print(f"\nDefault rate: {df['SeriousDeliquency2yrs'].mean():.2%}")
    print(f"\nShape: {df.shape}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
