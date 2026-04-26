import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/cs-training.csv')
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].apply(lambda c: c.fillna(c.median()))

target_col = 'SeriousDlqin2yrs'
risk_pct = 0.6 * df['RevolvingUtilizationOfUnsecuredLines'].rank(pct=True) + 0.4 * df['DebtRatio'].rank(pct=True)
df['fico_proxy'] = (850 - (risk_pct * 550)).clip(300, 850)
df['fico_bucket'] = pd.cut(df['fico_proxy'], bins=[300,580,670,740,800,851], labels=['Poor','Fair','Good','Very Good','Exceptional'], right=False)

X = df.drop(columns=[target_col, 'Unnamed: 0']) if 'Unnamed: 0' in df.columns else df.drop(columns=[target_col])
X = pd.get_dummies(X, drop_first=False)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f'ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}')
print(f'F1: {f1_score(y_test, y_pred):.4f}')
print('Shapes:', X_train.shape, X_test.shape, X_res.shape, y_res.shape)
