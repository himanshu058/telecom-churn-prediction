import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
X_train = pd.read_csv('data/processed/X_train.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

# Force correct dtypes — safety conversion
X_train = X_train.astype(float)
X_test  = X_test.astype(float)
y_train = y_train.astype(int)
y_test  = y_test.astype(int)

print("X_train:", X_train.shape)
print("Dtypes:", X_train.dtypes.value_counts().to_dict())
print("NaN:", X_train.isnull().sum().sum())
print("Churn rate in train:", round(y_train.mean() * 100, 2), "%")

# -------------------------------------------------------
# SMOTE — fixes 73.46% / 26.54% class imbalance
# Creates synthetic minority (churn=1) samples
# Result: equal number of churn and no-churn in training
# NEVER apply SMOTE to test data
# -------------------------------------------------------
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {X_bal.shape}")
print("Class balance:", pd.Series(y_bal).value_counts().to_dict())
# {0: 4130, 1: 4130} — perfectly balanced

# -------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION — simple baseline
# -------------------------------------------------------
print("\n--- Training Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_bal, y_bal)
print("Complete")

# -------------------------------------------------------
# MODEL 2: RANDOM FOREST — ensemble of decision trees
# -------------------------------------------------------
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_bal, y_bal)
print("Complete")

# -------------------------------------------------------
# MODEL 3: XGBOOST — gradient boosting, best for tabular data
# use_label_encoder is NOT included — removed in XGBoost 2.0+
# -------------------------------------------------------
print("\n--- Training XGBoost ---")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_bal, y_bal)
print("Complete")

# -------------------------------------------------------
# EVALUATE ALL 3 MODELS
# -------------------------------------------------------
models = {
    'Logistic Regression': lr,
    'Random Forest':       rf,
    'XGBoost':             xgb
}

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, y_pred, target_names=['No Churn', 'Churned']))
    print(f"ROC-AUC: {auc:.4f}")

# Expected results from your dataset:
# Logistic Regression AUC: ~0.836
# Random Forest AUC:       ~0.834
# XGBoost AUC:             ~0.834

# -------------------------------------------------------
# ROC CURVE COMPARISON CHART
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
colors  = ['#2196F3', '#4CAF50', '#F44336']
for (name, model), color in zip(models.items(), colors):
    y_prob      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
            color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — All Models Comparison', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('reports/roc_curve_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nROC curve saved")

# -------------------------------------------------------
# CONFUSION MATRIX — XGBOOST
# -------------------------------------------------------
y_pred_xgb = xgb.predict(X_test)
cm   = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churned']
)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — XGBoost', fontsize=14)
plt.savefig('reports/confusion_matrix_xgboost.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved")

# -------------------------------------------------------
# SAVE ALL 3 MODELS
# -------------------------------------------------------
joblib.dump(lr,  'models/logistic_regression.pkl')
joblib.dump(rf,  'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost_churn.pkl')
print("\nAll 3 models saved in models/ folder")