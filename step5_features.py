import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/processed/churn_cleaned.csv')
print("Loaded shape:", df.shape)
# (7032, 31)

# -------------------------------------------------------
# NEW FEATURE 1: Average charge per month
# Divides TotalCharges by tenure to get cost per month
# +1 avoids division by zero for tenure=0 customers
# -------------------------------------------------------
df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

# -------------------------------------------------------
# NEW FEATURE 2: Tenure group
# Groups customers by loyalty stage
# include_lowest=True ensures tenure=0 maps to group 0
# (without it, tenure=0 becomes NaN because bins are left-exclusive)
# fillna(0) handles any remaining edge case safely
# -------------------------------------------------------
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 36, 60, 100],
    labels=[0, 1, 2, 3],
    include_lowest=True
).fillna(0).astype(int)

# -------------------------------------------------------
# NEW FEATURE 3: High charges flag
# Median MonthlyCharges from your actual dataset is 70.35
# -------------------------------------------------------
MEDIAN_MONTHLY_CHARGE = 70.35
df['high_charges'] = (df['MonthlyCharges'] > MEDIAN_MONTHLY_CHARGE).astype(int)

print("Shape after feature engineering:", df.shape)
# (7032, 34)

# -------------------------------------------------------
# SPLIT FEATURES AND TARGET
# -------------------------------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

print("Features:", X.shape)
print("Target churn rate:", round(y.mean() * 100, 2), "%")

# 80% train, 20% test
# stratify=y keeps same 26.54% churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
# X_train: (5625, 33)  X_test: (1407, 33)
print(f"Train churn %: {y_train.mean()*100:.2f}")
print(f"Test churn  %: {y_test.mean()*100:.2f}")

# -------------------------------------------------------
# SCALE NUMERIC COLUMNS
# StandardScaler makes mean=0, std=1 for each column
# fit_transform ONLY on train — prevents data leakage
# transform ONLY on test — never fit on test
# -------------------------------------------------------
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month']
scaler   = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# -------------------------------------------------------
# CRITICAL FINAL STEP: Convert everything to float64
# This is the single most reliable fix for the SMOTE crash
# astype(float) converts int64, bool, and any other type to float64
# SMOTE requires pure float64 input
# -------------------------------------------------------
X_train = X_train.astype(float)
X_test  = X_test.astype(float)
y_train = y_train.astype(int)
y_test  = y_test.astype(int)

print("\nDtype check (must be all float64):")
print(X_train.dtypes.value_counts())
print("NaN in X_train:", X_train.isnull().sum().sum())
print("NaN in X_test: ", X_test.isnull().sum().sum())

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("\nScaler saved: models/scaler.pkl")

# Save all 4 CSV files for next steps
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv( 'data/processed/X_test.csv',  index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv( 'data/processed/y_test.csv',  index=False)
print("4 CSV files saved in data/processed/")

print("\nFinal feature names (33 total):")
print(X_train.columns.tolist())