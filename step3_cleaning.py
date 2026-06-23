import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)

df = pd.read_csv('data/raw/telecom_churn.csv')
print("Original shape:", df.shape)
# (7043, 21)

# -------------------------------------------------------
# NOTE: Your file has NO unnamed columns — confirmed from analysis
# TotalCharges has 11 rows with blank spaces (not NaN)
# These are customers with tenure=0 who just joined
# -------------------------------------------------------

# FIX 1: Convert TotalCharges from text to number
# The 11 blank space rows become NaN via errors='coerce'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
dropped = df['TotalCharges'].isnull().sum()
print(f"Rows with blank TotalCharges (will be dropped): {dropped}")
df = df.dropna(subset=['TotalCharges'])
print("Shape after drop:", df.shape)
# (7032, 21)

# FIX 2: Drop customerID
df = df.drop('customerID', axis=1)

# FIX 3: Encode target — Yes=1, No=0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# FIX 4: Encode binary Yes/No columns as 1/0
for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df[col] = (df[col] == 'Yes').astype(int)
df['gender'] = (df['gender'] == 'Male').astype(int)
# SeniorCitizen is already 0/1 — leave it

# FIX 5: One-hot encode multi-category columns
# dtype=int is CRITICAL — prevents bool dtype in pandas 3.x
# bool dtype causes SMOTE to crash with "boolean subtract" error
# drop_first=True drops base category from each column:
#   MultipleLines: drops 'No'     → keeps: No phone service, Yes
#   InternetService: drops 'DSL'  → keeps: Fiber optic, No
#   OnlineSecurity: drops 'No'    → keeps: No internet service, Yes
#   OnlineBackup: drops 'No'      → keeps: No internet service, Yes
#   DeviceProtection: drops 'No'  → keeps: No internet service, Yes
#   TechSupport: drops 'No'       → keeps: No internet service, Yes
#   StreamingTV: drops 'No'       → keeps: No internet service, Yes
#   StreamingMovies: drops 'No'   → keeps: No internet service, Yes
#   Contract: drops 'Month-to-month' → keeps: One year, Two year
#   PaymentMethod: drops 'Bank transfer (automatic)' → keeps: Credit card, Electronic check, Mailed check
cat_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# VERIFY
print("\nFinal shape:", df.shape)
# Must be (7032, 31)

print("Missing values:", df.isnull().sum().sum())
# Must be 0

print("Dtypes:")
print(df.dtypes.value_counts())
# Must show only int64 and float64

bool_check = df.select_dtypes(include='bool').columns.tolist()
obj_check  = df.select_dtypes(include='object').columns.tolist()
print("Bool columns (must be empty):", bool_check)
print("Object columns (must be empty):", obj_check)

if not bool_check and not obj_check:
    print("\nALL CLEAR — ready for next step")
else:
    print("\nWARNING — fix dtype issues before continuing")

print("\nAll 31 column names (confirmed from your dataset):")
for i, col in enumerate(df.columns):
    print(f"  {i+1:02d}. {col}")

df.to_csv('data/processed/churn_cleaned.csv', index=False)
print("\nSaved: data/processed/churn_cleaned.csv")