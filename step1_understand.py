import pandas as pd

df = pd.read_csv('data/raw/telecom_churn.csv')

print("=" * 55)
print("SHAPE:", df.shape)
# Your file: (7043, 21) — exactly 21 columns, no unnamed columns
print("=" * 55)

print("\nALL COLUMN NAMES:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:02d}. {repr(col)}")

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES (isnull check):")
print(df.isnull().sum())

print("\nTOTALCHARGES — checking for hidden blank spaces:")
blank_tc = df[df['TotalCharges'].str.strip() == '']
print(f"  Rows with blank TotalCharges: {len(blank_tc)}")
# Your file has 11 blank rows — these are customers with tenure=0

print("\nCHURN DISTRIBUTION:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).mul(100).round(2))

print("\nTENURE STATS:")
print(f"  Min: {df['tenure'].min()} | Max: {df['tenure'].max()}")
print(f"  Rows with tenure=0: {(df['tenure']==0).sum()}")

print("\nMONTHLY CHARGES:")
print(f"  Min: {df['MonthlyCharges'].min()}")
print(f"  Max: {df['MonthlyCharges'].max()}")
print(f"  Median: {df['MonthlyCharges'].median()}")