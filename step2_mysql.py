import pandas as pd
import mysql.connector

df = pd.read_csv('data/raw/telecom_churn.csv')
print("Loaded:", df.shape)

# Fix TotalCharges — 11 rows have blank spaces, convert those to NaN then drop
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
print("After dropping 11 blank TotalCharges rows:", df.shape)
# Should show (7032, 21)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Bisht@2006",
    database="telecom_churn_db"
)
cursor = conn.cursor()

insert_cols = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

for _, row in df[insert_cols].iterrows():
    cursor.execute(
        """INSERT IGNORE INTO customers VALUES
           (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        tuple(row)
    )

conn.commit()
print("Data loaded successfully. Rows:", cursor.rowcount)
cursor.close()
conn.close()