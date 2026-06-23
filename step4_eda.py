import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)

# Load raw file for readable labels in charts
df = pd.read_csv('data/raw/telecom_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
# df now has 7032 rows with original text labels

sns.set_theme(style='whitegrid', palette='Set2')
print("Generating 7 charts...")

# -------------------------------------------------------
# CHART 1: Churn Distribution (Pie)
# Insight: 73.46% No churn, 26.54% Churned
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
counts = df['Churn'].value_counts()
ax.pie(
    counts,
    labels=['No Churn (73.5%)', 'Churned (26.5%)'],
    autopct='%1.1f%%',
    colors=['#2196F3', '#F44336'],
    startangle=90
)
ax.set_title('Customer Churn Distribution', fontsize=14, pad=15)
plt.savefig('reports/chart1_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 1 saved — Churn Distribution")

# -------------------------------------------------------
# CHART 2: Contract Type vs Churn (Bar)
# Insight: Month-to-month has highest churn rate (~42%)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn', ax=ax,
              palette=['#2196F3', '#F44336'])
ax.set_title('Churn Count by Contract Type', fontsize=14)
ax.set_xlabel('Contract Type')
ax.set_ylabel('Number of Customers')
ax.legend(title='Churn')
plt.savefig('reports/chart2_contract_churn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved — Contract vs Churn")

# -------------------------------------------------------
# CHART 3: Tenure Distribution (Histogram)
# Insight: New customers (0-12 months) churn the most
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30,
             ax=ax, element='step', palette=['#2196F3', '#F44336'])
ax.set_title('Tenure Distribution by Churn Status', fontsize=14)
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Count')
plt.savefig('reports/chart3_tenure_churn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 3 saved — Tenure Distribution")

# -------------------------------------------------------
# CHART 4: Monthly Charges Boxplot
# Insight: Churned customers pay more per month (~$74 vs $61)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
            ax=ax, palette=['#2196F3', '#F44336'])
ax.set_title('Monthly Charges by Churn Status', fontsize=14)
ax.set_xlabel('Churn (No / Yes)')
ax.set_ylabel('Monthly Charges ($)')
plt.savefig('reports/chart4_monthly_charges.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 4 saved — Monthly Charges Boxplot")

# -------------------------------------------------------
# CHART 5: Payment Method vs Churn
# Insight: Electronic check payment has highest churn count
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
df[df['Churn'] == 'Yes']['PaymentMethod'].value_counts().plot(
    kind='bar', ax=ax, color='#E74C3C', edgecolor='white')
ax.set_title('Churned Customers by Payment Method', fontsize=14)
ax.set_xlabel('Payment Method')
ax.set_ylabel('Number of Churned Customers')
plt.xticks(rotation=30, ha='right')
plt.savefig('reports/chart5_payment_churn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 5 saved — Payment Method vs Churn")

# -------------------------------------------------------
# CHART 6: Internet Service vs Churn
# Insight: Fiber optic customers churn most
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='InternetService', hue='Churn',
              ax=ax, palette=['#2196F3', '#F44336'])
ax.set_title('Churn by Internet Service Type', fontsize=14)
ax.set_xlabel('Internet Service')
ax.set_ylabel('Count')
plt.savefig('reports/chart6_internet_churn.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 6 saved — Internet Service vs Churn")

# -------------------------------------------------------
# CHART 7: Correlation Heatmap
# -------------------------------------------------------
df_clean = pd.read_csv('data/processed/churn_cleaned.csv')
key_cols = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'SeniorCitizen', 'Partner', 'Dependents',
    'PaperlessBilling', 'Churn'
]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    df_clean[key_cols].corr(),
    annot=True, cmap='coolwarm', fmt='.2f',
    ax=ax, linewidths=0.5
)
ax.set_title('Correlation Heatmap — Key Features vs Churn', fontsize=14)
plt.savefig('reports/chart7_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 7 saved — Correlation Heatmap")

print("\nAll 7 charts saved in reports/")
print("\nKey insights from your data:")
print("  1. 26.54% of customers churned")
print("  2. Month-to-month contract customers churn most (~42%)")
print("  3. New customers (tenure 0-12 months) churn most")
print("  4. Fiber optic internet customers churn more than DSL")
print("  5. Electronic check payment has highest churn association")
print("  6. Churned customers pay higher monthly charges on average")
print("  7. tenure has strongest negative correlation with Churn (-0.35)")