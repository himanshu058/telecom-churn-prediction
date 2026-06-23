import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('reports', exist_ok=True)

xgb    = joblib.load('models/xgboost_churn.pkl')
X_test = pd.read_csv('data/processed/X_test.csv').astype(float)

X_sample = X_test.head(500).reset_index(drop=True)

print("Calculating SHAP values — please wait 30 to 60 seconds...")
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)
print("Done. Shape:", shap_values.shape)

# -------------------------------------------------------
# PLOT 1: SUMMARY BEESWARM
# Each dot = one customer
# X position = how much it pushed prediction (right = toward churn)
# Color = Red means high feature value, Blue means low
# Features sorted top to bottom by importance
# -------------------------------------------------------
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary — Feature Impact on Churn', fontsize=13)
plt.savefig('reports/shap_plot1_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plot 1 saved")

# -------------------------------------------------------
# PLOT 2: BAR CHART
# Average absolute SHAP value for each feature
# Longer bar = more important overall
# -------------------------------------------------------
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=13)
plt.savefig('reports/shap_plot2_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plot 2 saved")

# -------------------------------------------------------
# PLOT 3: WATERFALL FOR ONE CUSTOMER
# Explains exactly why Customer 0 got their prediction
# Base value = average prediction across all customers
# Each bar shows how much each feature changed that base value
# -------------------------------------------------------
exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_sample.iloc[0],
    feature_names=X_sample.columns.tolist()
)
plt.figure(figsize=(11, 7))
shap.waterfall_plot(exp, show=False, max_display=15)
plt.title('SHAP Waterfall — Why Was Customer 1 Predicted This Way?', fontsize=12)
plt.savefig('reports/shap_plot3_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plot 3 saved")

# -------------------------------------------------------
# PLOT 4: DEPENDENCE PLOT FOR TENURE
# Shows how SHAP value of tenure changes across different tenure values
# You will see: low tenure = high positive SHAP (pushes toward churn)
#               high tenure = negative SHAP (pushes away from churn)
# -------------------------------------------------------
plt.figure(figsize=(9, 6))
shap.dependence_plot('tenure', shap_values, X_sample, show=False)
plt.title('SHAP Dependence — Tenure vs Its Impact on Churn', fontsize=13)
plt.savefig('reports/shap_plot4_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plot 4 saved")

# Print feature importance ranking
importance_df = pd.DataFrame({
    'Feature':    X_sample.columns,
    'Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features by SHAP Importance:")
print(importance_df.head(10).to_string(index=False))
print("\nAll 4 SHAP plots saved in reports/")