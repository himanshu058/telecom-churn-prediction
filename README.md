# Telecom Customer Churn Prediction

## Project Overview
End-to-end machine learning project predicting whether a telecom customer
will churn using the IBM Telco Customer Churn dataset.

Built a complete project covering the complete data science
pipeline from data collection to model deployment.

## Dataset
- Source: Kaggle — IBM Telco Customer Churn (blastchar)
- Original size: 7,043 customers, 21 features
- After cleaning: 7,032 rows, 33 features used for training
- Target: Churn column (Yes = churned, No = stayed)
- Class distribution: 26.54% churned, 73.46% stayed

## Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | EDA visualization |
| Scikit-learn | Preprocessing and baseline models |
| XGBoost | Best performing model |
| SMOTE (imbalanced-learn) | Class imbalance handling |
| SHAP | Model explainability |
| MySQL | Data storage and querying |
| Power BI | Business intelligence dashboard |
| Streamlit | Web application deployment |
| GitHub | Version control |

## Model Performance
| Model | Accuracy | Precision | Recall | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 79% | 0.60 | 0.60 | 0.60 | 0.836 |
| Random Forest | 79% | 0.61 | 0.56 | 0.58 | 0.834 |
| XGBoost | 78% | 0.58 | 0.65 | 0.61 | 0.834 |

XGBoost selected as final model — highest Recall for churned class.

## Key Findings from EDA
- 26.54% of 7,032 customers churned
- Month-to-month contract customers have 42% churn rate
- Customers with tenure under 12 months churn the most
- Fiber optic internet service linked to higher churn than DSL
- Electronic check payment method has highest churn association
- Churned customers pay higher average monthly charges

## Top Features by SHAP Importance
1. tenure (most important — low tenure = high churn risk)
2. Contract_Month-to-month
3. MonthlyCharges
4. InternetService_Fiber optic
5. TotalCharges

## Project Structure

telecom-churn-prediction/

├── app/streamlit_app.py       web application
├── data/raw/                  original CSV dataset
├── data/processed/            cleaned and split CSV files
├── models/                    saved pkl model files
├── reports/                   charts, SHAP plots, dashboard
├── step1_understand.py        data inspection
├── step2_mysql.py             MySQL data loading
├── step3_cleaning.py          data cleaning pipeline
├── step4_eda.py               exploratory data analysis
├── step5_features.py          feature engineering and splitting
├── step6_models.py            model training and evaluation
├── step7_shap.py              SHAP explainability
├── requirements.txt           all Python dependencies
└── README.md

## How to Run

1. Clone
git clone https://github.com/himanshu058/telecom-churn-prediction.git
cd telecom-churn-prediction

2. Install dependencies
pip install -r requirements.txt

3. Add dataset to data/raw/telecom_churn.csv

4. Run pipeline in order
python step3_cleaning.py
python step5_features.py
python step6_models.py
python step7_shap.py

5. Launch web app
streamlit run app/streamlit_app.py

## Author
Name: Himanshu Singh Bisht
