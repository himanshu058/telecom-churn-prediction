import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------------------------------------
# PATH SETUP — works from any directory
# Path(__file__) = this file's location (app/streamlit_app.py)
# .parent        = app/ folder
# .parent.parent = project root (telecom-churn-prediction/)
# -------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "xgboost_churn.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
TRAIN_PATH  = BASE_DIR / "data" / "processed" / "X_train.csv"

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_feature_names():
    # Reads only the header row (nrows=0) — fast and memory efficient
    return pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

model         = load_model()
scaler        = load_scaler()
feature_names = load_feature_names()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.title("About This Project")
    st.write("Predicts whether a telecom customer will churn "
             "based on their account details.")
    st.divider()
    st.write("**Model:** XGBoost Classifier")
    st.write("**Dataset:** IBM Telco (7,032 customers)")
    st.write("**Features:** 33")
    st.write("**ROC-AUC:** ~0.834")
    st.write("**Accuracy:** ~78%")
    st.divider()
    st.write("**Made by:** Himanshu")

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------
st.title("📱 Telecom Customer Churn Predictor")
st.write("Fill in the customer details and click **Predict Churn**.")
st.divider()

# -------------------------------------------------------
# INPUT FORM — 3 columns
# -------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input(
        "Monthly Charges ($)", 18.25, 118.75, 65.0, step=0.5)
    total_charges   = st.number_input(
        "Total Charges ($)", 0.0, 8684.8, 780.0, step=10.0)

with col2:
    st.subheader("Services")
    contract        = st.selectbox("Contract Type",
                          ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service",
                          ["DSL", "Fiber optic", "No"])
    multiple_lines  = st.selectbox("Multiple Lines",
                          ["No", "Yes", "No phone service"])
    payment         = st.selectbox("Payment Method", [
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"
                      ])
    online_security = st.selectbox("Online Security",
                          ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",
                          ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",
                          ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Personal Info")
    gender        = st.selectbox("Gender", ["Male", "Female"])
    senior        = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner       = st.selectbox("Has Partner", ["Yes", "No"])
    dependents    = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless     = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):

    # Start with zeros for every one of the 33 features
    input_dict = {col: 0 for col in feature_names}

    # ---- Base numeric and binary features ----
    input_dict['tenure']           = tenure
    input_dict['MonthlyCharges']   = monthly_charges
    input_dict['TotalCharges']     = total_charges
    input_dict['SeniorCitizen']    = 1 if senior    == "Yes"  else 0
    input_dict['gender']           = 1 if gender    == "Male" else 0
    input_dict['Partner']          = 1 if partner   == "Yes"  else 0
    input_dict['Dependents']       = 1 if dependents == "Yes" else 0
    input_dict['PhoneService']     = 1 if phone_service == "Yes" else 0
    input_dict['PaperlessBilling'] = 1 if paperless == "Yes"  else 0

    # ---- Engineered features ----
    # MEDIAN_MONTHLY_CHARGE = 70.35 from your actual dataset
    input_dict['avg_charge_per_month'] = total_charges / (tenure + 1)
    input_dict['high_charges']         = 1 if monthly_charges > 70.35 else 0

    if   tenure <= 12: input_dict['tenure_group'] = 0
    elif tenure <= 36: input_dict['tenure_group'] = 1
    elif tenure <= 60: input_dict['tenure_group'] = 2
    else:              input_dict['tenure_group'] = 3

    # ---- MultipleLines ----
    # Base category dropped by get_dummies = 'No'
    # So when input is 'No', both encoded cols stay 0
    if   multiple_lines == "No phone service":
        input_dict['MultipleLines_No phone service'] = 1
    elif multiple_lines == "Yes":
        input_dict['MultipleLines_Yes'] = 1

    # ---- InternetService ----
    # Base category dropped = 'DSL'
    # When input is 'DSL', both encoded cols stay 0
    if   internet == "Fiber optic":
        input_dict['InternetService_Fiber optic'] = 1
    elif internet == "No":
        input_dict['InternetService_No'] = 1

    # ---- OnlineSecurity ----
    # Base category dropped = 'No'
    if   online_security == "No internet service":
        input_dict['OnlineSecurity_No internet service'] = 1
    elif online_security == "Yes":
        input_dict['OnlineSecurity_Yes'] = 1

    # ---- OnlineBackup ----
    # Base category dropped = 'No'
    if   online_backup == "No internet service":
        input_dict['OnlineBackup_No internet service'] = 1
    elif online_backup == "Yes":
        input_dict['OnlineBackup_Yes'] = 1

    # ---- DeviceProtection ----
    # Not shown in form — defaults to base category 'No' (both cols = 0)
    # This is intentional to keep the form simple

    # ---- TechSupport ----
    # Base category dropped = 'No'
    if   tech_support == "No internet service":
        input_dict['TechSupport_No internet service'] = 1
    elif tech_support == "Yes":
        input_dict['TechSupport_Yes'] = 1

    # ---- StreamingTV and StreamingMovies ----
    # Not shown in form — defaults to base category 'No' (both cols = 0)

    # ---- Contract ----
    # Base category dropped = 'Month-to-month'
    # When input is 'Month-to-month', both encoded cols stay 0
    if   contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1

    # ---- PaymentMethod ----
    # Base category dropped = 'Bank transfer (automatic)'
    # When input is 'Bank transfer', all 3 encoded cols stay 0
    if   payment == "Credit card (automatic)":
        input_dict['PaymentMethod_Credit card (automatic)'] = 1
    elif payment == "Electronic check":
        input_dict['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check":
        input_dict['PaymentMethod_Mailed check'] = 1

    # ---- Build DataFrame in exact training column order ----
    input_df = pd.DataFrame([input_dict])[feature_names]

    # ---- Scale the 4 numeric columns ----
    num_cols      = ['tenure', 'MonthlyCharges', 'TotalCharges',
                     'avg_charge_per_month']
    cols_to_scale = [c for c in num_cols if c in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # ---- Predict ----
    churn_prob = float(model.predict_proba(input_df)[0][1])
    churn_pred = int(model.predict(input_df)[0])

    # -------------------------------------------------------
    # DISPLAY RESULT
    # -------------------------------------------------------
    st.subheader("Prediction Result")
    r1, r2 = st.columns(2)

    with r1:
        if churn_pred == 1:
            st.error("⚠️ HIGH CHURN RISK")
        else:
            st.success("✅ LOW CHURN RISK")
        st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")

    with r2:
        st.write("**Risk Level Gauge**")
        st.progress(churn_prob)
        if   churn_prob > 0.70:
            st.write("🔴 Very High — Immediate action needed")
        elif churn_prob > 0.50:
            st.write("🟠 High — Consider intervention")
        elif churn_prob > 0.30:
            st.write("🟡 Medium — Monitor this customer")
        else:
            st.write("🟢 Low — Customer likely to stay")

    st.divider()
    st.subheader("Business Recommendation")
    if churn_prob > 0.50:
        st.info("""
        This customer is at high risk. Suggested actions:
        - Offer a discounted rate or loyalty bonus
        - Switch to a One year or Two year contract with a discount
        - Proactive customer support call within 48 hours
        - Offer a free upgrade on a service not currently used
        """)
    else:
        st.info("""
        This customer is likely to stay. Suggested actions:
        - Upsell StreamingTV, OnlineBackup, or TechSupport
        - Offer a referral bonus for bringing new customers
        - Enroll in a loyalty rewards program
        """)    