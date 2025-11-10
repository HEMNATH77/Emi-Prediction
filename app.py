import streamlit as st
import pandas as pd
import joblib
import os
import gdown
import traceback
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------
# ⚙️ CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(page_title="EMI Prediction App", layout="centered")

# 🌐 Google Drive File IDs
REG_MODEL_ID = "1ss1pyfzbbR_ItA7b84cqHslIfOcIQn7V"
CLS_MODEL_ID = "1KMOEefX8bbAPyiCe3q5xfqRvSVZczS7P"
CSV_ID = "1x24_0cCNRmWAiaX8_jJGk6HlIRgNUc2_"

# 📂 File Paths
REG_MODEL_PATH = "Final_Regression_Cp.pkl"
CLS_MODEL_PATH = "Final_Classification_Cp.pkl"
CSV_PATH = "cleaned_EMI.csv"

# ✅ Download Files from Google Drive if not present
if not os.path.exists(REG_MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={REG_MODEL_ID}", REG_MODEL_PATH, quiet=False)
if not os.path.exists(CLS_MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={CLS_MODEL_ID}", CLS_MODEL_PATH, quiet=False)
if not os.path.exists(CSV_PATH):
    gdown.download(f"https://drive.google.com/uc?id={CSV_ID}", CSV_PATH, quiet=False)

# -----------------------------------------------------------
# 🎨 STYLES
# -----------------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: white !important;
        }
        .main-title {
            text-align: center;
            color: #4a148c;
            font-size: 2.8rem;
            font-weight: 700;
            text-shadow: 0px 0px 8px rgba(74, 20, 140, 0.2);
        }
        .sub-header {
            text-align: center;
            color: #6a1b9a;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        .stButton button {
            background: linear-gradient(90deg, #8e2de2, #4a00e0) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #4a00e0, #8e2de2) !important;
        }
        h2, h3, h4, p, label {
            color: #311b92 !important;
        }
        div[data-testid="stMarkdownContainer"] {
            color: #311b92;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧭 PAGE NAVIGATION CONTROL
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# -----------------------------------------------------------
# 📊 LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv(CSV_PATH)
mode_values = df.mode().iloc[0]
median_values = df.median(numeric_only=True)

# -----------------------------------------------------------
# 🧾 INPUT PAGE
# -----------------------------------------------------------
if st.session_state.page == "input":
    st.markdown('<p class="main-title">💡 EMI Prediction Web App</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter your details below to check your EMI eligibility and prediction</p>', unsafe_allow_html=True)

    with st.container():
        st.subheader("📝 Enter Your Details")

        monthly_salary = st.number_input("💼 Monthly Salary", min_value=0.0, value=0.0, step=1000.0)
        house_type = st.selectbox("🏠 House Type", ["Rented", "Own", "Family"])
        school_fees = st.number_input("🏫 School Fees", min_value=0.0, value=0.0, step=500.0)
        college_fees = st.number_input("🎓 College Fees", min_value=0.0, value=0.0, step=500.0)
        groceries_utilities = st.number_input("🛒 Groceries & Utilities", min_value=0.0, value=0.0, step=500.0)
        existing_loans = st.selectbox("💳 Existing Loans", ["Yes", "No"])
        current_emi_amount = st.number_input("💰 Current EMI Amount", min_value=0.0, value=0.0, step=500.0)
        credit_score = st.number_input("📊 Credit Score", min_value=0.0, max_value=900.0, value=0.0, step=10.0)
        requested_amount = st.number_input("🏦 Requested Loan Amount", min_value=0.0, value=0.0, step=10000.0)
        bank_balance = st.number_input("🏧 Bank Balance", min_value=0.0, value=0.0, step=1000.0)
        requested_tenure = st.number_input("📅 Requested Tenure (months)", min_value=1, max_value=120, value=1, step=1)

        predict_button = st.button("🔍 Predict EMI")

    if predict_button:
        st.session_state.user_input = {
            "monthly_salary": monthly_salary,
            "house_type": house_type,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "groceries_utilities": groceries_utilities,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "requested_amount": requested_amount,
            "bank_balance": bank_balance,
            "requested_tenure": requested_tenure
        }
        st.session_state.page = "result"
        st.rerun()

# -----------------------------------------------------------
# 🎯 RESULT PAGE
# -----------------------------------------------------------
elif st.session_state.page == "result":
    st.markdown("<h2 style='text-align:center; color:#4a148c;'>🔮 EMI Prediction Result</h2>", unsafe_allow_html=True)

    if "user_input" not in st.session_state:
        st.warning("⚠️ Please go back and enter your details first.")
        if st.button("⬅️ Back"):
            st.session_state.page = "input"
            st.rerun()
    else:
        user_input = st.session_state.user_input

        input_data = pd.DataFrame([{
            col: (median_values[col] if col in median_values else mode_values[col])
            for col in df.columns if col not in ['emi_eligibility', 'max_monthly_emi']
        }])

        for key, val in user_input.items():
            input_data[key] = val

        cat_cols = ['gender', 'marital_status', 'education', 'employment_type',
                    'company_type', 'house_type', 'existing_loans', 'emi_scenario']

        try:
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    input_data[col] = le.transform(input_data[col].astype(str))

            reg = joblib.load(REG_MODEL_PATH)
            cls = joblib.load(CLS_MODEL_PATH)

            with st.spinner("Processing your prediction... ⏳"):
                eligibility_pred = cls.predict(input_data)[0]
                emi_pred = reg.predict(input_data)[0]

            eligibility_map = {0: ("✅ Eligible", "#00bfa5"), 1: ("⚠️ High Risk", "#ffb300"), 2: ("❌ Not Eligible", "#e53935")}
            status, color = eligibility_map.get(eligibility_pred, ("❓ Unknown", "#757575"))

            st.markdown(
                f"""
                <div style='background-color:#f3e5f5; padding:25px; border-radius:15px; text-align:center; box-shadow:0 0 10px rgba(74, 20, 140, 0.2);'>
                    <h3 style='color:{color}; font-weight:700;'>🏁 EMI Eligibility Result</h3>
                    <p style='font-size:1.3rem; font-weight:600; color:{color};'>
                        {status}
                    </p>
                    <hr style='border:1px solid #4a148c; margin:15px 0;'>
                    <h4 style='color:#4a148c;'>💰 Predicted EMI Amount</h4>
                    <p style='font-size:2rem; color:#00bfa5; font-weight:700;'>₹{emi_pred:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.balloons()

            if eligibility_pred == 0:
                st.success("🎉 Congratulations! You're eligible for the EMI. Manage your finances wisely!")
            elif eligibility_pred == 1:
                st.warning("⚠️ You're at moderate risk. Review your financial details carefully.")
            else:
                st.error("❌ Unfortunately, you're not eligible now. Try adjusting your requested amount or tenure.")

        except Exception as e:
            st.error("🚨 Error occurred while predicting. Please check details below.")
            st.code(traceback.format_exc())

        if st.button("⬅️ Back to Input Page"):
            st.session_state.page = "input"
            st.rerun()


