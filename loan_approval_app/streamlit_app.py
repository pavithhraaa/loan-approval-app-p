
import streamlit as st
import joblib
import numpy as np

model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏦 Loan Approval Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
loan_amount = st.slider("Loan Amount (in thousands)", 50, 700, 120)
loan_term = st.selectbox("Loan Term (in days)", [180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
applicant_income = st.number_input("Applicant Income", value=5000)
coapplicant_income = st.number_input("Coapplicant Income", value=0)

total_income = applicant_income + coapplicant_income
debt_ratio = loan_amount / total_income if total_income != 0 else 0

gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

input_data = np.array([[gender, married, dependents, education, self_employed,
                        loan_amount, loan_term, credit_history, property_area,
                        total_income, debt_ratio]])

input_scaled = scaler.transform(input_data)

if st.button("Predict Loan Status"):
    prediction = model.predict(input_scaled)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
    st.subheader(f"Loan Status: {result}")
