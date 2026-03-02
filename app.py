# ===============================
# Loan Approval Prediction App
# ===============================

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details:")

income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.selectbox("Credit History (0 = Bad, 1 = Good)", [0, 1])

if st.button("Predict"):

    input_data = np.array([[income, loan_amount, credit_history]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")