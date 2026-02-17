


import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import gdown

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

# ---------------- DOWNLOAD + LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_file = "churn_model.pkl"

    if not os.path.exists(model_file):
        file_id = "1MEita3ulOxBMR8lSmp-EORQjlzPMTNSa"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_file, quiet=False)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# ---------------- UI ----------------
st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn.")

st.sidebar.header("Customer Inputs")

def user_input():
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
    total_spend = st.sidebar.number_input("Total Spend", 0, 10000, 500)
    usage_frequency = st.sidebar.slider("Usage Frequency", 0, 100, 50)
    support_calls = st.sidebar.slider("Support Calls", 0, 20, 2)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    subscription = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

    # Encode categorical inputs same as training
    gender = 0 if gender == "Male" else 1
    subscription = {"Basic": 0, "Standard": 1, "Premium": 2}[subscription]

    data = pd.DataFrame({
        "age": [age],
        "tenure": [tenure],
        "total spend": [total_spend],
        "usage frequency": [usage_frequency],
        "support calls": [support_calls],
        "gender": [gender],
        "subscription type": [subscription]
    })

    return data

input_data = user_input()

st.subheader("Input Summary")
st.write(input_data)

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    result = "⚠️ Customer WILL churn" if prediction == 1 else "✅ Customer will NOT churn"
    st.success(result)

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame({
        "Will Not Churn": [probability[0]],
        "Will Churn": [probability[1]]
    })
    st.bar_chart(prob_df.T)

     
