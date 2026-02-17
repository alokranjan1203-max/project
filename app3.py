import streamlit as st
import pickle
import pandas as pd
import gdown
import os

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# ----------------------------------
# Model Config
# ----------------------------------
MODEL_PATH = "churn_model.pkl"

GDRIVE_URL = "https://drive.google.com/uc?id=1hB3P3v8UqIUoupZ7e4GvGlDtW3Tz65IS"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        pipeline, feature_columns = pickle.load(f)
    return pipeline, feature_columns

pipeline, feature_columns = load_model()

# ----------------------------------
# UI
# ----------------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("Enter customer details below:")

with st.form("prediction_form"):

    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (Months)", 0, 120, 12)
    usage_frequency = st.number_input("Usage Frequency", 0, 50, 5)
    support_calls = st.number_input("Support Calls", 0, 20, 1)
    payment_delay = st.number_input("Payment Delay (Days)", 0, 60, 0)
    total_spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 500.0)
    last_interaction = st.number_input("Last Interaction (Days Ago)", 0, 365, 10)

    submitted = st.form_submit_button("Predict")

# ----------------------------------
# Prediction
# ----------------------------------
if submitted:

    input_df = pd.DataFrame([[
        age,
        tenure,
        usage_frequency,
        support_calls,
        payment_delay,
        total_spend,
        last_interaction
    ]], columns=feature_columns)

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write(f"Churn Probability: **{probability:.2%}**")
    st.progress(float(probability))

