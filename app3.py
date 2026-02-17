import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

MODEL_PATH = "churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.title("📊 Customer Churn Prediction")

with st.form("form"):

    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (Months)", 0, 120, 12)
    usage_frequency = st.number_input("Usage Frequency", 0, 50, 5)
    support_calls = st.number_input("Support Calls", 0, 20, 1)
    payment_delay = st.number_input("Payment Delay (Days)", 0, 60, 0)
    total_spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 500.0)
    last_interaction = st.number_input("Last Interaction (Days Ago)", 0, 365, 10)

    submitted = st.form_submit_button("Predict")

if submitted:

    input_df = pd.DataFrame([[
        age,
        tenure,
        usage_frequency,
        support_calls,
        payment_delay,
        total_spend,
        last_interaction
    ]], columns=[
        "age",
        "tenure",
        "usage frequency",
        "support calls",
        "payment delay",
        "total spend",
        "last interaction"
    ])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write(f"Churn Probability: {probability:.2%}")
    st.progress(float(probability))


