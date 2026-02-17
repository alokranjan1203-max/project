import streamlit as st
import requests
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

MODEL_PATH = "churn_model.pkl"
HF_MODEL_URL = "https://huggingface.co/Alok2005/churn_model/resolve/main/churn_model.pkl"

# -----------------------------------
# Download and Load Model
# -----------------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(HF_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

    return joblib.load(MODEL_PATH)

loaded_object = download_and_load_model()

# Since you saved (model, scaler, feature_columns)
model, scaler, feature_columns = loaded_object

# -----------------------------------
# UI
# -----------------------------------
st.title("📊 Customer Churn Prediction")

with st.form("prediction_form"):

    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (Months)", 0, 120, 12)
    usage_frequency = st.number_input("Usage Frequency", 0, 50, 5)
    support_calls = st.number_input("Support Calls", 0, 20, 1)
    payment_delay = st.number_input("Payment Delay (Days)", 0, 60, 0)
    total_spend = st.number_input("Total Spend ($)", 0.0, 100000.0, 500.0)
    last_interaction = st.number_input("Last Interaction (Days Ago)", 0, 365, 10)

    submitted = st.form_submit_button("Predict")

# -----------------------------------
# Prediction
# -----------------------------------
if submitted:

    input_dict = {
        "age": age,
        "tenure": tenure,
        "usage frequency": usage_frequency,
        "support calls": support_calls,
        "payment delay": payment_delay,
        "total spend": total_spend,
        "last interaction": last_interaction,
    }

    input_df = pd.DataFrame([input_dict])

    # Reorder columns exactly as training
    input_df = input_df[feature_columns]

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write(f"Churn Probability: **{probability:.2%}**")
    st.progress(float(probability))






