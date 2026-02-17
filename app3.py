import streamlit as st
import requests
import joblib
import os
import pandas as pd

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# ----------------------------------
# Hugging Face Model URL
# ----------------------------------
MODEL_PATH = "churn_model.pkl"
HF_MODEL_URL = "https://huggingface.co/Alok2005/churn_model/resolve/main/churn_model.pkl"

# ----------------------------------
# Download & Load Model
# ----------------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(HF_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    return joblib.load(MODEL_PATH)

loaded_object = download_and_load_model()

# ----------------------------------
# Handle Tuple or Pipeline
# ----------------------------------
if isinstance(loaded_object, tuple):
    model, scaler = loaded_object
    use_scaler = True
else:
    model = loaded_object
    scaler = None
    use_scaler = False

# ----------------------------------
# UI
# ----------------------------------
st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details to predict churn probability.")

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
    ]], columns=[
        "age",
        "tenure",
        "usage frequency",
        "support calls",
        "payment delay",
        "total spend",
        "last interaction"
    ])

    # If scaler exists → scale input
    if use_scaler:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df

    prediction = model.predict(input_processed)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_processed)[0][1]
    else:
        probability = 0.0

    # ----------------------------------
    # Output
    # ----------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write(f"Churn Probability: **{probability:.2%}**")
    st.progress(float(probability))





