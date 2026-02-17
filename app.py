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

    # Delete corrupted file if exists
    if os.path.exists(model_file):
        os.remove(model_file)

    # Download from Google Drive
    file_id = "1hB3P3v8UqIUoupZ7e4GvGlDtW3Tz65IS"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_file, quiet=False, fuzzy=True)

    # Load model
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    if isinstance(model_data, tuple) and len(model_data) == 2:
        model, feature_columns = model_data
    else:
        model = model_data
        feature_columns = None

    return model, feature_columns

model, feature_columns = load_model()

# ---------------- UI ----------------
st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn.")

st.sidebar.header("Customer Inputs")

def user_input(feature_columns):
    user_dict = {}

    for feature in feature_columns:
        if feature == "gender":
            val = st.sidebar.selectbox("Gender", ["Male", "Female"])
            user_dict[feature] = 0 if val == "Male" else 1

        elif feature == "subscription type":
            val = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            user_dict[feature] = {"Basic": 0, "Standard": 1, "Premium": 2}[val]

        else:
            user_dict[feature] = st.sidebar.number_input(feature.capitalize(), 0.0, 10000.0, 50.0)

    return pd.DataFrame([user_dict], columns=feature_columns)

# ---------------- PREDICTION ----------------
if feature_columns is not None:
    input_data = user_input(feature_columns)

    st.subheader("Input Summary")
    st.write(input_data)
