import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Alzheimer's Risk Prediction App", layout="centered")
st.title("🧠 Alzheimer's Disease Risk Prediction")
st.markdown("Upload patient data below and get a risk prediction.")

# Load your model and preprocessors with correct filenames
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

def get_user_input():
    st.header("Enter Patient Information")
    # Replace with your actual feature inputs (example placeholders here)
    inputs = {}
    for feature in label_encoders.keys():
        # If your encoded features are categorical
        if hasattr(label_encoders[feature], 'classes_'):
            options = list(label_encoders[feature].classes_)
            inputs[feature] = st.selectbox(feature, options)
        else:
            # For numeric or other types, just do a text input or number input
            inputs[feature] = st.text_input(feature)

    # For features not encoded (numeric), add your sliders or number inputs
    # Example numeric input (update with your real features)
    # inputs['Age'] = st.slider('Age', 40, 100, 65)

    return pd.DataFrame([inputs])

def preprocess_input(data):
    # Encode categorical features
    for col in label_encoders.keys():
        if col in data.columns:
            data[col] = label_encoders[col].transform(data[col])
    # Scale
    data_scaled = scaler.transform(data)
    return data_scaled

user_data = get_user_input()

if st.button("Predict Alzheimer's Risk"):
    try:
        processed_data = preprocess_input(user_data)
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)[0][1]

        if prediction[0] == 1:
            st.error(f"High Risk of Alzheimer's ({prediction_proba * 100:.2f}%)")
        else:
            st.success(f"Low Risk of Alzheimer's ({(1 - prediction_proba) * 100:.2f}%)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
