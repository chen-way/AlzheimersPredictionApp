import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Alzheimer's Risk Predictor", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
            padding: 2rem;
            border-radius: 10px;
        }
        h1 {
            color: #3b3b3b;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Alzheimer's Disease Risk Predictor")
st.markdown("Predict your likelihood of developing Alzheimer's disease based on health and demographic data.")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter your information")

    age = st.number_input("Age", min_value=30, max_value=100, value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    apoe4 = st.selectbox("APOE4 Status", ["Negative", "Positive"])
    systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    diastolic = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=22.5)
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Very High"])
    education = st.selectbox("Education Level", ["High School", "Some College", "Bachelor's", "Master's/PhD"])
    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

    submit = st.form_submit_button("Predict")

# When user submits form
if submit:
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "APOE4_Status": [apoe4],
        "Systolic_BP": [systolic],
        "Diastolic_BP": [diastolic],
        "BMI": [bmi],
        "Cholesterol": [cholesterol],
        "Education": [education],
        "Physical_Activity": [activity]
    })

    # Encode categorical features
    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale the data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    predicted_label = target_encoder.inverse_transform(prediction)[0]

    # Display result
    st.success(f"🧬 Based on the data provided, the predicted diagnosis is: **{predicted_label}**")

    if predicted_label.lower() == "normal":
        st.info("Great news! Your risk seems low, but continue living a healthy lifestyle.")
    elif "mild" in predicted_label.lower():
        st.warning("This suggests early signs of potential Alzheimer's. Consider speaking with a medical professional.")
    else:
        st.error("This indicates higher risk. Please consult your healthcare provider for further testing and support.")
