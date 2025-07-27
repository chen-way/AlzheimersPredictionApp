import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Alzheimer's Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load trained model, scaler, and feature names
with open('alzheimers_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# App title
st.title("🧠 Alzheimer's Disease Risk Predictor")
st.markdown("This tool uses machine learning to estimate your risk of developing Alzheimer's disease based on health indicators.")

# Collect user input
st.subheader("Enter your health information below:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=30, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, step=1)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400)

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=250)
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    family_history = st.selectbox("Family history of Alzheimer's?", ["No", "Yes"])

# Encode categorical variables manually (you must have done similar during training)
gender_encoded = 1 if gender == "Male" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
family_history_encoded = 1 if family_history == "Yes" else 0

# Create input dataframe with proper column names
input_dict = {
    'Age': age,
    'Gender': gender_encoded,
    'SystolicBP': systolic_bp,
    'Cholesterol': cholesterol,
    'BMI': bmi,
    'Glucose': glucose,
    'Smoking': smoking_encoded,
    'FamilyHistory': family_history_encoded
}

input_df = pd.DataFrame([input_dict])

# Reorder columns to match training data
input_df = input_df[feature_names]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.error(f"High Risk of Alzheimer's Disease ({probability*100:.1f}% probability)")
    else:
        st.success(f"Low Risk of Alzheimer's Disease ({probability*100:.1f}% probability)")
