import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Page configuration
st.set_page_config(page_title="Alzheimer's Risk Predictor", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f7f5;
    }
    .stButton>button {
        background-color: #9ED2BE;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #70B89D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Alzheimer's Disease Risk Predictor")
st.write("Enter patient information below to estimate their risk of Alzheimer's.")

# Load pre-trained model and preprocessors (you must save these separately beforehand)
model = joblib.load("alzheimers_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")  # LabelEncoders for categorical features

# Define feature input fields
features = {
    "Country": st.selectbox("Country", ["USA", "Canada", "UK", "India", "Other"]),
    "Age": st.number_input("Age", min_value=0, max_value=120, value=65),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Education Level": st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"]),
    "BMI": st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5),
    "Physical Activity Level": st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"]),
    "Smoking Status": st.selectbox("Smoking Status", ["Never", "Former", "Current"]),
    "Alcohol Consumption": st.selectbox("Alcohol Consumption", ["None", "Occasional", "Regular"]),
    "Diabetes": st.selectbox("Diabetes", ["Yes", "No"]),
    "Hypertension": st.selectbox("Hypertension", ["Yes", "No"]),
    "Cholesterol Level": st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200),
    "Family History of Alzheimer’s": st.selectbox("Family History of Alzheimer’s", ["Yes", "No"]),
    "Cognitive Test Score": st.number_input("Cognitive Test Score", min_value=0, max_value=100, value=75),
    "Depression Level": st.selectbox("Depression Level", ["None", "Mild", "Moderate", "Severe"]),
    "Sleep Quality": st.selectbox("Sleep Quality", ["Poor", "Fair", "Good"]),
    "Dietary Habits": st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"]),
    "Air Pollution Exposure": st.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"]),
    "Employment Status": st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"]),
    "Marital Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"]),
    "Genetic Risk Factor (APOE-ε4 allele)": st.selectbox("APOE-ε4 allele Present?", ["Yes", "No"]),
    "Social Engagement Level": st.selectbox("Social Engagement Level", ["Low", "Moderate", "High"]),
    "Income Level": st.selectbox("Income Level", ["Low", "Medium", "High"]),
    "Stress Levels": st.selectbox("Stress Levels", ["Low", "Moderate", "High"]),
    "Urban vs Rural Living": st.selectbox("Living Area", ["Urban", "Rural"])
}

# Convert input to DataFrame
input_df = pd.DataFrame([features])

# Apply encoders for categorical columns
for col in input_df.columns:
    if col in encoder:
        input_df[col] = encoder[col].transform(input_df[col])

# Scale numerical features
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔍 Predict Alzheimer Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Alzheimer's Disease (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Alzheimer's Disease (Probability: {probability:.2%})")

# Disclaimer
st.markdown("---")
st.markdown("⚠️ This tool is for educational purposes only and should not replace professional medical advice.")
