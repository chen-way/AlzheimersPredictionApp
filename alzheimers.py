import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import random

# Load model and encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==============================
# 🌐 Streamlit Config & Styling
# ==============================
st.set_page_config(page_title="Alzheimer's Risk Predictor", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7fbff;
    }
    .stButton>button {
        background-color: #bcd4ec;
        color: black;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #89a7c2;
    }
    .stSelectbox select {
        background-color: #f2f2f2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# 🧠 Title and Instructions
# ==============================
st.title("🧠 Alzheimer's Disease Risk Predictor")
st.markdown("""
Enter the following information to get a prediction of Alzheimer's disease risk based on lifestyle, medical, and demographic data.
""")

# ==============================
# 📝 User Input Form
# ==============================
st.sidebar.header("📋 Patient Information")

def get_user_input():
    user_data = {
        "Country": st.sidebar.selectbox("Country", ["USA", "Canada", "UK", "India", "Other"]),
        "Age": st.sidebar.slider("Age", 30, 100, 65),
        "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "Education Level": st.sidebar.selectbox("Education Level", ["High School", "Some College", "Bachelor", "Master", "Doctorate"]),
        "BMI": st.sidebar.number_input("BMI", 10.0, 50.0, 22.5),
        "Physical Activity Level": st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"]),
        "Smoking Status": st.sidebar.selectbox("Smoking", ["Never", "Former", "Current"]),
        "Alcohol Consumption": st.sidebar.selectbox("Alcohol Consumption", ["Never", "Occasional", "Regular"]),
        "Diabetes": st.sidebar.selectbox("Diabetes", ["Yes", "No"]),
        "Hypertension": st.sidebar.selectbox("Hypertension", ["Yes", "No"]),
        "Cholesterol Level": st.sidebar.selectbox("Cholesterol", ["Normal", "Borderline", "High"]),
        "Family History of Alzheimer’s": st.sidebar.selectbox("Family History", ["Yes", "No"]),
        "Cognitive Test Score": st.sidebar.slider("Cognitive Test Score (0-30)", 0, 30, 22),
        "Depression Level": st.sidebar.selectbox("Depression Level", ["None", "Mild", "Moderate", "Severe"]),
        "Sleep Quality": st.sidebar.selectbox("Sleep Quality", ["Poor", "Average", "Good"]),
        "Dietary Habits": st.sidebar.selectbox("Diet", ["Poor", "Moderate", "Healthy"]),
        "Air Pollution Exposure": st.sidebar.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"]),
        "Employment Status": st.sidebar.selectbox("Employment", ["Employed", "Unemployed", "Retired"]),
        "Marital Status": st.sidebar.selectbox("Marital Status", ["Single", "Married", "Widowed", "Divorced"]),
        "Genetic Risk Factor (APOE-ε4 allele)": st.sidebar.selectbox("Genetic Risk (APOE-ε4)", ["Yes", "No"]),
        "Social Engagement Level": st.sidebar.selectbox("Social Engagement", ["Low", "Moderate", "High"]),
        "Income Level": st.sidebar.selectbox("Income", ["Low", "Middle", "High"]),
        "Stress Levels": st.sidebar.selectbox("Stress", ["Low", "Moderate", "High"]),
        "Urban vs Rural Living": st.sidebar.selectbox("Living Area", ["Urban", "Rural"])
    }
    return pd.DataFrame([user_data])

# ==============================
# 🔮 Prediction Logic
# ==============================
input_df = get_user_input()

if st.button("🔍 Predict Alzheimer's Risk"):
    with st.spinner("Analyzing data..."):
        time.sleep(1)

        # Apply label encoders
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numeric values
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        pred_label = target_encoder.inverse_transform(prediction)[0]
        pred_proba = model.predict_proba(input_scaled)[0].max()

        # Display Results
        st.subheader("🩺 Prediction Result")
        if pred_label == "High Risk":
            st.markdown(f"<p style='color:red'><b>High risk of Alzheimer's</b></p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:green'><b>Low risk of Alzheimer's</b></p>", unsafe_allow_html=True)

        st.write(f"Confidence: **{pred_proba:.2%}**")
        st.markdown("---")
        st.markdown("*Disclaimer: This is an educational tool and not medical advice.*")
