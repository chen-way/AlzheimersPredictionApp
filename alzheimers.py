import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and encoders
model = joblib.load("best_rf_compressed.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoder.pkl")  # This should be a dict of LabelEncoders

st.set_page_config(page_title="Alzheimer's Risk Prediction", layout="centered")
st.title("🧠 Alzheimer's Disease Risk Prediction")
st.markdown("Upload patient data to predict the risk of Alzheimer's disease.")

# ====== Get user input ======
def get_user_input():
    st.header("Enter Patient Information")

    country = st.selectbox("Country", ["USA", "Canada", "UK", "India", "Other"])
    age = st.slider("Age", 40, 100, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
    bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
    physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Borderline", "High"])
    family_history = st.selectbox("Family History of Alzheimer’s", ["No", "Yes"])
    cognitive_score = st.slider("Cognitive Test Score", 0, 30, 25)
    depression = st.selectbox("Depression Level", ["None", "Mild", "Moderate", "Severe"])
    sleep = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
    diet = st.selectbox("Dietary Habits", ["Unhealthy", "Average", "Healthy"])
    pollution = st.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"])
    employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    genetics = st.selectbox("Genetic Risk Factor (APOE-ε4 allele)", ["No", "Yes"])
    social = st.selectbox("Social Engagement Level", ["Low", "Moderate", "High"])
    income = st.selectbox("Income Level", ["Low", "Middle", "High"])
    stress = st.selectbox("Stress Levels", ["Low", "Moderate", "High"])
    urban_rural = st.selectbox("Urban vs Rural Living", ["Urban", "Rural"])

    user_input = pd.DataFrame({
        "Country": [country],
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "BMI": [bmi],
        "Physical Activity Level": [physical_activity],
        "Smoking Status": [smoking],
        "Alcohol Consumption": [alcohol],
        "Diabetes": [diabetes],
        "Hypertension": [hypertension],
        "Cholesterol Level": [cholesterol],
        "Family History of Alzheimer’s": [family_history],
        "Cognitive Test Score": [cognitive_score],
        "Depression Level": [depression],
        "Sleep Quality": [sleep],
        "Dietary Habits": [diet],
        "Air Pollution Exposure": [pollution],
        "Employment Status": [employment],
        "Marital Status": [marital],
        "Genetic Risk Factor (APOE-ε4 allele)": [genetics],
        "Social Engagement Level": [social],
        "Income Level": [income],
        "Stress Levels": [stress],
        "Urban vs Rural Living": [urban_rural]
    })

    return user_input

# ====== Preprocess input to match model format ======
def preprocess_input(df):
    df = df.copy()
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    df_scaled = scaler.transform(df)
    return df_scaled

# ====== Run prediction ======
user_data = get_user_input()

if st.button("Predict Alzheimer's Risk"):
    try:
        processed = preprocess_input(user_data)
        prediction = model.predict(processed)
        proba = model.predict_proba(processed)[0][1]

        if prediction[0] == 1:
            st.error(f"🚨 High Risk of Alzheimer’s ({proba * 100:.2f}%)")
        else:
            st.success(f"🟢 Low Risk of Alzheimer’s ({(1 - proba) * 100:.2f}%)")

    except Exception as e:
        st.exception(f"Something went wrong: {e}")
