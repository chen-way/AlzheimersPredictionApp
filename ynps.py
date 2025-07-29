import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import time
import random

st.set_page_config(page_title="Alzheimer's Prediction App", layout="wide")

# === LOAD MODEL AND ENCODERS (gzip) ===
try:
    with gzip.open("xgb_model.pkl.gz", "rb") as f:
        model = pickle.load(f)

    with gzip.open("target_encoder.pkl.gz", "rb") as f:
        target_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure xgb_model.pkl.gz and target_encoder.pkl.gz are in the same directory.")
    st.stop()

# === FEATURE ENCODING MAPPINGS ===
# These mappings convert categorical strings to numbers for the model
FEATURE_ENCODINGS = {
    'Country': {country: idx for idx, country in enumerate(['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'South Korea', 'India', 'China', 'Brazil', 'South Africa', 'Australia', 'Russia', 'Mexico', 'Italy'])},
    'Gender': {'Male': 1, 'Female': 0},
    'Education Level': {'No Formal Education': 0, 'Primary Education': 1, 'Secondary Education': 2, "Bachelor's Degree": 3, "Master's Degree": 4, 'Doctorate': 5},
    'Physical Activity Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Smoking Status': {'Never': 0, 'Former': 1, 'Current': 2},
    'Alcohol Consumption': {'None': 0, 'Moderate': 1, 'High': 2},
    'Diabetes': {'No': 0, 'Yes': 1},
    'Hypertension': {'No': 0, 'Yes': 1},
    'Cholesterol Level': {'Low': 0, 'Normal': 1, 'High': 2},
    'Family History of Alzheimer\'s': {'No': 0, 'Yes': 1},
    'Sleep Quality': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
    'Dietary Habits': {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2},
    'Air Pollution Exposure': {'Minimal': 0, 'Slight': 1, 'Moderate': 2, 'High': 3, 'Severe': 4},
    'Employment Status': {'Unemployed': 0, 'Student': 1, 'Employed': 2, 'Retired': 3},
    'Marital Status': {'Single': 0, 'Divorced': 1, 'Widowed': 2, 'Married': 3},
    'Genetic Risk Factor (APOE-ε4 allele)': {'No': 0, 'Yes': 1},
    'Social Engagement Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Income Level': {'Low': 0, 'Middle': 1, 'High': 2},
    'Urban vs Rural Living': {'Rural': 0, 'Urban': 1}
}

# === FEATURE DEFINITIONS ===
# Categorical feature options
CATEGORICAL_OPTIONS = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'South Korea', 'India', 'China', 'Brazil', 'South Africa', 'Australia', 'Russia', 'Mexico', 'Italy'],
    'Gender': ['Male', 'Female'],
    'Education Level': ['No Formal Education', 'Primary Education', 'Secondary Education', "Bachelor's Degree", "Master's Degree", 'Doctorate'],
    'Physical Activity Level': ['Low', 'Moderate', 'High'],
    'Smoking Status': ['Never', 'Former', 'Current'],
    'Alcohol Consumption': ['None', 'Moderate', 'High'],
    'Diabetes': ['Yes', 'No'],
    'Hypertension': ['Yes', 'No'],
    'Cholesterol Level': ['Low', 'Normal', 'High'],
    'Family History of Alzheimer\'s': ['Yes', 'No'],
    'Sleep Quality': ['Poor', 'Fair', 'Good', 'Excellent'],
    'Dietary Habits': ['Unhealthy', 'Moderate', 'Healthy'],
    'Employment Status': ['Employed', 'Unemployed', 'Retired', 'Student'],
    'Marital Status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'Genetic Risk Factor (APOE-ε4 allele)': ['Yes', 'No'],
    'Social Engagement Level': ['Low', 'Moderate', 'High'],
    'Income Level': ['Low', 'Middle', 'High'],
    'Urban vs Rural Living': ['Urban', 'Rural'],
    'Air Pollution Exposure': ['Minimal', 'Slight', 'Moderate', 'High', 'Severe']
}

# Numerical features
NUMERICAL_FEATURES = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']

# === FEATURE LIST (ordered for model) ===
feature_names = [
    'Country', 'Age', 'Gender', 'Education Level', 'BMI',
    'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
    'Diabetes', 'Hypertension', 'Cholesterol Level',
    'Family History of Alzheimer\'s', 'Cognitive Test Score', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels', 'Urban vs Rural Living'
]

# === ENCODING FUNCTION ===
def encode_categorical_features(df):
    """Convert categorical string values to numerical values for the model"""
    df_encoded = df.copy()
    
    for feature, mapping in FEATURE_ENCODINGS.items():
        if feature in df_encoded.columns:
            df_encoded[feature] = df_encoded[feature].map(mapping)
    
    return df_encoded

# === USER INPUT FUNCTION ===
def get_user_input():
    user_data = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    feature_count = 0
    for feature in feature_names:
        # Alternate between columns
        current_col = col1 if feature_count % 2 == 0 else col2
        
        with current_col:
            if feature in CATEGORICAL_OPTIONS:
                # Categorical features with dropdowns
                value = st.selectbox(
                    f"**{feature}**:", 
                    options=CATEGORICAL_OPTIONS[feature],
                    key=feature
                )
                user_data[feature] = value
            elif feature in NUMERICAL_FEATURES:
                # Numerical features with appropriate ranges and step sizes
                if feature == 'Age':
                    value = st.number_input(
                        f"**{feature}** (years):", 
                        min_value=18, max_value=120, value=65, step=1, key=feature
                    )
                elif feature == 'BMI':
                    value = st.number_input(
                        f"**{feature}** (kg/m²):", 
                        min_value=10.0, max_value=50.0, value=25.0, step=0.1, key=feature
                    )
                elif feature == 'Cognitive Test Score':
                    value = st.number_input(
                        f"**{feature}** (0-30):", 
                        min_value=0, max_value=30, value=25, step=1, key=feature
                    )
                elif feature == 'Depression Level':
                    value = st.number_input(
                        f"**{feature}** (0-15, higher = more depressed):", 
                        min_value=0, max_value=15, value=2, step=1, key=feature
                    )

                elif feature == 'Stress Levels':
                    value = st.number_input(
                        f"**{feature}** (0-10, higher = more stress):", 
                        min_value=0, max_value=10, value=5, step=1, key=feature
                    )
                else:
                    value = st.number_input(f"**{feature}**:", key=feature, step=1.0)
                
                user_data[feature] = value
        
        feature_count += 1
    
    return pd.DataFrame([user_data])

# === MAIN APP ===
st.title("🧠 Alzheimer's Disease Risk Prediction")
st.markdown("Fill in the information below to predict Alzheimer's risk. All fields are required for accurate prediction.")

# Add some helpful information
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **Instructions:**
    1. Fill out all the fields with accurate information
    2. For numerical values, use the sliders or input boxes
    3. For categorical values, select from the dropdown menus
    4. Click 'Predict' to get your risk assessment
    
    **Note:** This tool uses machine learning to assess risk factors but should not replace professional medical advice.
    """)

# Get user input
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🧪 Predict Alzheimer's Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing your data..."):
            # Add a small delay for better UX
            time.sleep(2)
            
            try:
                # Encode categorical features for the model
                user_input_encoded = encode_categorical_features(user_input_df)
                
                # Ensure all features are present and in the right order
                user_input_encoded = user_input_encoded[feature_names]
                
                # Make prediction
                prediction = model.predict(user_input_encoded)[0]
                probability = model.predict_proba(user_input_encoded)[0]
                
                # Get label from target encoder and make it more meaningful
                raw_label = target_encoder.inverse_transform([prediction])[0]
                
                # Convert raw labels to meaningful descriptions
                def interpret_prediction(raw_pred):
                    raw_str = str(raw_pred).lower()
                    
                    # Map common numeric/text labels to meaningful descriptions
                    if raw_str in ['1', '1.0', 'high', 'high risk', 'positive', 'yes']:
                        return "High Risk"
                    elif raw_str in ['0', '0.0', 'low', 'low risk', 'negative', 'no']:
                        return "Low Risk"
                    elif raw_str in ['2', '2.0', 'moderate', 'medium']:
                        return "Moderate Risk"
                    else:
                        # If we can't interpret it, show both raw and a generic description
                        return f"Risk Level {raw_pred}"
                
                label = interpret_prediction(raw_label)
                
                # Display results
                st.markdown("### 🧾 Prediction Results")
                
                # Create a nice result box with better risk level descriptions
                label_str = str(label).lower()
                if 'high' in label_str:
                    st.error(f"**Prediction: {label}**")
                    st.warning("⚠️ The model indicates elevated risk factors. Please consult with a healthcare professional for comprehensive evaluation and personalized guidance.")
                elif 'low' in label_str:
                    st.success(f"**Prediction: {label}**")
                    st.info("✅ The model indicates lower risk based on current factors. Continue maintaining healthy lifestyle habits and regular medical checkups.")
                elif 'moderate' in label_str:
                    st.warning(f"**Prediction: {label}**")
                    st.info("🔶 The model indicates moderate risk. Consider discussing prevention strategies with your healthcare provider.")
                else:
                    # For any other prediction
                    st.info(f"**Prediction: {label}**")
                    st.info("Please consult with a healthcare professional for proper evaluation and continue maintaining healthy lifestyle habits.")
                
                # Show probability if available
                if len(probability) > 1:
                    st.markdown("**Confidence Scores:**")
                    prob_df = pd.DataFrame({
                        'Risk Level': target_encoder.classes_,
                        'Probability': [f"{p:.1%}" for p in probability]
                    })
                    st.dataframe(prob_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.info("Please check your input values and try again.")
                # Debug information
                st.write("Debug info - DataFrame dtypes:")
                st.write(user_input_df.dtypes)

# === LIFESTYLE TIPS SECTION ===
st.markdown("---")
st.header("🧘 Lifestyle Tips to Reduce Alzheimer's Risk")

tips = [
    "🫐 **Diet:** Eat more brain-healthy foods like leafy greens, berries, fatty fish, and nuts (Mediterranean diet).",
    "🚶 **Exercise:** Stay physically active — aim for at least 150 minutes of moderate exercise per week.",
    "🧩 **Mental Stimulation:** Challenge your brain with puzzles, reading, learning new skills, or playing musical instruments.",
    "👫 **Social Connection:** Stay socially engaged with family, friends, and community activities.",
    "😴 **Sleep:** Maintain good sleep hygiene with 7–9 hours of quality sleep per night.",
    "🚭 **Smoking:** Avoid smoking and limit alcohol consumption to moderate levels.",
    "🩺 **Health Management:** Control blood pressure, diabetes, and cholesterol levels.",
    "🧘 **Stress Management:** Practice stress-reduction techniques like meditation or yoga.",
    "🏥 **Regular Checkups:** Have regular medical checkups and discuss cognitive health with your doctor.",
    "🎯 **Purpose:** Maintain a sense of purpose and engage in meaningful activities."
]

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("💡 Get a Random Tip"):
        selected_tip = random.choice(tips)
        with col2:
            st.info(selected_tip)

# Show all tips in an expander
with st.expander("📋 View All Prevention Tips"):
    for tip in tips:
        st.markdown(f"• {tip}")

# === SIDEBAR INFORMATION ===
st.sidebar.markdown("### 📊 About This Tool")
st.sidebar.info("""
This prediction tool uses machine learning to assess Alzheimer's disease risk based on various health and lifestyle factors.

**Features analyzed:**
- Demographics & lifestyle
- Health conditions
- Cognitive assessments
- Environmental factors
- Genetic risk factors
""")

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Important Disclaimer:** This tool is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Additional Resources")
st.sidebar.markdown("""
- [Alzheimer's Association](https://www.alz.org)
- [National Institute on Aging](https://www.nia.nih.gov)
- [CDC Alzheimer's Resources](https://www.cdc.gov/aging/aginginfo/alzheimers.htm)
""")
