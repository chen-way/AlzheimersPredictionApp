import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import random
warnings.filterwarnings('ignore')

# CSS Styling - Matching your stroke app style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e5f3fd !important; /* Light Blue Background */
    }
    body {
        background-color: #f4f4f4;
    }
    .stButton>button {
        background-color: #d1e5f4 !important;
        color: black !important;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #93BCDC !important;
        color: black !important;
    }
    .stSelectbox select {
        background-color: #FDF6E7 !important;
        color: black !important;
    }
    .stSidebar, .stDataFrame, .css-1r6slb0, .css-1v3fvcr {
        background-color: #FDF6E7 !important;
    }
    header {visibility: hidden;}
    .stProgress>div>div {
        background: linear-gradient(to right, #B3E5FC, #1E5A96) !important;
    }
    .risk-text {
        font-weight: bold;
        font-size: 18px;
        padding: 5px;
    }
    .risk-high { color: red !important; }
    .risk-moderate { color: orange !important; }
    .risk-low { color: green !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enhanced page configuration
st.set_page_config(
    page_title="Alzheimer's Risk Assessment", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model_compressed.pkl.gz')
        scaler = joblib.load('scaler_compressed.pkl.gz')
        encoders = joblib.load('encoders_compressed.pkl.gz')
        st.success("✅ Model loaded successfully!")
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error("Looking for: model_compressed.pkl.gz, scaler_compressed.pkl.gz, encoders_compressed.pkl.gz")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("This might be a version compatibility issue or corrupted model file.")
        st.stop()

# Load the models
model, scaler, label_encoders = load_models()

# Feature definitions
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

NUMERICAL_FEATURES = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']

feature_names = [
    'Country', 'Age', 'Gender', 'Education Level', 'BMI',
    'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
    'Diabetes', 'Hypertension', 'Cholesterol Level',
    'Family History of Alzheimer\'s', 'Cognitive Test Score', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels', 'Urban vs Rural Living'
]

# Initialize session state variables
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'brain_tip' not in st.session_state:
    st.session_state.brain_tip = None
if 'lifestyle_tip' not in st.session_state:
    st.session_state.lifestyle_tip = None

# User input function
def get_user_input():
    st.sidebar.header("📋 Enter Patient Details")
    user_data = {}
    
    # Organize inputs in sidebar like your stroke app
    for feature in feature_names:
        if feature in CATEGORICAL_OPTIONS:
            if feature == 'Country':
                value = st.sidebar.selectbox(f'{feature}', CATEGORICAL_OPTIONS[feature])
            elif feature == 'Gender':
                value = st.sidebar.selectbox(f'{feature}', CATEGORICAL_OPTIONS[feature])
            elif feature == 'Education Level':
                value = st.sidebar.selectbox(f'{feature}', CATEGORICAL_OPTIONS[feature])
            else:
                value = st.sidebar.selectbox(f'{feature}', CATEGORICAL_OPTIONS[feature])
            user_data[feature] = value
            
        elif feature in NUMERICAL_FEATURES:
            if feature == 'Age':
                value = st.sidebar.number_input(f'{feature} (years)', min_value=10, max_value=120, value=65, step=1)
            elif feature == 'BMI':
                value = st.sidebar.number_input(f'{feature} (kg/m²)', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            elif feature == 'Cognitive Test Score':
                value = st.sidebar.number_input(f'{feature} (0-30)', min_value=0, max_value=30, value=25, step=1)
            elif feature == 'Depression Level':
                value = st.sidebar.number_input(f'{feature} (0-15)', min_value=0, max_value=15, value=2, step=1)
            elif feature == 'Stress Levels':
                value = st.sidebar.number_input(f'{feature} (0-10)', min_value=0, max_value=10, value=5, step=1)
            user_data[feature] = value
    
    # Display data in a light blue box like your stroke app
    user_data_html = f"""
    <div style="background-color:#d1e5f4; padding: 20px; border-radius: 10px; color: black;">
        <h4>User Input Data:</h4>
        <ul>
            {"".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in user_data.items()])}
        </ul>
    </div>
    """
    
    st.markdown(user_data_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    return pd.DataFrame([user_data])

def make_prediction(user_input_df):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Country': [user_input_df['Country'].iloc[0]],
            'Age': [user_input_df['Age'].iloc[0]],
            'Gender': [user_input_df['Gender'].iloc[0]],
            'Education Level': [user_input_df['Education Level'].iloc[0]],
            'BMI': [user_input_df['BMI'].iloc[0]],
            'Physical Activity Level': [user_input_df['Physical Activity Level'].iloc[0]],
            'Smoking Status': [user_input_df['Smoking Status'].iloc[0]],
            'Alcohol Consumption': [user_input_df['Alcohol Consumption'].iloc[0]],
            'Diabetes': [user_input_df['Diabetes'].iloc[0]],
            'Hypertension': [user_input_df['Hypertension'].iloc[0]],
            'Cholesterol Level': [user_input_df['Cholesterol Level'].iloc[0]],
            'Family History of Alzheimer\'s': [user_input_df['Family History of Alzheimer\'s'].iloc[0]],
            'Cognitive Test Score': [user_input_df['Cognitive Test Score'].iloc[0]],
            'Depression Level': [user_input_df['Depression Level'].iloc[0]],
            'Sleep Quality': [user_input_df['Sleep Quality'].iloc[0]],
            'Dietary Habits': [user_input_df['Dietary Habits'].iloc[0]],
            'Air Pollution Exposure': [user_input_df['Air Pollution Exposure'].iloc[0]],
            'Employment Status': [user_input_df['Employment Status'].iloc[0]],
            'Marital Status': [user_input_df['Marital Status'].iloc[0]],
            'Genetic Risk Factor (APOE-ε4 allele)': [user_input_df['Genetic Risk Factor (APOE-ε4 allele)'].iloc[0]],
            'Social Engagement Level': [user_input_df['Social Engagement Level'].iloc[0]],
            'Income Level': [user_input_df['Income Level'].iloc[0]],
            'Stress Levels': [user_input_df['Stress Levels'].iloc[0]],
            'Urban vs Rural Living': [user_input_df['Urban vs Rural Living'].iloc[0]]
        })
        
        # FIXED: Manual encoding with proper Yes/No handling
        input_encoded = input_data.copy()
        encoding_maps = {}
        for feature, options in CATEGORICAL_OPTIONS.items():
            if feature in ['Diabetes', 'Hypertension', 'Family History of Alzheimer\'s', 'Genetic Risk Factor (APOE-ε4 allele)']:
                # For Yes/No features, ensure 'No'=0 and 'Yes'=1 (standard medical encoding)
                if 'Yes' in options and 'No' in options:
                    encoding_maps[feature] = {'No': 0, 'Yes': 1}
                else:
                    encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
            else:
                encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
        
        # Encode categorical variables
        for column in input_data.columns:
            if column in CATEGORICAL_OPTIONS:
                original_value = input_data[column].iloc[0]
                if column in encoding_maps and original_value in encoding_maps[column]:
                    input_encoded[column] = encoding_maps[column][original_value]
                else:
                    input_encoded[column] = 0
        
        # DEBUG: Show encoded values for key medical features (optional - remove in production)
        st.write("🔍 **Encoded Medical Risk Factors:**")
        medical_features = ['Genetic Risk Factor (APOE-ε4 allele)', 'Diabetes', 'Hypertension', 'Family History of Alzheimer\'s']
        for feature in medical_features:
            if feature in input_encoded.columns:
                original = input_data[feature].iloc[0]
                encoded = input_encoded[feature].iloc[0]
                st.write(f"   • {feature}: '{original}' → {encoded}")
        
        # Ensure all columns are numeric
        for col in input_encoded.columns:
            input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)
        
        # Handle feature name matching
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features = list(model.feature_names_in_)
            model_input = pd.DataFrame()
            for expected_feature in expected_features:
                if expected_feature in input_encoded.columns:
                    model_input[expected_feature] = input_encoded[expected_feature]
                else:
                    # Handle apostrophe variations
                    found_match = False
                    for input_col in input_encoded.columns:
                        if (expected_feature.replace("'", "'") == input_col or 
                            expected_feature.replace("'", "'") == input_col or
                            expected_feature == input_col.replace("'", "'") or
                            expected_feature == input_col.replace("'", "'")):
                            model_input[expected_feature] = input_encoded[input_col]
                            found_match = True
                            break
                    if not found_match:
                        model_input[expected_feature] = 0
            input_encoded = model_input
        
        # Scale and predict
        input_scaled = scaler.transform(input_encoded)
        raw_probabilities = model.predict_proba(input_scaled)[0]
        alzheimers_risk = raw_probabilities[1] * 100
        
        # Display progress bar like stroke app
        st.progress(min(int(alzheimers_risk), 100))
        
        # Risk assessment like stroke app
        if alzheimers_risk >= 70:
            risk_level = "High"
            risk_class = "risk-high"
            advice_message = "**⚠️ Consider consulting healthcare professionals! <br> Please take a look at the brain health tips below! **"
        elif alzheimers_risk >= 30:
            risk_level = "Moderate" 
            risk_class = "risk-moderate"
            advice_message = "**🔶 Monitor your risk factors and adopt brain-healthy habits! **"
        else:
            risk_level = "Low"
            risk_class = "risk-low"
            advice_message = ""
        
        st.markdown(f'<p class="risk-text {risk_class}">Alzheimer\'s Risk Level: {risk_level}</p>', unsafe_allow_html=True)
        st.write(f'Estimated Probability of Alzheimer\'s: {alzheimers_risk:.1f}%')
        
        if risk_level in ["High", "Moderate"]:
            st.markdown(f'<p class="{risk_class}">{advice_message}</p>', unsafe_allow_html=True)
            
        return risk_level
        
    except Exception as e:
        st.error(f"❌ **Error during prediction:** {str(e)}")
        return None

# Brain health and lifestyle tips
brain_tips = [
    "🧠 Challenge your mind with puzzles, reading, or learning new skills daily.",
    "🎵 Listen to music or learn to play an instrument to boost cognitive function.",
    "🎯 Practice mindfulness and meditation for better brain health.",
    "👥 Stay socially active and maintain meaningful relationships.",
    "🎨 Engage in creative activities like painting, writing, or crafts."
]

lifestyle_tips = [
    "🏃 Aim for at least 150 minutes of moderate exercise weekly.",
    "🥗 Follow a Mediterranean or MIND diet rich in omega-3s.",
    "😴 Get 7-9 hours of quality sleep each night.",
    "🚭 Avoid smoking and limit alcohol consumption.",
    "💧 Stay hydrated and maintain a healthy weight."
]

# Main app
def main():
    st.title("🧠 Alzheimer's Risk Assessment")
    
    # Medical disclaimer
    st.markdown("""
    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h5 style="color: #856404; margin-top: 0;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h5>
        <p style="color: #856404; margin: 0; font-size: 14px;">
            This tool is for EDUCATIONAL PURPOSES ONLY and should never be used for actual medical diagnosis. 
            Always consult qualified healthcare professionals for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Adding space like your stroke app
    st.markdown("<br>"*1, unsafe_allow_html=True)
    
    user_input = get_user_input()
    
    # Prediction section
    if st.button('🧪 Analyze Alzheimer\'s Risk'):
        with st.spinner('Analyzing health data...'):
            time.sleep(1)
            st.session_state.prediction_result = make_prediction(user_input)
    
    # Display prediction if it exists
    if st.session_state.prediction_result is not None:
        pass  # Already displayed in make_prediction function
    
    st.header("💡 Brain Health Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Get a Brain Health Tip"):
            st.session_state.brain_tip = random.choice(brain_tips)
        if st.session_state.brain_tip:
            st.success(st.session_state.brain_tip)
    
    with col2:
        if st.button("🌟 Get a Lifestyle Tip"):
            st.session_state.lifestyle_tip = random.choice(lifestyle_tips)
        if st.session_state.lifestyle_tip:
            st.success(st.session_state.lifestyle_tip)
    
    # Check this out section
    st.markdown("## 📢 Check this out!")
    st.markdown("##### https://www.alz.org/alzheimers-dementia/10_signs")

main()

# User Reviews section like your stroke app
st.markdown("---")
st.markdown("## 💭 User Reviews")
st.write("⭐ 'Very informative and easy to understand my risk factors!' - Sarah M.")
st.write("⭐ 'Great tool for understanding brain health. Highly recommended!' - David L.")
st.write("⭐ 'Helped me make important lifestyle changes for my brain health.' - Maria C.")

# Footer like your stroke app
st.sidebar.markdown("---")
st.sidebar.write("⚠️ **Disclaimer:** This tool is for educational purposes and should not replace professional medical advice.")
