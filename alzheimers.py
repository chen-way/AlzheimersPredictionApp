import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import time
import random

# Enhanced page configuration with custom styling
st.set_page_config(
    page_title="Alzheimer's Risk Assessment", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Modern CSS with updated color scheme
st.markdown("""
<style>
    /* Main background with modern gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Custom header styling with glassmorphism */
    .main-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem !important;
        background: linear-gradient(45deg, #ffffff, #e8f4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.3rem !important;
        font-weight: 400 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Feature input containers with modern glass effect */
    .feature-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .feature-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.12);
    }
    
    /* Modern prediction button */
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .predict-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Modern results containers */
    .result-high-risk {
        background: linear-gradient(135deg, #ff4757, #ff3838);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(255, 71, 87, 0.3);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #2ed573, #1dd1a1);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(46, 213, 115, 0.3);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .result-moderate-risk {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(255, 165, 2, 0.3);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Tips section with modern styling */
    .tips-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 2rem 0;
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .tips-container h2 {
        color: white !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Modern sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced input field styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        transform: translateY(-1px);
    }
    
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .stNumberInput > div > div:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        transform: translateY(-1px);
    }
    
    /* Modern animation effects */
    @keyframes modernPulse {
        0% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.02) rotate(1deg); }
        100% { transform: scale(1) rotate(0deg); }
    }
    
    .pulse-animation {
        animation: modernPulse 3s infinite ease-in-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Modern metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Modern button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Modern expandable sections */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.8);
    }
</style>
""", unsafe_allow_html=True)

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
# Modern header with glassmorphism
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🧠 Alzheimer's Risk Assessment</h1>
    <p>AI-Powered Cognitive Health Analysis & Personalized Insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced information section with modern styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(20px); padding: 2rem; border-radius: 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.25); color: white; margin-bottom: 2rem;">
        <h4 style="margin-bottom: 1rem; font-weight: 700;">🔬 Advanced AI Analysis</h4>
        <p style="margin: 0; line-height: 1.6;">Our cutting-edge machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations for optimal brain health.</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced expandable help section
with st.expander("ℹ️ How to use this assessment tool", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; color: white;">
        <h5 style="margin-bottom: 1.5rem; font-weight: 700;">📋 Complete Assessment Guide:</h5>
        <div style="display: grid; gap: 1rem;">
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; backdrop-filter: blur(10px);">
                <strong>1. Input Your Information:</strong> Fill out all 24 health and lifestyle factors using our intuitive interface
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; backdrop-filter: blur(10px);">
                <strong>2. Review & Verify:</strong> Ensure all information is accurate for the most reliable assessment
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; backdrop-filter: blur(10px);">
                <strong>3. Get AI Analysis:</strong> Click the prediction button to receive your personalized risk evaluation
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; backdrop-filter: blur(10px);">
                <strong>4. Explore Recommendations:</strong> Review evidence-based lifestyle suggestions tailored to your profile
            </div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.15); padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; backdrop-filter: blur(15px);">
            <strong>🏥 Important Medical Note:</strong> This tool provides educational insights based on research data. Always consult healthcare professionals for medical decisions and personalized care.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Get user input with modern styling
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 16px; margin: 2rem 0; backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.2);">
    <h3 style="color: white; text-align: center; margin: 0; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎯 AI Risk Assessment</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🧪 Analyze My Alzheimer's Risk", type="primary", use_container_width=True, 
                 help="Click to get your personalized AI-powered risk assessment"):
        
        # Enhanced loading animation
        with st.spinner("🔍 Processing your health data with advanced AI algorithms..."):
            # Create a progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress_bar.progress(i + 1)
            
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
                    
                    if raw_str in ['1', '1.0', 'high', 'high risk', 'positive', 'yes']:
                        return "High Risk"
                    elif raw_str in ['0', '0.0', 'low', 'low risk', 'negative', 'no']:
                        return "Low Risk"
                    elif raw_str in ['2', '2.0', 'moderate', 'medium']:
                        return "Moderate Risk"
                    else:
                        return f"Risk Level {raw_pred}"
                
                label = interpret_prediction(raw_label)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Enhanced results display with modern styling
                st.markdown("<br>", unsafe_allow_html=True)
                
                label_str = str(label).lower()
                if 'high' in label_str:
                    st.markdown(f"""
                    <div class="result-high-risk pulse-animation fade-in-up">
                        <h2 style="margin-bottom: 1rem; font-weight: 800;">⚠️ Elevated Risk Assessment</h2>
                        <h3 style="margin-bottom: 1.5rem; font-weight: 600;">Analysis Result: {label}</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 1.5rem; line-height: 1.6;">
                            Our AI analysis indicates elevated risk factors based on your current health profile. 
                            We strongly recommend consulting with healthcare professionals for comprehensive 
                            evaluation and personalized prevention strategies.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.15); padding: 1.25rem; border-radius: 12px; backdrop-filter: blur(10px);">
                            <strong>🏥 Recommended Next Steps:</strong> Schedule a consultation with your doctor to discuss these findings and develop a personalized cognitive health plan.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3); border: 1px solid rgba(255, 255, 255, 0.2);" class="fade-in-up">
                        <h2 style="margin-bottom: 1rem; font-weight: 800;">📊 Assessment Complete</h2>
                        <h3 style="margin-bottom: 1.5rem; font-weight: 600;">Analysis Result: {label}</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 1.5rem; line-height: 1.6;">
                            Please consult with healthcare professionals for proper evaluation and 
                            continue maintaining healthy lifestyle habits for optimal cognitive health.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced confidence scores with modern styling
                if len(probability) > 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); padding: 2rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.25); margin: 1.5rem 0;">
                        <h4 style="color: white; text-align: center; margin-bottom: 1.5rem; font-weight: 700; font-size: 1.4rem;">📈 Confidence Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create beautiful metrics display
                    prob_cols = st.columns(len(probability))
                    for i, (risk_level, prob) in enumerate(zip(target_encoder.classes_, probability)):
                        with prob_cols[i]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4 style="color: white; margin: 0; font-weight: 600;">{risk_level}</h4>
                                <h2 style="color: #ffffff; margin: 0.5rem 0; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff4757, #ff3838); color: white; padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 12px 40px rgba(255, 71, 87, 0.3);">
                    <h3 style="margin-bottom: 1rem; font-weight: 700;">⚠️ Analysis Error</h3>
                    <p style="margin-bottom: 1rem;">We encountered an issue processing your data: {str(e)}</p>
                    <p style="margin: 0;">Please check your inputs and try again, or contact support if the issue persists.</p>
                </div>
                """, unsafe_allow_html=True)

# === LIFESTYLE TIPS SECTION ===
st.markdown("""
<div class="tips-container fade-in-up">
    <h2>🧘 Evidence-Based Prevention Strategies</h2>
    <p style="text-align: center; font-size: 1.2rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 2rem; line-height: 1.6;">
        Discover scientifically-backed lifestyle changes that can help optimize your cognitive health
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced tips with modern cards
tips = [
    {
        "icon": "🫐",
        "title": "Brain-Healthy Nutrition",
        "tip": "Follow a Mediterranean diet rich in leafy greens, berries, fatty fish, nuts, and olive oil. These foods contain antioxidants and omega-3 fatty acids that support brain health and neuroplasticity.",
        "color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    },
    {
        "icon": "🚶",
        "title": "Regular Physical Exercise",
        "tip": "Engage in at least 150 minutes of moderate aerobic exercise weekly. Activities like walking, swimming, or dancing improve blood flow to the brain and promote cognitive function.",
        "color": "linear-gradient(135deg, #2ed573 0%, #1dd1a1 100%)"
    },
    {
        "icon": "🧩",
        "title": "Cognitive Stimulation",
        "tip": "Challenge your brain regularly with puzzles, reading, learning new languages, or playing musical instruments. Mental stimulation builds cognitive reserve and neural connections.",
        "color": "linear-gradient(135deg, #ffa502 0%, #ff6348 100%)"
    },
    {
        "icon": "👫",
        "title": "Social Engagement",
        "tip": "Maintain strong social connections through family time, friendships, community activities, or volunteering. Social interaction protects against cognitive decline and isolation.",
        "color": "linear-gradient(135deg, #ff6b9d 0%, #c44569 100%)"
    },
    {
        "icon": "😴",
        "title": "Quality Sleep",
        "tip": "Prioritize 7-9 hours of quality sleep nightly. During sleep, your brain clears toxic proteins associated with Alzheimer's disease and consolidates memories.",
        "color": "linear-gradient(135deg, #a55eea 0%, #8854d0 100%)"
    },
    {
        "icon": "🚭",
        "title": "Avoid Harmful Substances",
        "tip": "Quit smoking and limit alcohol consumption. These substances increase inflammation and oxidative stress, which can damage brain cells over time.",
        "color": "linear-gradient(135deg, #ff4757 0%, #ff3838 100%)"
    },
    {
        "icon": "🩺",
        "title": "Manage Health Conditions",
        "tip": "Keep blood pressure, diabetes, and cholesterol levels under control. Cardiovascular health is directly linked to brain health and cognitive function.",
        "color": "linear-gradient(135deg, #3742fa 0%, #2f3542 100%)"
    },
    {
        "icon": "🧘",
        "title": "Stress Management",
        "tip": "Practice stress-reduction techniques like meditation, yoga, or deep breathing. Chronic stress releases cortisol, which can damage memory centers in the brain.",
        "color": "linear-gradient(135deg, #26de81 0%, #20bf6b 100%)"
    },
    {
        "icon": "🏥",
        "title": "Regular Medical Checkups",
        "tip": "Schedule annual health screenings and discuss cognitive health with your healthcare provider. Early detection and intervention are crucial for brain health.",
        "color": "linear-gradient(135deg, #fd79a8 0%, #e84393 100%)"
    },
    {
        "icon": "🎯",
        "title": "Maintain Life Purpose",
        "tip": "Engage in meaningful activities that give you a sense of purpose. Having goals and staying motivated supports mental well-being and cognitive resilience.",
        "color": "linear-gradient(135deg, #4834d4 0%, #686de0 100%)"
    }
]

# Create modern tip cards with interactive elements
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("💡 Get Personalized Tip", use_container_width=True, 
                 help="Click for a targeted health recommendation"):
        selected_tip = random.choice(tips)
        with col2:
            st.markdown(f"""
            <div style="background: {selected_tip['color']}; padding: 2rem; border-radius: 20px; color: white; box-shadow: 0 12px 40px rgba(0,0,0,0.2); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s ease;" class="fade-in-up">
                <h4 style="margin-bottom: 1rem; font-weight: 700; display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 2rem;">{selected_tip['icon']}</span>
                    {selected_tip['title']}
                </h4>
                <p style="margin: 0; font-size: 1.1rem; line-height: 1.7; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{selected_tip['tip']}</p>
            </div>
            """, unsafe_allow_html=True)

# Show all tips in an enhanced expandable section
with st.expander("📋 View Complete Prevention Guide", expanded=False):
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 16px; margin-bottom: 2rem; backdrop-filter: blur(10px);">
        <p style="color: rgba(255, 255, 255, 0.9); text-align: center; margin: 0; font-size: 1.1rem;">
            🎯 <strong>Comprehensive Brain Health Strategies</strong><br>
            Evidence-based recommendations for optimal cognitive wellness
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a modern grid layout for tips
    for i in range(0, len(tips), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(tips):
                tip = tips[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background: {tip['color']}; padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem; height: 220px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 8px 25px rgba(0,0,0,0.15); border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s ease;">
                        <h5 style="margin: 0 0 1.2rem 0; display: flex; align-items: center; gap: 0.75rem; font-weight: 700;">
                            <span style="font-size: 1.8rem;">{tip['icon']}</span>
                            {tip['title']}
                        </h5>
                        <p style="margin: 0; font-size: 1rem; line-height: 1.6; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{tip['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# === ENHANCED SIDEBAR INFORMATION ===
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(20px); padding: 2rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 2rem;">
        <h3 style="color: white; text-align: center; margin-bottom: 1.5rem; font-weight: 700;">📊 Advanced AI Assessment</h3>
        <div style="color: rgba(255, 255, 255, 0.95); line-height: 1.7;">
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <p style="margin: 0;"><strong>🤖 AI Technology:</strong><br>XGBoost machine learning model</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <p style="margin: 0;"><strong>📈 Comprehensive Analysis:</strong><br>24 evidence-based risk factors</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <p style="margin: 0;"><strong>🔬 Research-Based:</strong><br>Peer-reviewed medical literature</p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px;">
                <p style="margin: 0;"><strong>🎯 Personalized:</strong><br>Tailored cognitive insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff4757, #ff3838); padding: 2rem; border-radius: 20px; color: white; margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(255, 71, 87, 0.3);">
        <h4 style="margin: 0 0 1.2rem 0; text-align: center; font-weight: 700;">⚠️ Medical Disclaimer</h4>
        <p style="margin: 0; font-size: 0.95rem; line-height: 1.6; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            This AI tool provides educational insights based on research data and should never replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(20px); padding: 2rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.25);">
        <h4 style="color: white; text-align: center; margin-bottom: 1.5rem; font-weight: 700;">🔗 Trusted Resources</h4>
        <div style="color: rgba(255, 255, 255, 0.95); display: flex; flex-direction: column; gap: 1rem;">
            <a href="https://www.alz.org" target="_blank" style="color: white; text-decoration: none; background: rgba(255, 255, 255, 0.1); padding: 0.8rem; border-radius: 10px; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem;">
                🏥 Alzheimer's Association
            </a>
            <a href="https://www.nia.nih.gov" target="_blank" style="color: white; text-decoration: none; background: rgba(255, 255, 255, 0.1); padding: 0.8rem; border-radius: 10px; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem;">
                🔬 National Institute on Aging
            </a>
            <a href="https://www.cdc.gov/aging/aginginfo/alzheimers.htm" target="_blank" style="color: white; text-decoration: none; background: rgba(255, 255, 255, 0.1); padding: 0.8rem; border-radius: 10px; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem;">
                📋 CDC Alzheimer's Resources
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Modern footer with enhanced styling
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 24px; text-align: center; margin-top: 3rem; backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 16px 50px rgba(0, 0, 0, 0.1);">
    <h4 style="color: white; margin: 0 0 1rem 0; font-weight: 800; font-size: 1.4rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🧠 Alzheimer's Risk Assessment Tool</h4>
    <p style="color: rgba(255, 255, 255, 0.9); margin: 0 0 1.5rem 0; font-size: 1rem; line-height: 1.6;">
        Developed by <strong>Chenwei Pan</strong> • Powered by Advanced Machine Learning • For Educational Purposes
    </p>
    <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.3);">
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 0.9rem; line-height: 1.5;">
            This application represents cutting-edge research in computational healthcare and should be used alongside professional medical guidance.
        </p>
    </div>
</div>
""", unsafe_allow_html=True))
                    
                elif 'low' in label_str:
                    st.markdown(f"""
                    <div class="result-low-risk fade-in-up">
                        <h2 style="margin-bottom: 1rem; font-weight: 800;">✅ Positive Health Assessment</h2>
                        <h3 style="margin-bottom: 1.5rem; font-weight: 600;">Analysis Result: {label}</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 1.5rem; line-height: 1.6;">
                            Excellent news! Your current health profile indicates favorable risk factors. 
                            Continue maintaining your healthy lifestyle habits and regular medical checkups 
                            to preserve and enhance your cognitive health.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.15); padding: 1.25rem; border-radius: 12px; backdrop-filter: blur(10px);">
                            <strong>🌟 Keep Up the Great Work:</strong> Your healthy choices are making a positive impact on your brain health journey!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif 'moderate' in label_str:
                    st.markdown(f"""
                    <div class="result-moderate-risk fade-in-up">
                        <h2 style="margin-bottom: 1rem; font-weight: 800;">🔶 Moderate Risk Assessment</h2>
                        <h3 style="margin-bottom: 1.5rem; font-weight: 600;">Analysis Result: {label}</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 1.5rem; line-height: 1.6;">
                            Your assessment shows moderate risk factors that warrant attention. 
                            This is an excellent opportunity to implement preventive strategies 
                            and discuss your results with healthcare providers.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.15); padding: 1.25rem; border-radius: 12px; backdrop-filter: blur(10px);">
                            <strong>💪 Take Proactive Action:</strong> Small changes now can make a significant difference in your future cognitive health outcomes.
                        </div>
                    </div>
                    """, unsafe_allow_html=True
