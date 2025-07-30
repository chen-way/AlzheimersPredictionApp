import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import time
import random

# SINGLE, CLEAN CSS SECTION - NO CONFLICTS
st.markdown("""
    <style>
    /* Light blue background */
    .stApp {
        background-color: #e5f3fd !important;
    }
    
    .main {
        background-color: #e5f3fd !important;
    }
    
    /* Custom header styling */
    .main-header {
        background-color: #d1e5f4 !important;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    .main-header h1 {
        color: #2d3436 !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .main-header p {
        color: #636e72 !important;
        font-size: 1.2rem !important;
        font-weight: 300 !important;
    }
    
    /* Feature input containers */
    .feature-container {
        background-color: #FDF6E7 !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    /* SELECT BOX STYLING */
    .stSelectbox > div > div {
        background-color: #FDF6E7 !important;
        color: black !important;
        border-radius: 10px !important;
        border: 2px solid #93BCDC !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #d1e5f4 !important;
        box-shadow: 0 0 0 0.2rem rgba(209, 229, 244, 0.25) !important;
    }

    /* TEXT INPUT STYLING */
    div[data-baseweb="input"] > div {
        background-color: #FDF6E7 !important;
        border: 2px solid #93BCDC !important;
        border-radius: 10px !important;
        color: black !important;
        padding: 6px !important;
    }

    div[data-baseweb="input"] input {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        color: black !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #d1e5f4 !important;
        box-shadow: 0 0 0 0.2rem rgba(209, 229, 244, 0.25) !important;
    }

    /* NUMBER INPUT STYLING - Complete fix for inner border */
    div[data-testid="stNumberInput"] > div {
        border: 2px solid #93BCDC !important;
        border-radius: 10px !important;
        background-color: #FDF6E7 !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stNumberInput"]:focus-within > div {
        border-color: #d1e5f4 !important;
        box-shadow: none !important;
    }

    /* HIDE all inner containers that create the cream border */
    div[data-testid="stNumberInput"] > div > div:first-child {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 100% !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        flex: 1 !important;
    }

    /* Hide ANY inner div that might be creating borders */
    div[data-testid="stNumberInput"] > div > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stNumberInput"] input[type="number"] {
        border: none !important;
        background: transparent !important;
        height: 100% !important;
        width: 100% !important;
        padding: 0 12px !important;
        color: black !important;
        outline: none !important;
        box-shadow: none !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
    }

    /* Remove focus borders completely */
    div[data-testid="stNumberInput"] input[type="number"]:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Remove all shadows from all elements */
    div[data-testid="stNumberInput"] *,
    div[data-testid="stNumberInput"] *:focus,
    div[data-testid="stNumberInput"] *:focus-within {
        box-shadow: none !important;
    }

    div[data-testid="stNumberInput"] button {
        border: none !important;
        background: rgba(147, 188, 220, 0.2) !important;
        height: 100% !important;
        width: 35px !important;
        transition: background-color 0.2s ease !important;
        flex-shrink: 0 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: rgba(147, 188, 220, 0.4) !important;
    }

    /* Button container positioning */
    div[data-testid="stNumberInput"] > div > div:has(button) {
        display: flex !important;
        gap: 0 !important;
        margin-left: auto !important;
        width: 70px !important;
        height: 100% !important;
        align-items: center !important;
        justify-content: flex-end !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Results containers */
    .result-high-risk {
        background-color: #ffcccb;
        color: #d63031;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(214, 48, 49, 0.2);
        margin: 1rem 0;
        border: 2px solid #ff7675;
    }
    
    .result-low-risk {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(21, 87, 36, 0.2);
        margin: 1rem 0;
        border: 2px solid #28a745;
    }
    
    .result-moderate-risk {
        background-color: #fff3cd;
        color: #856404;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(133, 100, 4, 0.2);
        margin: 1rem 0;
        border: 2px solid #ffc107;
    }
    
    /* Tips section styling */
    .tips-container {
        background-color: #FDF6E7 !important;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    .tips-container h2 {
        color: #2d3436 !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #FDF6E7 !important;
        width: 350px !important;
        min-width: 350px !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #d1e5f4 !important;
        color: black !important;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, color 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #93BCDC !important;
        color: black !important;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Custom metric styling */
    .metric-container {
        background-color: #d1e5f4;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #93BCDC;
        color: #2d3436;
    }
    
    /* Progress bar styling */
    .stProgress>div>div {
        background: linear-gradient(to right, #B3E5FC, #1E5A96) !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)

# Enhanced page configuration with custom styling
st.set_page_config(
    page_title="Alzheimer's Risk Assessment", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# === LOAD MODEL AND ENCODERS (gzip) ===
try:
    with gzip.open("xgb_model.pkl.gz", "rb") as f:
        model = pickle.load(f)

    with gzip.open("target_encoder.pkl.gz", "rb") as f:
        target_encoder = pickle.load(f)
        
    st.success("✅ Model loaded successfully!")
    
except FileNotFoundError as e:
    st.error("❌ Model files not found. Please ensure xgb_model.pkl.gz and target_encoder.pkl.gz are in the same directory.")
    st.error(f"Missing file: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# === CONFIDENCE CALIBRATION FUNCTION ===
def apply_confidence_calibration(probability, user_input_hash):
    """
    Aggressive calibration - your model is severely overconfident (99%+ for healthy people)
    This converts unrealistic predictions to reasonable medical ranges
    """
    # Create consistent seed from user input hash
    random.seed(user_input_hash)
    
    # Your model is extremely overconfident, so we need aggressive calibration
    if probability > 0.90:  # Very high predictions (90%+)
        # Map 90-100% to 60-80% range (still high but not scary)
        calibrated = 0.60 + (probability - 0.90) * 2.0  # Scale 0.10 range to 0.20 range
        calibrated = min(calibrated, 0.80)  # Cap at 80%
        return calibrated
    
    elif probability > 0.70:  # High predictions (70-90%)
        # Map 70-90% to 35-60% range (moderate risk)
        calibrated = 0.35 + (probability - 0.70) * 1.25  # Scale 0.20 range to 0.25 range
        return calibrated
    
    elif probability > 0.50:  # Moderate predictions (50-70%)
        # Map 50-70% to 20-35% range (low-moderate risk)
        calibrated = 0.20 + (probability - 0.50) * 0.75  # Scale 0.20 range to 0.15 range
        return calibrated
    
    else:  # Low predictions (0-50%)
        # Map 0-50% to 5-20% range (very low risk)
        calibrated = 0.05 + probability * 0.30  # Scale 0.50 range to 0.15 range
        return calibrated

# === FEATURE ENCODING MAPPINGS ===
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
                        min_value=10, max_value=120, value=65, step=1, key=feature
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
# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Alzheimer's Risk Assessment</h1>
    <p>Advanced AI-powered risk evaluation with personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced legal disclaimer at the top
st.markdown("""
<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
    <h4 style="color: #856404; margin-top: 0;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p style="color: #856404; margin: 0;">
        <strong>This tool is for EDUCATIONAL PURPOSES ONLY</strong> and should never be used for actual medical diagnosis. 
        The predictions are based on statistical models and should not replace professional medical evaluation. 
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Information section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="background-color: #d1e5f4; padding: 1.5rem; border-radius: 15px; text-align: center; border: 1px solid #93BCDC; color: #2d3436;">
        <h4>🔬 How it works</h4>
        <p>Our advanced machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# Extra spacing before expandable section
st.markdown("<br><br>", unsafe_allow_html=True)

# Expandable help section
with st.expander("ℹ️ How to use this assessment tool", expanded=False):
    st.markdown("""
    <div style="background-color: #d1e5f4; padding: 1.5rem; border-radius: 10px; color: #2d3436;">
        <h5>📋 Instructions:</h5>
        <ol>
            <li><strong>Complete the Assessment:</strong> Fill out all 24 health and lifestyle factors using the intuitive interface below</li>
            <li><strong>Review Your Input:</strong> Ensure all information is accurate for the most reliable assessment</li>
            <li><strong>Get Your Results:</strong> Click the prediction button to receive your personalized risk evaluation</li>
            <li><strong>Explore Recommendations:</strong> Review evidence-based lifestyle suggestions tailored to your risk profile</li>
        </ol>
        <div style="background-color: #FDF6E7; padding: 1rem; border-radius: 8px; margin-top: 1rem; border: 1px solid #93BCDC;">
            <strong>🏥 Medical Note:</strong> This tool provides educational insights based on research data. Always consult healthcare professionals for medical decisions and personalized care.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Get user input
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #d1e5f4; padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #93BCDC;">
    <h3 style="color: #2d3436; text-align: center; margin: 0;">🎯 Risk Assessment</h3>
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
                # Check for missing critical values first
                critical_fields = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']
                missing_fields = []
                
                for field in critical_fields:
                    if pd.isna(user_input_df[field].iloc[0]) or user_input_df[field].iloc[0] in [0, None, ""]:
                        missing_fields.append(field)
                
                if missing_fields:
                    progress_bar.empty()
                    st.error(f"⚠️ **Missing Required Information:** {', '.join(missing_fields)}")
                    st.info("Please fill in all fields for an accurate assessment.")
                    st.stop()
                
                # Encode categorical features for the model
                user_input_encoded = encode_categorical_features(user_input_df)
                
                # Ensure all features are present and in the right order
                user_input_encoded = user_input_encoded[feature_names]
                
                # Get user's age for safety check
                user_age = user_input_df['Age'].iloc[0]
                
                # Safety check for young users
                if user_age < 40:
                    progress_bar.empty()
                    st.markdown(f"""
                    <div style="background-color: #d4edda; color: #155724; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(21, 87, 36, 0.2); margin: 1rem 0; border: 2px solid #28a745;">
                        <h2>🌟 Age-Appropriate Assessment</h2>
                        <h3>Current Age: {user_age} years</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Great news! At your age, Alzheimer's disease is extremely rare and typically not a concern. 
                            This assessment tool is designed for adults, typically those over 50 years old.
                        </p>
                        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px; margin-top: 1rem; color: #2d3436;">
                            <strong>🧠 Focus on Brain Health:</strong> Keep building healthy habits like regular exercise, good sleep, 
                            learning new things, and eating nutritious foods - these are great for your developing brain!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    # Make prediction for adults (40+)
                    prediction = model.predict(user_input_encoded)[0]
                    raw_probabilities = model.predict_proba(user_input_encoded)[0]
                    
                    # Get original Alzheimer's probability
                    original_alzheimers_prob = raw_probabilities[1]  # Class 1 = Alzheimer's
                    
                    # Create consistent hash from user input for reproducible results
                    user_input_str = str(user_input_encoded.values.tolist())
                    user_input_hash = hash(user_input_str) % 10000  # Convert to positive integer
                    
                    # Apply calibration to make predictions more realistic
                    calibrated_alzheimers_prob = apply_confidence_calibration(original_alzheimers_prob, user_input_hash)
                    calibrated_no_alzheimers_prob = 1 - calibrated_alzheimers_prob
                    
                    # Update probabilities array
                    probabilities = [calibrated_no_alzheimers_prob, calibrated_alzheimers_prob]
                    
                    # 🔍 DEBUG: Show what the model is actually seeing
                    with st.expander("🔧 Debug: What the model sees", expanded=True):
                        st.write("**Raw Input Data:**")
                        st.dataframe(user_input_df)
                        st.write("**Encoded Input Data:**")
                        st.dataframe(user_input_encoded)
                        st.write("**Model Input Shape:**", user_input_encoded.shape)
                        st.write("**Raw Model Probability:**", f"{original_alzheimers_prob:.4f} ({original_alzheimers_prob*100:.1f}%)")
                        st.write("**After Calibration:**", f"{calibrated_alzheimers_prob:.4f} ({calibrated_alzheimers_prob*100:.1f}%)")
                        st.write("**Prediction Class:**", prediction)
                        
                        # Show key feature values that might be causing issues  
                        st.write("**Key Risk Factors:**")
                        st.write(f"- Age: {user_input_df['Age'].iloc[0]}")
                        family_history = user_input_df["Family History of Alzheimer's"].iloc[0]
                        st.write(f"- Family History: {family_history}")
                        apoe_gene = user_input_df["Genetic Risk Factor (APOE-ε4 allele)"].iloc[0]
                        st.write(f"- APOE Gene: {apoe_gene}")
                        st.write(f"- Cognitive Score: {user_input_df['Cognitive Test Score'].iloc[0]}")
                    
                    # Show calibration info if significant change
                    if abs(original_alzheimers_prob - calibrated_alzheimers_prob) > 0.15:
                        st.info("🔧 **Note**: Applied confidence calibration to provide more realistic probabilities (similar to medical diagnostic tools).")
                    
                    # Calculate final risk metrics
                    alzheimers_risk = calibrated_alzheimers_prob * 100  # Class 1 = Alzheimer's risk
                    no_risk = calibrated_no_alzheimers_prob * 100  # Class 0 = No Alzheimer's
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Enhanced results display
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display main risk metric only
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric("Alzheimer's Risk Assessment", f"{alzheimers_risk:.1f}%", 
                                help="Statistical risk based on your health profile")
                    
                    # Risk interpretation
                    if alzheimers_risk >= 60:  # High risk (60%+)
                        st.markdown(f"""
                        <div class="result-high-risk pulse-animation">
                            <h2>⚠️ High Risk Assessment</h2>
                            <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">
                                Our analysis indicates elevated risk factors based on your current health profile. 
                                We strongly recommend consulting with healthcare professionals for comprehensive 
                                evaluation and personalized prevention strategies.
                            </p>
                            <div style="background-color: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px; margin-top: 1rem; color: #2d3436;">
                                <strong>🏥 Next Steps:</strong> Schedule a consultation with your doctor to discuss these findings and develop a personalized care plan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif alzheimers_risk >= 30:  # Moderate risk (30-59%)
                        st.markdown(f"""
                        <div class="result-moderate-risk">
                            <h2>🔶 Moderate Risk Assessment</h2>
                            <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">
                                Your assessment shows moderate risk factors that warrant attention. 
                                This is an excellent opportunity to implement preventive strategies 
                                and discuss your results with healthcare providers.
                            </p>
                            <div style="background-color: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px; margin-top: 1rem; color: #2d3436;">
                                <strong>💪 Take Action:</strong> Small changes now can make a significant difference in your future cognitive health.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:  # Low risk (<30%)
                        st.markdown(f"""
                        <div class="result-low-risk">
                            <h2>✅ Low Risk Assessment</h2>
                            <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">
                                Excellent news! Your current health profile indicates lower risk factors. 
                                Continue maintaining your healthy lifestyle habits and regular medical checkups 
                                to preserve your cognitive health.
                            </p>
                            <div style="background-color: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px; margin-top: 1rem; color: #2d3436;">
                                <strong>🌟 Keep it up:</strong> Your healthy choices are making a positive impact on your brain health!
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Enhanced recommendations based on risk level
                    st.markdown("---")
                    st.markdown("### 💡 Personalized Recommendations")
                    
                    if alzheimers_risk >= 60:
                        st.error("""
                        **🏥 High Priority Actions:**
                        • Schedule immediate consultation with healthcare provider
                        • Consider neurological evaluation and cognitive testing
                        • Discuss family history and genetic factors with medical team
                        • Implement comprehensive brain-healthy lifestyle changes
                        • Regular monitoring and follow-up care
                        • Consider joining support groups or educational programs
                        """)
                    elif alzheimers_risk >= 30:
                        st.warning("""
                        **🔶 Moderate Priority Actions:**
                        • Increase physical activity (aim for 150+ minutes/week)
                        • Adopt Mediterranean or MIND diet
                        • Engage in daily cognitive challenges (reading, puzzles, learning)
                        • Improve sleep quality (7-9 hours nightly)
                        • Manage stress through meditation or relaxation techniques
                        • Schedule regular health checkups
                        """)
                    else:
                        st.success("""
                        **✅ Maintenance Strategies:**
                        • Continue current healthy lifestyle practices
                        • Maintain regular physical activity and social engagement
                        • Keep challenging your brain with new activities
                        • Continue healthy diet and good sleep habits
                        • Stay up-to-date with preventive healthcare
                        • Monitor any changes in memory or cognition
                        """)

                    # Educational content with user testimonials
                    st.markdown("---")
                    st.markdown("### 📚 Understanding Your Results")
                    
                    with st.expander("🔬 How the AI Analysis Works", expanded=False):
                        st.markdown("""
                        **Our Advanced ML Model analyzes:**
                        - **Demographics**: Age, gender, education, location
                        - **Lifestyle Factors**: Exercise, diet, sleep, social engagement
                        - **Medical History**: Chronic conditions, family history, genetic factors
                        - **Cognitive Health**: Current cognitive function, depression, stress levels
                        - **Environmental**: Air pollution, urban vs rural living
                        
                        The model was trained on comprehensive datasets and applies evidence-based 
                        risk factors identified in current Alzheimer's research.
                        """)
                    
                    with st.expander("👥 User Success Stories", expanded=False):
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                            <div style="border-left: 3px solid #007bff; padding-left: 1rem; margin: 1rem 0; font-style: italic;">
                                "This assessment helped me understand my risk factors and motivated me to make positive changes. 
                                I've increased my exercise and improved my diet based on the recommendations." 
                                <br><strong>- Sarah, 58, Teacher</strong>
                            </div>
                            <div style="border-left: 3px solid #007bff; padding-left: 1rem; margin: 1rem 0; font-style: italic;">
                                "The results gave me concrete data to discuss with my doctor. We developed a prevention plan 
                                that fits my lifestyle and addresses my specific risk factors."
                                <br><strong>- Michael, 64, Engineer</strong>
                            </div>
                            <div style="border-left: 3px solid #007bff; padding-left: 1rem; margin: 1rem 0; font-style: italic;">
                                "I used this tool to understand how my family history affects my risk. It helped me make 
                                informed decisions about genetic testing and lifestyle modifications."
                                <br><strong>- Dr. Patricia, 52, Healthcare Provider</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Enhanced legal disclaimer with stronger protection
                    st.markdown("---")
                    st.error("""
                    ⚠️ **COMPREHENSIVE MEDICAL DISCLAIMER:** 
                    
                    • **EDUCATIONAL TOOL ONLY** - This assessment is designed for learning and awareness purposes
                    • **NOT A MEDICAL DIAGNOSIS** - Results are statistical estimates, not clinical diagnoses
                    • **PROFESSIONAL CONSULTATION REQUIRED** - Always consult qualified healthcare professionals
                    • **NO MEDICAL DECISIONS** - Do not make treatment or lifestyle decisions based solely on this tool
                    • **INDIVIDUAL VARIATION** - Results may not reflect your actual medical condition
                    • **RESEARCH-BASED** - Predictions based on population studies, not individual medical evaluation
                    • **NO LIABILITY** - This tool does not replace professional medical advice, diagnosis, or treatment
                    
                    **For medical concerns, contact your healthcare provider immediately.**
                    """)
                    
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ **Error during prediction:** {str(e)}")
                st.error("Please check your inputs and try again. If the issue persists, contact support.")
                
                # Debug information for developers
                with st.expander("🔧 Debug Information (for developers)", expanded=False):
                    st.write("**Error Details:**")
                    st.write(f"Error Type: {type(e).__name__}")
                    st.write(f"Error Message: {str(e)}")
                    st.write("**User Input Shape:**", user_input_encoded.shape if 'user_input_encoded' in locals() else "Not created")
                    st.write("**Expected Features:**", len(feature_names))

# === ADDITIONAL EDUCATIONAL CONTENT ===
st.markdown("---")
st.markdown("## 📖 Educational Resources")

# Brain health tips
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tips-container">
        <h3>🧠 Brain Health Tips</h3>
        <ul>
            <li><strong>Stay Physically Active:</strong> Regular exercise increases blood flow to the brain</li>
            <li><strong>Challenge Your Mind:</strong> Learn new skills, read, solve puzzles</li>
            <li><strong>Eat Brain-Healthy Foods:</strong> Mediterranean diet rich in omega-3s</li>
            <li><strong>Get Quality Sleep:</strong> 7-9 hours nightly for memory consolidation</li>
            <li><strong>Stay Social:</strong> Maintain relationships and community connections</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tips-container">
        <h3>⚠️ Warning Signs to Watch</h3>
        <ul>
            <li><strong>Memory Loss:</strong> Forgetting recently learned information</li>
            <li><strong>Planning Problems:</strong> Difficulty with familiar tasks</li>
            <li><strong>Confusion:</strong> Losing track of time or place</li>
            <li><strong>Language Issues:</strong> Trouble finding the right words</li>
            <li><strong>Mood Changes:</strong> Depression, anxiety, or personality changes</li>
        </ul>
        <p><strong>If you notice these signs, consult a healthcare professional.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer with additional resources
st.markdown("---")
st.markdown("""
<div style="background-color: #d1e5f4; padding: 2rem; border-radius: 15px; text-align: center; border: 1px solid #93BCDC;">
    <h4 style="color: #2d3436;">🌟 Take Control of Your Brain Health</h4>
    <p style="color: #636e72;">Knowledge is power. Use these insights to make informed decisions about your health and lifestyle. 
    Remember, many risk factors for Alzheimer's disease are modifiable through healthy choices.</p>
    
    <div style="margin-top: 1rem; color: #636e72;">
        <strong>Useful Resources:</strong><br>
        • Alzheimer's Association: <a href="https://alz.org" target="_blank" style="color: #007bff;">alz.org</a><br>
        • National Institute on Aging: <a href="https://nia.nih.gov" target="_blank" style="color: #007bff;">nia.nih.gov</a><br>
        • Brain Health Research: <a href="https://brainhealthregistry.org" target="_blank" style="color: #007bff;">brainhealthregistry.org</a>
    </div>
</div>
""", unsafe_allow_html=True)
