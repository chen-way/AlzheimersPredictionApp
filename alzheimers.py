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
except FileNotFoundError:
    st.error("Model files not found. Please ensure xgb_model.pkl.gz and target_encoder.pkl.gz are in the same directory.")
    st.stop()

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
                # Encode categorical features for the model
                user_input_encoded = encode_categorical_features(user_input_df)
                
                # Ensure all features are present and in the right order
                user_input_encoded = user_input_encoded[feature_names]
                
                # Get user's age for safety check
                user_age = user_input_df['Age'].iloc[0]
                
                # Safety check for young users
                if user_age < 40:
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
                    # Make prediction only for adults (40+)
                    prediction = model.predict(user_input_encoded)[0]
                    probability = model.predict_proba(user_input_encoded)[0]
                    
                    # Get label from target encoder and make it more meaningful
                    raw_label = target_encoder.inverse_transform([prediction])[0]
                    
                    # DEBUG SECTION - Remove after fixing
                    st.write("🔍 **DEBUG INFO:**")
                    st.write(f"Raw prediction (class): {prediction}")
                    st.write(f"Raw label from encoder: {raw_label}")
                    st.write(f"Probability array: {probability}")
                    st.write(f"Probability shape: {probability.shape}")
                    st.write(f"Class 0 probability: {probability[0]:.3f} ({probability[0]*100:.1f}%)")
                    st.write(f"Class 1 probability: {probability[1]:.3f} ({probability[1]*100:.1f}%)")
                    st.write(f"Sum of probabilities: {sum(probability):.3f}")
                    st.write(f"Target encoder classes: {target_encoder.classes_}")
                    
                    if prediction == 0:
                        st.write(f"Model predicted class 0, which corresponds to: {target_encoder.classes_[0]}")
                    else:
                        st.write(f"Model predicted class 1, which corresponds to: {target_encoder.classes_[1]}")
                    
                    # Define risk levels based on probability percentages
                    def interpret_prediction_with_thresholds(raw_pred, probabilities):
                        # Get the probability of the positive class (assuming index 1 is high risk)
                        # If binary classification, probabilities[1] is usually the positive class
                        if len(probabilities) == 2:
                            risk_probability = probabilities[1] * 100  # Convert to percentage
                        else:
                            # For multi-class, use the maximum probability
                            risk_probability = max(probabilities) * 100
                        
                        # Classify based on percentage ranges
                        if risk_probability >= 60:
                            return "High Risk", risk_probability
                        elif risk_probability >= 30:
                            return "Moderate Risk", risk_probability
                        else:
                            return "Low Risk", risk_probability
                    
                    label, risk_percentage = interpret_prediction_with_thresholds(raw_label, probability)
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Enhanced results display
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    label_str = str(label).lower()
                    if 'high' in label_str:
                        st.markdown(f"""
                        <div class="result-high-risk pulse-animation">
                            <h2>⚠️ High Risk Assessment</h2>
                            <h3>Prediction: {label} ({risk_percentage:.1f}%)</h3>
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
                        
                    elif 'low' in label_str:
                        st.markdown(f"""
                        <div class="result-low-risk">
                            <h2>✅ Low Risk Assessment</h2>
                            <h3>Prediction: {label} ({risk_percentage:.1f}%)</h3>
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
                        
                    elif 'moderate' in label_str:
                        st.markdown(f"""
                        <div class="result-moderate-risk">
                            <h2>🔶 Moderate Risk Assessment</h2>
                            <h3>Prediction: {label} ({risk_percentage:.1f}%)</h3>
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

                
            except Exception as e:
                st.markdown(f"""
                <div style="background-color: #ffcccb; color: #d63031; padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #ff7675;">
                    <h3>⚠️ Analysis Error</h
