import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import time

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
        
    # Show model information for verification
    st.success("✅ Model loaded successfully!")
    
    # Model verification section
    with st.expander("🔍 Model Information", expanded=False):
        st.write("**Model Type:**", type(model).__name__)
        if hasattr(model, 'n_features_in_'):
            st.write("**Expected Features:**", model.n_features_in_)
        if hasattr(model, 'feature_names_in_'):
            st.write("**Feature Names:**", list(model.feature_names_in_) if model.feature_names_in_ is not None else "None")
        if hasattr(model, 'classes_'):
            st.write("**Model Classes:**", model.classes_)
            
except FileNotFoundError as e:
    st.error("❌ Model files not found. Please ensure xgb_model.pkl.gz and target_encoder.pkl.gz are in the same directory.")
    st.error(f"Missing file: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.error("This might be a version compatibility issue or corrupted model file.")
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
""", unsafe_allow_html=True)_allow_html=True)

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
    if st.button("🧪 Analyze My Alzheimer's Risk", type="primary", use_container_width=True):
        
        with st.spinner("🔍 Processing your health data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                # Check for missing critical values
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
                
                # Make prediction
                prediction = model.predict(user_input_encoded)[0]
                raw_probabilities = model.predict_proba(user_input_encoded)[0]
                
                # Clear progress bar
                progress_bar.empty()
                
                # Debug section to see what's happening
                with st.expander("🔧 Debug: Model Predictions", expanded=True):
                    st.write("**Raw Input Data:**")
                    st.dataframe(user_input_df)
                    st.write("**Encoded Input Data:**")
                    st.dataframe(user_input_encoded)
                    st.write("**Model Input Shape:**", user_input_encoded.shape)
                    st.write("**Raw Model Probabilities:**", raw_probabilities)
                    st.write("**Prediction Class:**", prediction)
                    st.write("**Class 0 (No Alzheimer's):**", f"{raw_probabilities[0]:.4f} ({raw_probabilities[0]*100:.1f}%)")
                    st.write("**Class 1 (Alzheimer's):**", f"{raw_probabilities[1]:.4f} ({raw_probabilities[1]*100:.1f}%)")
                    
                    # Show key features
                    family_history = user_input_df["Family History of Alzheimer's"].iloc[0]
                    apoe_gene = user_input_df["Genetic Risk Factor (APOE-ε4 allele)"].iloc[0]
                    st.write("**Key Risk Factors:**")
                    st.write(f"- Age: {user_input_df['Age'].iloc[0]}")
                    st.write(f"- Family History: {family_history}")
                    st.write(f"- APOE Gene: {apoe_gene}")
                    st.write(f"- Cognitive Score: {user_input_df['Cognitive Test Score'].iloc[0]}")
                
                # Use raw model predictions without artificial calibration
                alzheimers_risk = raw_probabilities[1] * 100  # Class 1 = Alzheimer's risk
                no_risk = raw_probabilities[0] * 100  # Class 0 = No Alzheimer's
                
                # Display main risk metric
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Alzheimer's Risk Assessment", f"{alzheimers_risk:.1f}%", 
                             help="Raw model prediction probability")
                
                # Risk interpretation based on actual model output
                if alzheimers_risk >= 70:  # High risk
                    st.markdown(f"""
                    <div class="result-high-risk">
                        <h2>⚠️ High Risk Assessment</h2>
                        <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                        <p>The model indicates elevated risk factors. Please consult healthcare professionals.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif alzheimers_risk >= 30:  # Moderate risk
                    st.markdown(f"""
                    <div class="result-moderate-risk">
                        <h2>🔶 Moderate Risk Assessment</h2>
                        <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                        <p>The model shows moderate risk factors that warrant attention.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:  # Low risk
                    st.markdown(f"""
                    <div class="result-low-risk">
                        <h2>✅ Low Risk Assessment</h2>
                        <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                        <p>Your current health profile indicates lower risk factors.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendations based on risk level
                st.markdown("---")
                st.markdown("### 💡 Recommendations")
                
                if alzheimers_risk >= 70:
                    st.error("""
                    **High Priority Actions:**
                    • Schedule consultation with healthcare provider
                    • Consider neurological evaluation
                    • Implement comprehensive brain-healthy lifestyle changes
                    """)
                elif alzheimers_risk >= 30:
                    st.warning("""
                    **Moderate Priority Actions:**
                    • Increase physical activity and cognitive challenges
                    • Adopt brain-healthy diet (Mediterranean/MIND diet)
                    • Improve sleep quality and stress management
                    """)
                else:
                    st.success("""
                    **Maintenance Strategies:**
                    • Continue current healthy lifestyle practices
                    • Maintain regular physical activity and social engagement
                    • Keep challenging your brain with new activities
                    """)

                # Legal disclaimer
                st.markdown("---")
                st.error("""
                ⚠️ **MEDICAL DISCLAIMER:** This tool provides educational insights only. 
                Always consult healthcare professionals for medical decisions.
                """)
                    
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ **Error during prediction:** {str(e)}")
                st.error("Please check your inputs and try again.")
                
                # Debug information
                with st.expander("🔧 Debug Information", expanded=False):
                    st.write("**Error Details:**")
                    st.write(f"Error Type: {type(e).__name__}")
                    st.write(f"Error Message: {str(e)}")
                    if 'user_input_encoded' in locals():
                        st.write("**User Input Shape:**", user_input_encoded.shape)
                        st.write("**Feature Names Length:**", len(feature_names))

# Educational content
st.markdown("---")
st.markdown("## 📖 Educational Resources")

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
    """, unsafe
