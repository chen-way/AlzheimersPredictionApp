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

# IMPROVED CSS WITH LIGHTER GRADIENT AND BETTER TEXT VISIBILITY
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Lighter modern glassmorphism background */
    .stApp {
        background: linear-gradient(135deg, #a8b5ff 0%, #c4b5fd 50%, #e0e7ff 100%) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .main {
        background: transparent !important;
        padding: 0.5rem !important;
    }
    
    /* Glassmorphism header with floating effect */
    .main-header {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        padding: 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2) !important;
        transform: translateY(0);
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .main-header:hover::before {
        left: 100%;
    }
    
    .main-header:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.3) !important;
    }
    
    .main-header h1 {
        color: #1e293b !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        background: linear-gradient(45deg, #1e293b, #475569);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        color: #475569 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        text-shadow: none !important;
    }
    
    /* Floating glassmorphism containers with better contrast */
    .feature-container, .tips-container {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15) !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    
    .feature-container:hover, .tips-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25) !important;
        border-color: rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Modern input styling with better visibility */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        border-radius: 12px !important;
        border: 2px solid rgba(147, 188, 220, 0.4) !important;
        transition: all 0.3s ease !important;
        min-height: 48px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), 0 4px 20px rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 1) !important;
    }

    /* Enhanced text inputs */
    div[data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(147, 188, 220, 0.4) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        padding: 12px !important;
        min-height: 48px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
    }

    div[data-baseweb="input"] input {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        color: #1e293b !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), 0 4px 20px rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 1) !important;
    }

    /* Fixed number inputs alignment */
    div[data-testid="stNumberInput"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    div[data-testid="stNumberInput"] > div {
        border: 2px solid rgba(147, 188, 220, 0.4) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        min-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100% !important;
    }

    div[data-testid="stNumberInput"] > div:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12) !important;
        background: rgba(255, 255, 255, 1) !important;
    }

    div[data-testid="stNumberInput"] input[type="number"] {
        border: none !important;
        background: transparent !important;
        height: 100% !important;
        width: calc(100% - 80px) !important;
        padding: 0 15px !important;
        color: #1e293b !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        outline: none !important;
        margin: 0 !important;
        text-align: left !important;
    }

    div[data-testid="stNumberInput"]:focus-within > div {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), 0 4px 20px rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 1) !important;
    }

    div[data-testid="stNumberInput"] button {
        border: none !important;
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        height: 100% !important;
        width: 40px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        flex-shrink: 0 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: linear-gradient(45deg, #5a67d8, #6b46c1) !important;
        transform: scale(1.05);
    }

    /* Ensure all input containers have consistent alignment */
    .stSelectbox, .stNumberInput, .stTextInput {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    .stSelectbox > div, .stNumberInput > div, .stTextInput > div {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    
    /* Better result containers with improved contrast */
    .result-high-risk, .result-low-risk, .result-moderate-risk {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        animation: slideInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-high-risk {
        background: rgba(254, 242, 242, 0.95);
        color: #991b1b;
        border-color: rgba(248, 113, 113, 0.6);
        box-shadow: 0 8px 32px rgba(248, 113, 113, 0.2);
    }
    
    .result-low-risk {
        background: rgba(240, 253, 244, 0.95);
        color: #14532d;
        border-color: rgba(34, 197, 94, 0.6);
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.2);
    }
    
    .result-moderate-risk {
        background: rgba(255, 251, 235, 0.95);
        color: #92400e;
        border-color: rgba(245, 158, 11, 0.6);
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2);
    }
    
    .result-high-risk h2, .result-low-risk h2, .result-moderate-risk h2 {
        font-size: 1.8rem !important;
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
    }
    
    .result-high-risk h3, .result-low-risk h3, .result-moderate-risk h3 {
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Better sidebar */
    .stSidebar {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Premium button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        border: none !important;
        min-height: 48px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(45deg, #5a67d8 0%, #6b46c1 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Animated loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Enhanced metric styling with better contrast */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        color: #1e293b;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Better progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
        border-radius: 10px !important;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: 200px 0; }
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        .main-header {
            padding: 1.5rem;
        }
        
        .feature-container, .tips-container {
            padding: 1rem;
        }
        
        .result-high-risk h2, .result-low-risk h2, .result-moderate-risk h2 {
            font-size: 1.4rem !important;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #5a67d8, #6b46c1);
    }
    
    /* Much better text styling with high contrast */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1e293b !important;
        text-shadow: none !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #374151 !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }
    
    .stMarkdown strong {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Better expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.85) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-top: none !important;
    }

    /* Mobile-specific adjustments */
    @media (max-width: 480px) {
        .main {
            padding: 0.25rem !important;
        }
        
        .main-header {
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.9rem !important;
        }
        
        .tips-container {
            padding: 0.75rem;
        }
        
        .feature-container {
            padding: 0.75rem;
        }
    }
    
    /* Ensure proper text sizing on mobile */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        line-height: 1.2 !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        line-height: 1.4 !important;
        font-size: 0.95rem !important;
    }
    
    /* Mobile-friendly expander */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    /* Touch-friendly spacing */
    .stSelectbox, .stNumberInput, .stTextInput {
        margin-bottom: 1rem !important;
    }

    /* Better alert/message styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Success/Warning/Error message improvements */
    .stAlert > div {
        color: #1e293b !important;
    }

    /* Better metric display */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
    }

    div[data-testid="metric-container"] > div {
        color: #1e293b !important;
    }

    /* Label improvements */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced mobile-friendly page configuration
st.set_page_config(
    page_title="Alzheimer's Risk Assessment", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"  # Collapsed by default for mobile
)

# Load models
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model_compressed.pkl.gz')
        scaler = joblib.load('scaler_compressed.pkl.gz')
        encoders = joblib.load('encoders_compressed.pkl.gz')
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
    "Family History of Alzheimer’s": ['Yes', 'No'],
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
    "Family History of Alzheimer’s", 'Cognitive Test Score', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels', 'Urban vs Rural Living'
]

def get_user_input():
    user_data = {}
    
    # Mobile-first: single column layout for better mobile experience
    st.markdown("### 📝 Health Information Form")
    st.markdown("*Please fill in all fields for the most accurate assessment*")
    
    # Group related features for better organization on mobile
    feature_groups = {
        "👤 Personal Information": ['Country', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Marital Status', 'Income Level'],
        "🏥 Health Metrics": ['BMI', 'Diabetes', 'Hypertension', 'Cholesterol Level', 'Cognitive Test Score', 'Depression Level'],
        "🏃 Lifestyle Factors": ['Physical Activity Level', 'Smoking Status', 'Alcohol Consumption', 'Sleep Quality', 'Dietary Habits', 'Stress Levels'],
        "🧬 Risk Factors": ["Family History of Alzheimer’s", 'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level'],
        "🌍 Environment": ['Urban vs Rural Living', 'Air Pollution Exposure']
    }
    
    for group_name, features in feature_groups.items():
        with st.expander(group_name, expanded=True):
            # Use single column for mobile compatibility
            for feature in features:
                if feature in CATEGORICAL_OPTIONS:
                    # Categorical features with dropdowns
                    value = st.selectbox(
                        f"**{feature}**:", 
                        options=CATEGORICAL_OPTIONS[feature],
                        key=feature,
                        help=f"Select your {feature.lower()}"
                    )
                    user_data[feature] = value
                elif feature in NUMERICAL_FEATURES:
                    # Numerical features with appropriate ranges and step sizes
                    if feature == 'Age':
                        value = st.number_input(
                            f"**{feature}** (years):", 
                            min_value=10, max_value=120, value=65, step=1, key=feature,
                            help="Your current age in years"
                        )
                    elif feature == 'BMI':
                        value = st.number_input(
                            f"**{feature}** (kg/m²):", 
                            min_value=10.0, max_value=50.0, value=25.0, step=0.1, key=feature,
                            help="Body Mass Index: weight in kg divided by height in meters squared"
                        )
                    elif feature == 'Cognitive Test Score':
                        value = st.number_input(
                            f"**{feature}** (0-30):", 
                            min_value=0, max_value=30, value=25, step=1, key=feature,
                            help="Mini-Mental State Exam score (higher is better)"
                        )
                    elif feature == 'Depression Level':
                        value = st.number_input(
                            f"**{feature}** (0-15, higher = more depressed):", 
                            min_value=0, max_value=15, value=2, step=1, key=feature,
                            help="Depression severity scale (0=none, 15=severe)"
                        )
                    elif feature == 'Stress Levels':
                        value = st.number_input(
                            f"**{feature}** (0-10, higher = more stress):", 
                            min_value=0, max_value=10, value=5, step=1, key=feature,
                            help="Perceived stress level (0=no stress, 10=maximum stress)"
                        )
                    else:
                        value = st.number_input(f"**{feature}**:", key=feature, step=1.0)
                    
                    user_data[feature] = value
    
    return pd.DataFrame([user_data])

def make_prediction(user_input_df):
    try:
        # Get user's age for safety check
        user_age = user_input_df['Age'].iloc[0]
        
        # Under 50 automatic low risk check
        if user_age < 50:
            st.markdown(f"""
            <div class="result-low-risk">
                <h2>✅ Low Risk Assessment</h2>
                <h3>Alzheimer's Risk: Very Low</h3>
                <p>Age under 50 typically indicates very low risk. Continue healthy lifestyle practices!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Maintenance Strategies")
            st.success("""
            **Maintenance Strategies:**
            • Continue current healthy lifestyle practices
            • Maintain regular physical activity and social engagement
            • Keep challenging your brain with new activities
            """)
            return "Low"
        
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
            'Family History of Alzheimer’s': [user_input_df["Family History of Alzheimer’s"].iloc[0]],
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
        
        # Manual encoding with proper Yes/No handling
        input_encoded = input_data.copy()
        encoding_maps = {}
        for feature, options in CATEGORICAL_OPTIONS.items():
            if feature in ['Diabetes', 'Hypertension', "Family History of Alzheimer’s", 'Genetic Risk Factor (APOE-ε4 allele)']:
                # For Yes/No features, ensure 'No'=0 and 'Yes'=1
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
        
        # Display main risk metric - mobile optimized
        st.markdown("### 🎯 Your Risk Assessment")
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
            
            st.markdown("### 💡 Recommendations")
            st.error("""
            **High Priority Actions:**
            • Schedule consultation with healthcare provider
            • Consider neurological evaluation  
            • Implement comprehensive brain-healthy lifestyle changes
            """)
            
        elif alzheimers_risk >= 30:  # Moderate risk
            st.markdown(f"""
            <div class="result-moderate-risk">
                <h2>🔶 Moderate Risk Assessment</h2>
                <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                <p>The model shows moderate risk factors that warrant attention.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Recommendations")
            st.warning("""
            **Moderate Priority Actions:**
            • Increase physical activity and cognitive challenges
            • Adopt brain-healthy diet (Mediterranean/MIND diet)
            • Improve sleep quality and stress management
            """)
            
        else:  # Low risk
            st.markdown(f"""
            <div class="result-low-risk">
                <h2>✅ Low Risk Assessment</h2>
                <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                <p>Your current health profile indicates lower risk factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Recommendations")
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
        
        return "Complete"
        
    except Exception as e:
        st.error(f"❌ **Error during prediction:** {str(e)}")
        return None

# Header
st.markdown("""
<div class="main-header float-animation">
    <h1>🧠 Alzheimer's Risk Assessment</h1>
    <p>AI-powered risk evaluation with personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced legal disclaimer at the top
st.markdown("""
<div style="background-color: #fef3c7; border: 1px solid #f59e0b; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
    <h4 style="color: #92400e; margin-top: 0; font-size: 1rem;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p style="color: #92400e; margin: 0; font-size: 0.85rem;">
        <strong>This tool is for EDUCATIONAL PURPOSES ONLY</strong> and should never be used for actual medical diagnosis. 
        The predictions are based on statistical models and should not replace professional medical evaluation. 
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Information section - mobile optimized
with st.expander("🔬 How it works", expanded=False):
    st.markdown("""
    Our machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations.
    
    **Key Features:**
    - Comprehensive health factor analysis
    - Personalized risk scoring
    - Evidence-based recommendations
    - Mobile-friendly interface
    """)

# Get user input
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎯 Get Your Risk Assessment")

# Mobile-friendly button layout
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
            
            # Clear progress bar
            progress_bar.empty()
            
            # Make prediction
            make_prediction(user_input_df)
                
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ **Error during prediction:** {str(e)}")
            st.error("Please check your inputs and try again.")

# Educational content - mobile optimized
st.markdown("---")
st.markdown("## 📖 Educational Resources")

# Initialize session state variables for tips
if 'brain_tip' not in st.session_state:
    st.session_state.brain_tip = None
if 'lifestyle_tip' not in st.session_state:
    st.session_state.lifestyle_tip = None
if 'show_all_tips' not in st.session_state:
    st.session_state.show_all_tips = False

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

# Mobile-optimized layout: stacked sections
st.markdown("### 🧠 Brain Health Tips")
st.markdown("""
<div class="tips-container">
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li><strong>Stay Physically Active:</strong> Regular exercise increases blood flow to the brain</li>
        <li><strong>Challenge Your Mind:</strong> Learn new skills, read, solve puzzles</li>
        <li><strong>Eat Brain-Healthy Foods:</strong> Mediterranean diet rich in omega-3s</li>
        <li><strong>Get Quality Sleep:</strong> 7-9 hours nightly for memory consolidation</li>
        <li><strong>Stay Social:</strong> Maintain relationships and community connections</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("### ⚠️ Warning Signs to Watch")
st.markdown("""
<div class="tips-container">
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li><strong>Memory Loss:</strong> Forgetting recently learned information</li>
        <li><strong>Planning Problems:</strong> Difficulty with familiar tasks</li>
        <li><strong>Confusion:</strong> Losing track of time or place</li>
        <li><strong>Language Issues:</strong> Trouble finding the right words</li>
        <li><strong>Mood Changes:</strong> Depression, anxiety, or personality changes</li>
    </ul>
    <p style="margin-top: 1rem; margin-bottom: 0;"><strong>If you notice these signs, consult a healthcare professional.</strong></p>
</div>
""", unsafe_allow_html=True)

# Interactive Tips Section - mobile optimized
st.markdown("### 💡 Get Personalized Tips")

# Stack buttons vertically on mobile for better touch experience
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧠 Brain Tip", use_container_width=True):
        st.session_state.brain_tip = random.choice(brain_tips)

with col2:
    if st.button("🌟 Lifestyle Tip", use_container_width=True):
        st.session_state.lifestyle_tip = random.choice(lifestyle_tips)

with col3:
    if st.button("📋 All Tips", use_container_width=True):
        st.session_state.show_all_tips = not st.session_state.show_all_tips

# Display random tips
if st.session_state.brain_tip:
    st.success(f"🧠 **Brain Tip:** {st.session_state.brain_tip}")
if st.session_state.lifestyle_tip:
    st.success(f"🌟 **Lifestyle Tip:** {st.session_state.lifestyle_tip}")

# Show all tips section - mobile friendly
if st.session_state.show_all_tips:
    st.markdown("---")
    
    st.markdown("#### 🧠 All Brain Health Tips")
    for i, tip in enumerate(brain_tips, 1):
        st.write(f"{i}. {tip}")
    
    st.markdown("#### 🌟 All Lifestyle Tips")
    for i, tip in enumerate(lifestyle_tips, 1):
        st.write(f"{i}. {tip}")

# Footer with additional resources - mobile optimized
st.markdown("---")
st.markdown("### 🌟 Take Control of Your Brain Health")

st.markdown("""
<div class="tips-container">
    <p style="margin-bottom: 1rem;">
        Knowledge is power. Use these insights to make informed decisions about your health and lifestyle. 
        Remember, many risk factors for Alzheimer's disease are modifiable through healthy choices.
    </p>
    <h4 style="margin-bottom: 0.5rem;">📚 Useful Resources:</h4>
    <p style="margin-bottom: 0.25rem;">• <strong>Alzheimer's Association:</strong> <a href="https://alz.org" target="_blank" style="color: #667eea;">alz.org</a></p>
    <p style="margin-bottom: 0.25rem;">• <strong>National Institute on Aging:</strong> <a href="https://nia.nih.gov" target="_blank" style="color: #667eea;">nia.nih.gov</a></p>
    <p style="margin-bottom: 0;">• <strong>Brain Health Research:</strong> <a href="https://brainhealthregistry.org" target="_blank" style="color: #667eea;">brainhealthregistry.org</a></p>
</div>
""", unsafe_allow_html=True)

# Final disclaimer
st.markdown("---")
st.info("""
💡 **Remember:** This tool is for educational purposes only. Always consult with healthcare professionals 
for medical advice, diagnosis, or treatment decisions. Early detection and lifestyle modifications can 
make a significant difference in brain health outcomes.
""")
