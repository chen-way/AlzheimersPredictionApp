# === LIFESTYLE TIPS SECTION ===
st.markdown("""
<div class="tips-container">
    <h2>🧘 Evidence-Based Prevention Strategies</h2>
    <p style="text-alignimport streamlit as st
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

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem !important;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2rem !important;
        font-weight: 300 !important;
    }
    
    /* Feature input containers */
    .feature-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Prediction button styling */
    .predict-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Results containers */
    .result-high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(238, 90, 36, 0.3);
        margin: 1rem 0;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(32, 191, 107, 0.3);
        margin: 1rem 0;
    }
    
    .result-moderate-risk {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Tips section styling */
    .tips-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .tips-container h2 {
        color: #2d3436 !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #e0e6ed;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #4facfe;
        box-shadow: 0 0 0 0.2rem rgba(79, 172, 254, 0.25);
    }
    
    .stNumberInput > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #e0e6ed;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #4facfe;
        box-shadow: 0 0 0 0.2rem rgba(79, 172, 254, 0.25);
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
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
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
# Custom header with beautiful styling
st.markdown("""
<div class="main-header">
    <h1>🧠 Alzheimer's Risk Assessment</h1>
    <p>Advanced AI-powered risk evaluation with personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced information section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; text-align: center; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); color: white;">
        <h4>🔬 How it works</h4>
        <p>Our advanced machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Enhanced expandable help section
with st.expander("ℹ️ How to use this assessment tool", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); padding: 1.5rem; border-radius: 10px; color: white;">
        <h5>📋 Instructions:</h5>
        <ol>
            <li><strong>Complete the Assessment:</strong> Fill out all 24 health and lifestyle factors using the intuitive interface below</li>
            <li><strong>Review Your Input:</strong> Ensure all information is accurate for the most reliable assessment</li>
            <li><strong>Get Your Results:</strong> Click the prediction button to receive your personalized risk evaluation</li>
            <li><strong>Explore Recommendations:</strong> Review evidence-based lifestyle suggestions tailored to your risk profile</li>
        </ol>
        <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>🏥 Medical Note:</strong> This tool provides educational insights based on research data. Always consult healthcare professionals for medical decisions and personalized care.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Get user input
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
    <h3 style="color: white; text-align: center; margin: 0;">🎯 Risk Assessment</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🧪 Analyze My Alzheimer's Risk", type="primary", use_container_width=True, 
                 help="Click to get your personalized risk assessment"):
        
        # Enhanced loading animation
        with st.spinner("🔍 Analyzing your health data with advanced AI..."):
            # Create a progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
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
                
                # Enhanced results display with beautiful styling
                st.markdown("<br>", unsafe_allow_html=True)
                
                label_str = str(label).lower()
                if 'high' in label_str:
                    st.markdown(f"""
                    <div class="result-high-risk pulse-animation">
                        <h2>⚠️ High Risk Assessment</h2>
                        <h3>Prediction: {label}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Our analysis indicates elevated risk factors based on your current health profile. 
                            We strongly recommend consulting with healthcare professionals for comprehensive 
                            evaluation and personalized prevention strategies.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <strong>🏥 Next Steps:</strong> Schedule a consultation with your doctor to discuss these findings and develop a personalized care plan.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif 'low' in label_str:
                    st.markdown(f"""
                    <div class="result-low-risk">
                        <h2>✅ Low Risk Assessment</h2>
                        <h3>Prediction: {label}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Excellent news! Your current health profile indicates lower risk factors. 
                            Continue maintaining your healthy lifestyle habits and regular medical checkups 
                            to preserve your cognitive health.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <strong>🌟 Keep it up:</strong> Your healthy choices are making a positive impact on your brain health!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif 'moderate' in label_str:
                    st.markdown(f"""
                    <div class="result-moderate-risk">
                        <h2>🔶 Moderate Risk Assessment</h2>
                        <h3>Prediction: {label}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Your assessment shows moderate risk factors that warrant attention. 
                            This is an excellent opportunity to implement preventive strategies 
                            and discuss your results with healthcare providers.
                        </p>
                        <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <strong>💪 Take action:</strong> Small changes now can make a big difference in your future cognitive health.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                        <h2>📊 Assessment Complete</h2>
                        <h3>Prediction: {label}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Please consult with healthcare professionals for proper evaluation and 
                            continue maintaining healthy lifestyle habits.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced confidence scores with beautiful styling
                if len(probability) > 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2);">
                        <h4 style="color: white; text-align: center; margin-bottom: 1rem;">📈 Confidence Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create beautiful metrics display
                    prob_cols = st.columns(len(probability))
                    for i, (risk_level, prob) in enumerate(zip(target_encoder.classes_, probability)):
                        with prob_cols[i]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h4 style="color: white; margin: 0;">{risk_level}</h4>
                                <h2 style="color: #4facfe; margin: 0.5rem 0;">{prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff7675, #d63031); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;">
                    <h3>⚠️ Analysis Error</h3>
                    <p>We encountered an issue processing your data: {str(e)}</p>
                    <p>Please check your inputs and try again, or contact support if the issue persists.</p>
                </div>
                """, unsafe_allow_html=True)

# === LIFESTYLE TIPS SECTION ===
st.markdown("""
<div class="tips-container">
    <h2>🧘 Evidence-Based Prevention Strategies</h2>
    <p style="text-align: center; font-size: 1.1rem; color: #2d3436; margin-bottom: 2rem;">
        Discover scientifically-backed lifestyle changes that can help reduce your Alzheimer's risk
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced tips with beautiful cards
tips = [
    {
        "icon": "🫐",
        "title": "Brain-Healthy Nutrition",
        "tip": "Follow a Mediterranean diet rich in leafy greens, berries, fatty fish, nuts, and olive oil. These foods contain antioxidants and omega-3 fatty acids that support brain health.",
        "color": "linear-gradient(135deg, #74b9ff, #0984e3)"
    },
    {
        "icon": "🚶",
        "title": "Regular Physical Exercise",
        "tip": "Engage in at least 150 minutes of moderate aerobic exercise weekly. Activities like walking, swimming, or dancing improve blood flow to the brain and promote neuroplasticity.",
        "color": "linear-gradient(135deg, #00b894, #00a085)"
    },
    {
        "icon": "🧩",
        "title": "Cognitive Stimulation",
        "tip": "Challenge your brain regularly with puzzles, reading, learning new languages, or playing musical instruments. Mental stimulation builds cognitive reserve.",
        "color": "linear-gradient(135deg, #fdcb6e, #e17055)"
    },
    {
        "icon": "👫",
        "title": "Social Engagement",
        "tip": "Maintain strong social connections through family time, friendships, community activities, or volunteering. Social interaction protects against cognitive decline.",
        "color": "linear-gradient(135deg, #fd79a8, #e84393)"
    },
    {
        "icon": "😴",
        "title": "Quality Sleep",
        "tip": "Prioritize 7-9 hours of quality sleep nightly. During sleep, your brain clears toxic proteins associated with Alzheimer's disease.",
        "color": "linear-gradient(135deg, #a29bfe, #6c5ce7)"
    },
    {
        "icon": "🚭",
        "title": "Avoid Harmful Substances",
        "tip": "Quit smoking and limit alcohol consumption. These substances increase inflammation and damage brain cells over time.",
        "color": "linear-gradient(135deg, #ff7675, #d63031)"
    },
    {
        "icon": "🩺",
        "title": "Manage Health Conditions",
        "tip": "Keep blood pressure, diabetes, and cholesterol levels under control. Cardiovascular health is directly linked to brain health.",
        "color": "linear-gradient(135deg, #55a3ff, #3742fa)"
    },
    {
        "icon": "🧘",
        "title": "Stress Management",
        "tip": "Practice stress-reduction techniques like meditation, yoga, or deep breathing. Chronic stress releases hormones that can damage the brain.",
        "color": "linear-gradient(135deg, #26de81, #20bf6b)"
    },
    {
        "icon": "🏥",
        "title": "Regular Medical Checkups",
        "tip": "Schedule annual health screenings and discuss cognitive health with your healthcare provider. Early detection and intervention are key.",
        "color": "linear-gradient(135deg, #ffa726, #ff9800)"
    },
    {
        "icon": "🎯",
        "title": "Maintain Life Purpose",
        "tip": "Engage in meaningful activities that give you a sense of purpose. Having goals and staying motivated supports mental well-being.",
        "color": "linear-gradient(135deg, #67b26f, #4ca2cd)"
    }
]

# Create beautiful tip cards
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("💡 Get Random Prevention Tip", use_container_width=True, 
                 help="Click for a personalized health recommendation"):
        selected_tip = random.choice(tips)
        with col2:
            st.markdown(f"""
            <div style="background: {selected_tip['color']}; padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h4>{selected_tip['icon']} {selected_tip['title']}</h4>
                <p style="margin: 0; font-size: 1.05rem; line-height: 1.6;">{selected_tip['tip']}</p>
            </div>
            """, unsafe_allow_html=True)

# Show all tips in an enhanced expandable section
with st.expander("📋 View Complete Prevention Guide", expanded=False):
    # Create a grid layout for tips
    for i in range(0, len(tips), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(tips):
                tip = tips[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background: {tip['color']}; padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                        <h5 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.5rem;">{tip['icon']}</span>
                            {tip['title']}
                        </h5>
                        <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">{tip['tip']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# === ENHANCED SIDEBAR INFORMATION ===
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 2rem;">
        <h3 style="color: white; text-align: center; margin-bottom: 1rem;">📊 About This Assessment</h3>
        <div style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
            <p><strong>🤖 AI Technology:</strong> Advanced XGBoost machine learning model</p>
            <p><strong>📈 Comprehensive Analysis:</strong> 24 evidence-based risk factors</p>
            <p><strong>🔬 Research-Based:</strong> Built on peer-reviewed medical literature</p>
            <p><strong>🎯 Personalized:</strong> Tailored insights for your unique profile</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h4 style="margin: 0 0 1rem 0; text-align: center;">⚠️ Important Medical Disclaimer</h4>
        <p style="margin: 0; font-size: 0.9rem; line-height: 1.5;">
            This tool provides educational insights based on research data and should never replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions and personalized care planning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2);">
        <h4 style="color: white; text-align: center; margin-bottom: 1rem;">🔗 Trusted Resources</h4>
        <div style="color: rgba(255, 255, 255, 0.9);">
            <p><a href="https://www.alz.org" target="_blank" style="color: #4facfe; text-decoration: none;">🏥 Alzheimer's Association</a></p>
            <p><a href="https://www.nia.nih.gov" target="_blank" style="color: #4facfe; text-decoration: none;">🔬 National Institute on Aging</a></p>
            <p><a href="https://www.cdc.gov/aging/aginginfo/alzheimers.htm" target="_blank" style="color: #4facfe; text-decoration: none;">📋 CDC Alzheimer's Resources</a></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with enhanced styling
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 3rem;">
    <h4 style="color: white; margin: 0 0 1rem 0;">🧠 Alzheimer's Risk Assessment Tool</h4>
    <p style="color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 0.9rem;">
        Developed by <strong>Chenwei Pan</strong> • Powered by Advanced Machine Learning • For Educational Purposes
    </p>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.3);">
        <p style="color: rgba(255, 255, 255, 0.7); margin: 0; font-size: 0.8rem;">
            This application represents cutting-edge research in computational healthcare and should be used alongside professional medical guidance.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
