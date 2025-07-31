import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS - EXACT same styling as XGBoost version
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
        border: 3px solid rgba(255,255,255,0.1);
    }
    
    .risk-card h1 {
        font-size: 4rem;
        margin: 0;
        font-weight: 700;
    }
    
    .risk-card h3 {
        font-size: 1.5rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model_compressed.pkl.gz')
        scaler = joblib.load('scaler_compressed.pkl.gz')
        encoders = joblib.load('encoders_compressed.pkl.gz')
        st.success("✅ All models loaded successfully!")
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Looking for: model_compressed.pkl.gz, scaler_compressed.pkl.gz, encoders_compressed.pkl.gz")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load the models
model, scaler, label_encoders = load_models()

# Main header - EXACT same as XGBoost version
st.markdown('<h1 class="main-title">🧠 Alzheimer\'s Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-Powered Risk Assessment Using Machine Learning</p>', unsafe_allow_html=True)

# Layout - EXACT same structure
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📋 Patient Information")
    
    # Demographics section - SAME layout
    st.markdown("#### 👤 Demographics")
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        country = st.selectbox("Country", [
            "USA", "Canada", "UK", "Germany", "France", "Japan", 
            "South Korea", "India", "China", "Brazil", "South Africa", 
            "Australia", "Russia", "Mexico", "Italy"
        ], index=7)
    
    with demo_col2:
        education = st.selectbox("Education Level", [
            "No Formal Education", "Primary Education", "Secondary Education", 
            "Bachelor's Degree", "Master's Degree", "Doctorate"
        ], index=2)
        employment = st.selectbox("Employment Status", [
            "Unemployed", "Student", "Employed", "Retired"
        ], index=3)
        marital_status = st.selectbox("Marital Status", [
            "Single", "Divorced", "Widowed", "Married"
        ], index=3)

    # Health Metrics - SAME layout
    st.markdown("#### 🏥 Health Metrics")
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        cognitive_score = st.slider("Cognitive Test Score", 0, 30, 25)
        depression_level = st.slider("Depression Level", 0, 10, 2)
        stress_level = st.slider("Stress Level", 0, 10, 1)
    
    with health_col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"], index=1)
        family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])

    # Lifestyle - SAME layout
    st.markdown("#### 🏃‍♂️ Lifestyle Factors")
    lifestyle_col1, lifestyle_col2 = st.columns(2)
    
    with lifestyle_col1:
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"], index=1)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"], index=3)
    
    with lifestyle_col2:
        diet = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"], index=2)
        pollution = st.selectbox("Air Pollution Exposure", [
            "Minimal", "Slight", "Moderate", "High", "Severe"
        ])
        social_engagement = st.selectbox("Social Engagement Level", ["Low", "Moderate", "High"], index=1)
        living_area = st.selectbox("Living Area", ["Rural", "Urban"], index=1)

    # Additional Factors - SAME layout
    st.markdown("#### 🧬 Additional Risk Factors")
    additional_col1, additional_col2 = st.columns(2)
    
    with additional_col1:
        income = st.selectbox("Income Level", ["Low", "Middle", "High"], index=1)
        genetic_risk = st.selectbox("Genetic Risk Factor (APOE-ε4)", ["No", "Yes"])
    
    with additional_col2:
        st.write("")  # Spacing to match original layout

# Prediction button - SAME styling
st.markdown("<br>", unsafe_allow_html=True)
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("🔮 Predict Alzheimer's Risk", 
                              type="primary", 
                              use_container_width=True)

# Results section - EXACT same layout as XGBoost version
if predict_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Country': [country],
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'BMI': [bmi],
        'Physical Activity Level': [physical_activity],
        'Smoking Status': [smoking],
        'Alcohol Consumption': [alcohol],
        'Diabetes': [diabetes],
        'Hypertension': [hypertension],
        'Cholesterol Level': [cholesterol],
        'Family History of Alzheimer\'s': [family_history],
        'Cognitive Test Score': [cognitive_score],
        'Depression Level': [depression_level],
        'Sleep Quality': [sleep_quality],
        'Dietary Habits': [diet],
        'Air Pollution Exposure': [pollution],
        'Employment Status': [employment],
        'Marital Status': [marital_status],
        'Genetic Risk Factor (APOE-ε4 allele)': [genetic_risk],
        'Social Engagement Level': [social_engagement],
        'Income Level': [income],
        'Stress Levels': [stress_level],
        'Urban vs Rural Living': [living_area]
    })
    
    # Encode categorical variables
    input_encoded = input_data.copy()
    for column in input_data.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            try:
                input_encoded[column] = label_encoders[column].transform(input_data[column])
            except ValueError:
                input_encoded[column] = 0
    
    # Scale features
    input_scaled = scaler.transform(input_encoded)
    
    # Make prediction
    risk_probability = model.predict_proba(input_scaled)[0][1]
    risk_percentage = risk_probability * 100
    
    # Results layout - EXACT same as XGBoost version
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        # Risk card - SAME styling
        st.markdown(f"""
        <div class="risk-card">
            <h3>🎯 Alzheimer's Disease Risk</h3>
            <h1>{risk_percentage:.1f}%</h1>
            <p style="font-size: 1.1rem; margin-top: 1rem;">
                Risk Assessment Based on Current Health Profile
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk interpretation - SAME as XGBoost version
        if risk_percentage < 30:
            st.success("🎉 **Low Risk**: Based on your current health profile, you have a relatively low risk of developing Alzheimer's disease. Keep maintaining your healthy lifestyle!")
        elif risk_percentage < 70:
            st.warning("⚠️ **Moderate Risk**: Your risk level is moderate. Consider discussing preventive measures with your healthcare provider.")
        else:
            st.error("🚨 **High Risk**: Your risk level is elevated. It's important to consult with a healthcare professional for a comprehensive evaluation.")

    # Gauge chart - SAME as XGBoost version
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Percentage", 'font': {'size': 24, 'color': '#333'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#c8e6c9'},
                {'range': [30, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"},
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights - SAME layout as XGBoost version
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div class="metric-container">
            <h4>📊 Key Risk Factors</h4>
            <div class="feature-box">🎂 <strong>Age:</strong> {}</div>
            <div class="feature-box">🧬 <strong>Genetic Risk:</strong> {}</div>
            <div class="feature-box">👨‍👩‍👧‍👦 <strong>Family History:</strong> {}</div>
            <div class="feature-box">🧠 <strong>Cognitive Score:</strong> {}/30</div>
        </div>
        """.format(age, genetic_risk, family_history, cognitive_score), unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="metric-container">
            <h4>💡 Lifestyle Factors</h4>
            <div class="feature-box">🏃‍♂️ <strong>Physical Activity:</strong> {}</div>
            <div class="feature-box">🥗 <strong>Diet Quality:</strong> {}</div>
            <div class="feature-box">😴 <strong>Sleep Quality:</strong> {}</div>
            <div class="feature-box">🚭 <strong>Smoking:</strong> {}</div>
        </div>
        """.format(physical_activity, diet, sleep_quality, smoking), unsafe_allow_html=True)
    
    # Recommendations - SAME as XGBoost version
    st.markdown("""
    <div class="recommendation-box">
        <h4>📋 General Recommendations</h4>
        <ul>
            <li><strong>🧠 Cognitive Health:</strong> Engage in mentally stimulating activities like reading, puzzles, or learning new skills</li>
            <li><strong>🏃‍♂️ Physical Exercise:</strong> Regular aerobic exercise can help maintain brain health</li>
            <li><strong>🥗 Healthy Diet:</strong> Mediterranean-style diet rich in fruits, vegetables, and omega-3 fatty acids</li>
            <li><strong>😴 Quality Sleep:</strong> Maintain 7-9 hours of good quality sleep per night</li>
            <li><strong>👥 Social Engagement:</strong> Stay socially active and maintain strong relationships</li>
            <li><strong>⚕️ Medical Care:</strong> Regular check-ups and management of chronic conditions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer - EXACT same as XGBoost version
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
    <p>🧠 <em>Powered by Advanced Random Forest Machine Learning Algorithm</em></p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        For accurate diagnosis and treatment recommendations, please consult with qualified healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)
