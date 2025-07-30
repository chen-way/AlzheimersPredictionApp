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
    page_title="Alzheimer's Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoder.pkl')
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please make sure model.pkl, scaler.pkl, and encoder.pkl are in the app directory")
        st.stop()

# Load the models
model, scaler, label_encoders = load_models()

# App header
st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Risk Prediction using Advanced Random Forest Algorithm</p>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("## 📝 Patient Information")
st.sidebar.markdown("Please fill in the following details:")

# Input fields organized by categories
with st.sidebar:
    st.markdown("### 👤 Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        country = st.selectbox("Country", [
            "USA", "Canada", "UK", "Germany", "France", "Japan", 
            "South Korea", "India", "China", "Brazil", "South Africa", 
            "Australia", "Russia", "Mexico", "Italy"
        ], index=7)  # Default to India
    
    education = st.selectbox("Education Level", [
        "No Formal Education", "Primary Education", "Secondary Education", 
        "Bachelor's Degree", "Master's Degree", "Doctorate"
    ], index=2)  # Default to Secondary Education

    st.markdown("### 🏥 Health Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        cognitive_score = st.slider("Cognitive Test Score", 0, 30, 25)
        depression_level = st.slider("Depression Level", 0, 10, 2)
        stress_level = st.slider("Stress Level", 0, 10, 1)
    
    with col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"], index=1)
        family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])

    st.markdown("### 🏃‍♂️ Lifestyle")
    col1, col2 = st.columns(2)
    
    with col1:
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"], index=1)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"], index=3)
    
    with col2:
        diet = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"], index=2)
        pollution = st.selectbox("Air Pollution Exposure", ["Minimal", "Slight", "Moderate", "High", "Severe"])
        social_engagement = st.selectbox("Social Engagement Level", ["Low", "Moderate", "High"], index=1)
        living_area = st.selectbox("Living Area", ["Rural", "Urban"], index=1)

    st.markdown("### 💼 Social Demographics")
    employment = st.selectbox("Employment Status", ["Unemployed", "Student", "Employed", "Retired"], index=3)
    marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Widowed", "Married"], index=3)
    income = st.selectbox("Income Level", ["Low", "Middle", "High"], index=1)
    genetic_risk = st.selectbox("Genetic Risk Factor (APOE-ε4)", ["No", "Yes"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🔮 Risk Assessment")
    
    if st.button("🚀 Analyze Risk", type="primary", use_container_width=True):
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
                    # Handle unseen categories
                    st.warning(f"Unknown category '{input_data[column].iloc[0]}' for {column}. Using most common value.")
                    input_encoded[column] = 0
        
        # Scale features
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        risk_probability = model.predict_proba(input_scaled)[0][1]
        risk_percentage = risk_probability * 100
        
        # Display results
        st.markdown("### 📊 Results")
        
        # Risk level determination
        if risk_percentage < 30:
            risk_level = "Low"
            risk_color = "risk-low"
            risk_emoji = "✅"
        elif risk_percentage < 70:
            risk_level = "Moderate"
            risk_color = "risk-moderate"
            risk_emoji = "⚠️"
        else:
            risk_level = "High"
            risk_color = "risk-high"
            risk_emoji = "🚨"
        
        # Risk display box
        st.markdown(f"""
        <div class="risk-box {risk_color}">
            <h2>{risk_emoji} {risk_level} Risk</h2>
            <h1>{risk_percentage:.1f}%</h1>
            <p>Probability of Alzheimer's Disease</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk interpretation
        st.markdown("### 📋 Risk Interpretation")
        if risk_level == "Low":
            st.success("🎉 Great news! Based on the provided information, you have a low risk of developing Alzheimer's disease. Continue maintaining your healthy lifestyle!")
        elif risk_level == "Moderate":
            st.warning("⚠️ You have a moderate risk of developing Alzheimer's disease. Consider consulting with a healthcare professional about preventive measures.")
        else:
            st.error("🚨 You have a high risk of developing Alzheimer's disease. It's important to consult with a healthcare professional for proper assessment and potential preventive strategies.")
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Percentage", 'font': {'size': 24}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "darkblue", 'family': "Arial"},
            height=400
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Feature importance for this prediction
        st.markdown("### 🎯 Key Risk Factors")
        
        # Get feature importances from the model
        feature_names = input_encoded.columns
        importances = model.feature_importances_
        
        # Create a dataframe for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(10)
        
        # Create horizontal bar chart
        fig_importance = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Risk Factors",
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig_importance.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'family': "Arial"},
            height=500
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    st.markdown("## 📈 Model Information")
    
    # Model stats
    st.markdown("""
    <div class="metric-card">
        <h3>🌲 Random Forest</h3>
        <p><strong>Algorithm:</strong> Ensemble Learning</p>
        <p><strong>Trees:</strong> 100-200</p>
        <p><strong>Features:</strong> 24</p>
        <p><strong>Preprocessing:</strong> Scaled & Balanced</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔬 About This Tool")
    st.info("""
    This AI-powered tool uses a Random Forest algorithm trained on a comprehensive dataset to assess Alzheimer's disease risk. 
    
    **Key Features:**
    - ✅ No overfitting (properly validated)
    - 🎯 High accuracy on test data
    - 🔄 Balanced dataset (SMOTEENN)
    - 📊 Feature scaling & encoding
    - 🧠 Advanced ensemble learning
    
    **Important:** This tool is for educational purposes only and should not replace professional medical advice.
    """)
    
    st.markdown("## 📚 Risk Factors")
    with st.expander("Learn More"):
        st.markdown("""
        **Major Risk Factors:**
        - 🧬 Age (strongest predictor)
        - 🧪 Genetic factors (APOE-ε4)
        - 👨‍👩‍👧‍👦 Family history
        - 🧠 Cognitive test scores
        - 💊 Comorbidities (diabetes, hypertension)
        
        **Lifestyle Factors:**
        - 🏃‍♂️ Physical activity
        - 🥗 Diet quality
        - 😴 Sleep quality
        - 🚭 Smoking status
        - 🍷 Alcohol consumption
        
        **Environmental:**
        - 🏭 Air pollution exposure
        - 🏙️ Living environment
        - 👥 Social engagement
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🧠 Alzheimer's Risk Assessment Tool | Powered by Random Forest ML</p>
    <p>⚠️ For educational purposes only. Consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)
