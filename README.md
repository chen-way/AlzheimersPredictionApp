# 🧠 Alzheimer's Disease Risk Prediction App

**By Chenwei Pan**  

---

## Overview

Alzheimer's disease (AD) is a progressive neurological disorder characterized by memory loss and cognitive decline. Early detection is essential for timely treatment and better patient outcomes, yet traditional diagnostic methods often identify the disease only after significant progression.

This project develops a machine learning-based prediction model using patient health data to estimate the risk of Alzheimer's. The model is integrated into an interactive web app where users can input their health information and receive risk assessments instantly with personalized lifestyle recommendations.

---

## Objective

- Build a reliable Alzheimer's prediction model using XGBoost with advanced feature encoding
- Provide an easy-to-use Streamlit web app for individual risk assessment with user-friendly categorical inputs
- Support comprehensive data preprocessing including target encoding for categorical features
- Deliver meaningful risk interpretations with professional healthcare guidance
- Offer evidence-based lifestyle tips for Alzheimer's prevention

---

## Features Used for Prediction

The model analyzes **24 comprehensive risk factors** across multiple health domains:

### 🔢 Numeric Features
| Feature | Description | Range |
|---------|-------------|-------|
| Age | Patient age in years | 18-120 |
| BMI | Body Mass Index (kg/m²) | 10.0-50.0 |
| Cognitive Test Score | Cognitive assessment score | 0-30 |
| Depression Level | Depression severity scale | 0-15 |
| Stress Levels | Stress assessment scale | 0-10 |

### 🌍 Categorical Features
| Feature | Options |
|---------|---------|
| **Country** | USA, Canada, UK, Germany, France, Japan, South Korea, India, China, Brazil, South Africa, Australia, Russia, Mexico, Italy |
| **Gender** | Male, Female |
| **Education Level** | No Formal Education, Primary Education, Secondary Education, Bachelor's Degree, Master's Degree, Doctorate |
| **Physical Activity Level** | Low, Moderate, High |
| **Smoking Status** | Never, Former, Current |
| **Alcohol Consumption** | None, Moderate, High |
| **Diabetes** | Yes, No |
| **Hypertension** | Yes, No |
| **Cholesterol Level** | Low, Normal, High |
| **Family History of Alzheimer's** | Yes, No |
| **Sleep Quality** | Poor, Fair, Good, Excellent |
| **Dietary Habits** | Unhealthy, Moderate, Healthy |
| **Air Pollution Exposure** | Minimal, Slight, Moderate, High, Severe |
| **Employment Status** | Employed, Unemployed, Retired, Student |
| **Marital Status** | Single, Married, Divorced, Widowed |
| **Genetic Risk Factor (APOE-ε4 allele)** | Yes, No |
| **Social Engagement Level** | Low, Moderate, High |
| **Income Level** | Low, Middle, High |
| **Urban vs Rural Living** | Urban, Rural |

---

## How to Use the App

### 🚀 **Getting Started**
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run the App**: `streamlit run ynps.py`
3. **Open Browser**: Navigate to `http://localhost:8501`

### 📋 **Using the Risk Assessment**
1. **Fill Out the Form**: Complete all 24 health and lifestyle factors using the intuitive dropdowns and input fields
2. **Get Your Prediction**: Click "🧪 Predict Alzheimer's Risk" to receive your assessment
3. **View Results**: See your risk level (Low/Moderate/High) with confidence scores and professional guidance
4. **Explore Recommendations**: Get personalized lifestyle tips for risk reduction based on current research

### 🎯 **Key Features**
- **User-Friendly Interface**: No medical jargon - simple dropdowns and clear labels
- **Intelligent Risk Interpretation**: Converts model predictions to meaningful risk levels
- **Professional Guidance**: Appropriate medical disclaimers and consultation recommendations
- **Lifestyle Tips**: Evidence-based prevention strategies with randomized suggestions
- **Comprehensive Assessment**: Analyzes demographics, health conditions, cognitive factors, environmental exposure, and genetic risk

---

## Model Training (Summary)

The XGBoost model was trained on a comprehensive dataset using advanced machine learning techniques:

- **Data Preprocessing**: Advanced categorical encoding using target encoding and custom feature mappings
- **Model Architecture**: XGBoost classifier optimized for healthcare prediction tasks
- **Feature Engineering**: 24 carefully selected features covering all major Alzheimer's risk domains
- **Categorical Handling**: Smart encoding system that converts user-friendly inputs to model-compatible numerical values
- **Risk Interpretation**: Intelligent label mapping that converts model outputs to meaningful risk assessments

### 🔧 **Technical Implementation**
- **Frontend**: Streamlit with responsive two-column layout and professional medical UI
- **Backend**: XGBoost classifier with gzipped model serialization
- **Encoding**: Custom categorical feature encoding with comprehensive mappings
- **Error Handling**: Robust error management with user-friendly debugging information

---

## Repository Contents

| File | Description |
|------|-------------|
| `ynps.py` | Main Streamlit application with full risk assessment interface |
| `xgb_model.pkl.gz` | Trained XGBoost model (compressed) |
| `target_encoder.pkl.gz` | Target label encoder for prediction interpretation |
| `requirements.txt` | Python dependencies needed to run the app |
| `README.md` | This comprehensive documentation |

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/alzheimer-risk-prediction.git
cd alzheimer-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run ynps.py
```

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
```

---

## ⚠️ Important Medical Disclaimers

- **Educational Purpose Only**: This tool is designed for educational and informational purposes
- **Not Medical Advice**: Should never replace professional medical consultation, diagnosis, or treatment
- **Consult Healthcare Providers**: Always discuss health concerns and risk factors with qualified medical professionals
- **Research Tool**: Based on machine learning models and statistical associations from research data
- **Individual Variation**: Results may not reflect individual medical circumstances or family history nuances

---

## 🔬 Technical Features

- **Advanced ML Pipeline**: XGBoost classifier with optimized hyperparameters
- **Smart Categorical Encoding**: Converts user-friendly categories to numerical features automatically
- **Professional UI/UX**: Medical-grade interface with appropriate disclaimers and guidance
- **Comprehensive Risk Assessment**: 24-factor analysis covering all major Alzheimer's risk domains
- **Intelligent Result Interpretation**: Meaningful risk levels instead of raw model outputs
- **Evidence-Based Recommendations**: Lifestyle tips based on current Alzheimer's prevention research

---

## 📊 Model Performance & Validation

The model underwent rigorous validation to ensure reliable risk assessment:
- **Cross-Validation**: Comprehensive testing across different patient populations
- **Feature Importance**: All 24 features contribute meaningfully to risk prediction
- **Balanced Assessment**: Handles diverse demographic and health profiles
- **Clinical Relevance**: Risk factors align with established medical research

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🔗 Additional Resources

- [Alzheimer's Association](https://www.alz.org) - Comprehensive information and support
- [National Institute on Aging](https://www.nia.nih.gov) - Research and clinical guidelines
- [CDC Alzheimer's Resources](https://www.cdc.gov/aging/aginginfo/alzheimers.htm) - Public health information

---

## Contact

**Chenwei Pan**  
For questions, suggestions, or support, please reach out on GitHub or open an issue in this repository.

---

*This application represents a research and educational tool designed to increase awareness about Alzheimer's risk factors. It should be used in conjunction with, not as a replacement for, professional medical care and consultation.*
