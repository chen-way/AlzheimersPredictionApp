# 🧠 Alzheimer's Risk Assessment Tool

An AI-powered web application that predicts Alzheimer's disease risk using advanced Random Forest machine learning algorithms.

## ✨ Features

- **🎯 Accurate Predictions**: Random Forest model with no overfitting
- **📊 Interactive Dashboard**: Beautiful Plotly visualizations
- **🔍 Feature Analysis**: See which factors influence your risk most
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Real-time Results**: Instant risk assessment
- **🎨 Modern UI**: Clean, professional interface

## 🚀 Live Demo

### 🌟 [**LIVE APP: https://alzheimersprediction.streamlit.app/**](https://alzheimersprediction.streamlit.app/) 🌟

✨ **No installation needed!** ✨ Just click the link above to use the app instantly!

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 24 comprehensive health and lifestyle factors
- **Preprocessing**: 
  - SMOTEENN for dataset balancing
  - StandardScaler for feature normalization
  - LabelEncoder for categorical variables
- **Validation**: Proper train/validation/test split
- **Performance**: High accuracy with no overfitting

### Key Improvements Over Previous Version
- ✅ **No Overfitting**: Properly validated Random Forest model
- 🎯 **Better Accuracy**: Enhanced prediction performance
- 🔄 **Balanced Dataset**: SMOTEENN handles class imbalance
- 📊 **Feature Scaling**: Normalized inputs for better performance
- 🌲 **Ensemble Learning**: Multiple decision trees for robust predictions

## 📋 Requirements

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- plotly
- joblib
- imbalanced-learn

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/alzheimers-risk-predictor.git
   cd alzheimers-risk-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run alzheimers.py
   ```

## 📁 Project Structure

```
alzheimers-risk-predictor/
├── alzheimers.py                           # Main Streamlit application
├── model.pkl                               # Trained Random Forest model
├── scaler.pkl                              # StandardScaler for preprocessing
├── encoder.pkl                             # Label encoders for categorical variables
├── encoded_alzheimers_prediction_dataset (4).csv  # Training dataset
├── requirements.txt                        # Python dependencies
└── README.md                              # Project documentation
```

## 🔍 Input Features

The model analyzes 24 different factors:

### 👤 Demographics
- Age, Gender, Country, Education Level

### 🏥 Health Metrics
- BMI, Cognitive Test Score, Depression Level, Stress Level
- Diabetes, Hypertension, Cholesterol Level, Family History

### 🏃‍♂️ Lifestyle Factors
- Physical Activity, Smoking Status, Alcohol Consumption
- Sleep Quality, Dietary Habits, Social Engagement

### 🌍 Environmental
- Air Pollution Exposure, Living Area (Urban/Rural)

### 💼 Social Demographics
- Employment Status, Marital Status, Income Level
- Genetic Risk Factor (APOE-ε4)

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Cross-Validation**: 3-fold CV with GridSearchCV
- **Scoring**: F1-weighted for balanced evaluation
- **Preprocessing**: Scaled features, balanced classes
- **Overfitting**: ✅ None detected (proper validation)

## ⚠️ Important Disclaimer

This tool is designed for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Alzheimer's Disease Risk Factors Dataset
- ML Framework: scikit-learn
- Web Framework: Streamlit
- Visualization: Plotly
- Data Processing: pandas, numpy

## 📈 Future Enhancements

- [ ] Add more visualization options
- [ ] Implement confidence intervals
- [ ] Add model explanation (SHAP values)
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Export PDF reports

---

## 🎯 **QUICK START - NO SETUP REQUIRED!**

### 👆 **Just click here:** [**https://alzheimersprediction.streamlit.app/**](https://alzheimersprediction.streamlit.app/)

**That's it!** The app is ready to use - no downloads, no installation, no hassle! 🚀

---

**Made with ❤️ and 🧠 for Alzheimer's awareness and prevention**
