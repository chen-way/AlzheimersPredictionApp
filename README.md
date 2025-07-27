# Alzheimer’s Disease Prediction App 🧠

**By Chenwei Pan**  
Westmount Charter Mid-High School

---

## Overview

Alzheimer’s disease (AD) is a progressive neurological disorder characterized by memory loss and cognitive decline. Early detection is essential for timely treatment and better patient outcomes, yet traditional diagnostic methods often identify the disease only after significant progression.

This project develops a machine learning-based prediction model using patient health data to estimate the risk of Alzheimer’s. The model is integrated into an interactive web app where users can upload patient datasets and receive predictions instantly.

---

## Objective

- Build a reliable Alzheimer’s prediction model using Random Forest with hyperparameter tuning.
- Provide an easy-to-use Streamlit web app for uploading patient data and getting diagnosis predictions.
- Support data preprocessing steps including encoding categorical features and feature scaling.
- Handle imbalanced data through SMOTEENN resampling to improve model performance.

---

## Features Used for Prediction

| Feature                         |
|--------------------------------|
| Country                        |
| Age                            |
| Gender                         |
| Education Level                |
| BMI                            |
| Physical Activity Level         |
| Smoking Status                 |
| Alcohol Consumption            |
| Diabetes                       |
| Hypertension                   |
| Cholesterol Level              |
| Family History of Alzheimer’s   |
| Cognitive Test Score            |
| Depression Level               |
| Sleep Quality                  |
| Dietary Habits                 |
| Air Pollution Exposure         |
| Employment Status              |
| Marital Status                 |
| Genetic Risk Factor (APOE-ε4 allele) |
| Social Engagement Level        |
| Income Level                   |
| Stress Levels                  |
| Urban vs Rural Living          |

---

## How to Use the App

1.
---

## Model Training (Summary)

The Random Forest model was trained on a cleaned and encoded dataset using these steps:

- **Data preprocessing:** Encoding categorical variables using `LabelEncoder` and scaling numeric features with `StandardScaler`.
- **Handling imbalanced classes:** Using SMOTEENN resampling.
- **Hyperparameter tuning:** GridSearchCV over parameters like number of trees, max depth, and min samples per leaf.
- **Evaluation:** Model achieved approximately 92% accuracy on a held-out test set.

---

## Repository Contents

| File                 | Description                                            |
|----------------------|--------------------------------------------------------|
| `alzheimers_app.py`    | Main Streamlit app for prediction                      |
| `best_model.pkl`       | Trained Random Forest model                             |
| `scaler.pkl`           | StandardScaler object for feature scaling              |
| `label_encoders.pkl`   | Dictionary of LabelEncoders for categorical features   |
| `target_encoder.pkl`   | LabelEncoder for target variable (Alzheimer’s Diagnosis)|
| `requirements.txt`     | Python dependencies needed to run the app              |
| `README.md`            | This documentation file                                |

---

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `streamlit`
- `joblib`

---

## Notes

- Ensure the CSV uploaded to the app matches the feature columns and format used during training.
- The app automatically handles encoding and scaling based on saved transformers.
- For best results, use datasets similar in distribution to the training data.

---

## Contact

For questions or support, please reach out on GitHub


