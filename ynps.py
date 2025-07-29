import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import xgboost as xgb

# Change this to your CSV file name
filename = r"encoded_alzheimers_prediction_dataset (4).csv"


# Load data
df = pd.read_csv(filename)
df = df.dropna()

# Label encode categorical columns (except target)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Alzheimer’s Diagnosis':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df['Alzheimer’s Diagnosis'] = target_encoder.fit_transform(df['Alzheimer’s Diagnosis'])

# Split features and target
X = df.drop(columns=['Alzheimer’s Diagnosis'])
y = df['Alzheimer’s Diagnosis']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset with SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Save model and transformers
joblib.dump(model, 'xgb_model.pkl', compress=('gzip', 3))
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

model_size = os.path.getsize('xgb_model.pkl') / (1024 * 1024)
print(f"Model saved as xgb_model.pkl ({model_size:.2f} MB)")
