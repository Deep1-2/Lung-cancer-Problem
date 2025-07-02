# Lung-cancer-Problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid")
file_path = r"C:\lungs\lung_cancer\Lung Cancer\dataset_med.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded!")
    display(df.head())
except Exception as e:
    print("Failed to load:", e)
if 'diagnosis_date' in df.columns:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')

if 'end_treatment_date' in df.columns:
    df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')

# Create duration column
if {'diagnosis_date', 'end_treatment_date'}.issubset(df.columns):
    df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

# Drop columns not needed
df.drop(columns=[col for col in ['id', 'diagnosis_date', 'end_treatment_date', 'country'] if col in df.columns], inplace=True)

# Impute numeric columns
numerical_cols = ['age', 'bmi', 'cholesterol_level']
if 'treatment_duration' in df.columns:
    numerical_cols.append('treatment_duration')

df[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(df[numerical_cols])
plt.figure(figsize=(16, 8))
sns.countplot(x='survived', data=df)
plt.title("Survival Distribution")
plt.show()
categorical_cols = ['gender', 'cancer_stage', 'family_history', 'smoking_status',
                    'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']
categorical_cols = [col for col in categorical_cols if col in df.columns]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df['survived'] = target_encoder.fit_transform(df['survived'].astype(str))
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Trained!")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
joblib.dump(model, "lung_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("Model and preprocessors saved.")
def load_model():
    model = joblib.load("lung_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_order = joblib.load("feature_order.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    return model, scaler, label_encoders, feature_order, target_encoder


def preprocess_input(data_dict, scaler, label_encoders, num_cols, cat_cols, feature_order):
    df_input = pd.DataFrame([data_dict])

    for col in cat_cols:
        le = label_encoders[col]
        val = df_input[col].iloc[0]
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, val)
        df_input[col] = le.transform([val])

    df_input[num_cols] = scaler.transform(df_input[num_cols])
    df_input = df_input.reindex(columns=feature_order, fill_value=0)
    return df_input


def predict_survival(patient_dict):
    model, scaler, label_encoders, feature_order, target_encoder = load_model()

    num_cols = ['age', 'bmi', 'cholesterol_level']
    if 'treatment_duration' in patient_dict:
        num_cols.append('treatment_duration')

    cat_cols = list(label_encoders.keys())

    processed_input = preprocess_input(patient_dict, scaler, label_encoders, num_cols, cat_cols, feature_order)

    pred = model.predict(processed_input)[0]
    label = target_encoder.inverse_transform([pred])[0]
    return label
    new_patient = {
    'age': 60,
    'bmi': 26.5,
    'cholesterol_level': 200,
    'treatment_duration': 120,
    'gender': 'Female',
    'cancer_stage': 'Stage 3',
    'family_history': 'Yes',
    'smoking_status': 'Former smoker',
    'hypertension': 'Yes',
    'asthma': 'No',
    'cirrhosis': 'No',
    'other_cancer': 'No',
    'treatment_type': 'Radiation'
}

result = predict_survival(new_patient)
print("🩺 Predicted Outcome for the Patient:", result)
