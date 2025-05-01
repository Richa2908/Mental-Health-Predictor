import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('survey.csv')

# Clean column names (just in case)
df.columns = df.columns.str.strip().str.lower()

# Only select the necessary columns
df = df[['age', 'gender', 'family_history', 'treatment']]  # Drop other columns

# Drop rows with missing values
df = df.dropna()

# Normalize gender values
df['gender'] = df['gender'].str.lower().str.strip()
df['gender'] = df['gender'].replace({
    'm': 'male',
    'male': 'male',
    'f': 'female',
    'female': 'female',
    'cis male': 'male',
    'cis female': 'female',
    'trans female': 'other',
    'trans male': 'other',
    'non-binary': 'other',
    'genderfluid': 'other',
    'agender': 'other',
    'other': 'other',
    'something else': 'other'
})

# Normalize family_history
df['family_history'] = df['family_history'].str.lower().str.strip()
df['treatment'] = df['treatment'].str.lower().str.strip()

# Label Encoding
le_gender = LabelEncoder()
le_family = LabelEncoder()
le_treatment = LabelEncoder()

# Fit and transform
df['gender'] = le_gender.fit_transform(df['gender'])
df['family_history'] = le_family.fit_transform(df['family_history'])
df['treatment'] = le_treatment.fit_transform(df['treatment'])

# Prepare features (X) and target (y)
X = df[['age', 'gender', 'family_history']]  # Features
y = df['treatment']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_family, 'le_family.pkl')
joblib.dump(le_treatment, 'le_treatment.pkl')  # optional, in case you need to decode predictions

print("✅ Model and encoders saved successfully.")
