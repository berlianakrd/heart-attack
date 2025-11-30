import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

print("Loading data...")
df = pd.read_csv('data/heart_attack_data.csv')

# Separate features and target
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Handle categorical variables
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining models...")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"{name}: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")

# Save everything
print("\nSaving model files...")

joblib.dump(best_model, 'models/best_model.pkl')
print("✓ Saved: models/best_model.pkl")

joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✓ Saved: models/label_encoders.pkl")

metadata = {
    'model_name': best_name,
    'accuracy': float(best_score),
    'features': list(X.columns),
    'n_samples': len(df)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: models/model_metadata.json")

print("\n✅ All files saved successfully!")