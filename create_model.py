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

print("="*60)
print("INDONESIA HEART ATTACK PREDICTION - MODEL TRAINING")
print("="*60)

print("\nStep 1: Loading data...")
df = pd.read_csv('data/heart_attack_data.csv')
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nStep 2: Data cleaning...")
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

print(f"✓ After cleaning: {df.shape[0]} rows")

print("\nStep 3: Preparing features...")
# Check if target column exists
if 'heart_attack' not in df.columns:
    print("ERROR: 'heart_attack' column not found!")
    print("Available columns:", df.columns.tolist())
    exit()

# Separate features and target
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")

print("\nStep 4: Encoding categorical variables...")
# Handle categorical variables
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded: {col}")

print("\nStep 5: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

print("\nStep 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

print("\nStep 7: Training models...")
print("-"*60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"  → Accuracy: {score:.4f} ({score*100:.2f}%)")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print("-"*60)
print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")

print("\nStep 8: Saving model files...")
print("-"*60)

# Save model
joblib.dump(best_model, 'models/best_model.pkl')
print("✓ Saved: models/best_model.pkl")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✓ Saved: models/label_encoders.pkl")

# Save metadata
metadata = {
    'model_name': best_name,
    'accuracy': float(best_score),
    'features': list(X.columns),
    'categorical_features': categorical_cols,
    'n_samples': len(df),
    'n_train': len(X_train),
    'n_test': len(X_test)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: models/model_metadata.json")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  - models/best_model.pkl")
print("  - models/scaler.pkl")
print("  - models/label_encoders.pkl")
print("  - models/model_metadata.json")
print("\nYou can now run: python app.py")
print("="*60)