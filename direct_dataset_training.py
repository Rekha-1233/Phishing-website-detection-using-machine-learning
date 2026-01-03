# scripts/direct_dataset_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load the dataset
df = pd.read_csv("Data/phishing_cleaned_final.csv")

# Separate features and target
X = df.drop("Type", axis=1)
y = df["Type"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phishing_rf_model.pkl")
print("✅ Model saved as: models/phishing_rf_model.pkl")
