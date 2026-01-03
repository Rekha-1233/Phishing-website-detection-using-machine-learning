import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Load Dataset
df = pd.read_csv("Data/phishing_cleaned_final.csv")  # Adjust path if needed

# Step 2: Separate features and label
X = df.drop('Type', axis=1)
y = df['Type'].astype(int)

# Step 3: Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,         # Number of trees
    random_state=42,
    n_jobs=-1,                # Use all CPU cores
    verbose=1                 # Show training logs
)
rf_model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred = rf_model.predict(X_test_scaled)

# Step 7: Evaluation
print("\n Random Forest Result :\n", classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save Model and Scaler
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/rf_scaler.pkl")
print(" Random Forest model and scaler saved successfully.")
