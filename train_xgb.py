import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib

# Step 1: Load Dataset
df = pd.read_csv("Data/phishing_cleaned_final.csv")  # Adjust path as needed

# Step 2: Separate features and label
X = df.drop('Type', axis=1)
y = df['Type']

# Optional: Ensure labels are integers and verify label distribution
y = y.astype(int)
print("Label distribution:\n", y.value_counts())

# Step 3: Split dataset with stratification to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# Step 4: Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train XGBoost classifier
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbosity=1              # Show training logs
)
xgb_model.fit(X_train_scaled, y_train)

# Step 6: Predictions on the test set
y_pred = xgb_model.predict(X_test_scaled)

# Step 7: Model evaluation
print("" \
" Accuracy:", accuracy_score(y_test, y_pred))
print("\n XGBoost Result :\n", classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Save the trained model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgboost_phishing_model.pkl")
joblib.dump(scaler, "models/xgboost_scaler.pkl")
print(" Model and scaler saved successfully.")
