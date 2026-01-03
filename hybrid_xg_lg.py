import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Data/phishing_cleaned_final.csv")

# Define features
features = [
    "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url", "number_of_digits_in_url",
    "number_of_special_char_in_url", "number_of_hyphens_in_url", "number_of_slash_in_url",
    "number_of_questionmark_in_url", "number_of_equal_in_url", "number_of_at_in_url",
    "number_of_exclamation_in_url", "number_of_percent_in_url", "domain_length", "number_of_dots_in_domain",
    "number_of_hyphens_in_domain", "having_special_characters_in_domain", "number_of_special_characters_in_domain",
    "having_digits_in_domain", "having_repeated_digits_in_domain", "number_of_subdomains",
    "having_dot_in_subdomain", "having_hyphen_in_subdomain", "average_subdomain_length",
    "average_number_of_hyphens_in_subdomain", "having_special_characters_in_subdomain",
    "number_of_special_characters_in_subdomain", "having_digits_in_subdomain", "number_of_digits_in_subdomain",
    "having_path", "path_length", "having_query", "having_anchor", "entropy_of_url", "entropy_of_domain"
]

X = df[features].values
y = df["Type"].astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Load models and scalers
model_xgb = joblib.load("models/xgboost_phishing_model.pkl")
scaler_xgb = joblib.load("models/xgboost_scaler.pkl")

model_lgbm = joblib.load("models/lgbm_phishing_model.pkl")
scaler_lgbm = joblib.load("models/lgbm_scaler.pkl")

# Soft voting prediction
xgb_probs = model_xgb.predict_proba(scaler_xgb.transform(X_test))[:, 1]
lgbm_probs = model_lgbm.predict_proba(scaler_lgbm.transform(X_test))[:, 1]

# Average the probabilities
avg_probs = (xgb_probs + lgbm_probs) / 2

# Final binary prediction
y_pred = (avg_probs >= 0.5).astype(int)

# Evaluate model
print(" Hybrid Model (Soft Voting: XGBoost + LightGBM):")
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, avg_probs))
print("\nHybrid XGBoost+LightGBM Result:\n", classification_report(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save hybrid model class
class HybridModel:
    def __init__(self, model1, scaler1, model2, scaler2):
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2
        self.scaler2 = scaler2


    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        return (p1 + p2) / 2

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return (avg_proba >= 0.5).astype(int)

# Save for deployment
hybrid_model = HybridModel(model_xgb, scaler_xgb, model_lgbm, scaler_lgbm)
joblib.dump(hybrid_model, "models/hybrid_model.pkl")
print(" Hybrid model saved as models/hybrid_model.pkl")
