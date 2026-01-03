import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import os

# === Load Data ===
df = pd.read_csv("Data/phishing_cleaned_final.csv")

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

X = df[features]
y = df["Type"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Scaling for XGB ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Models ===
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_scaled, y_train)

# === Save Models ===
os.makedirs("models", exist_ok=True)
joblib.dump(cat_model, "models/catboost_model.pkl")
joblib.dump(xgb_model, "models/xgboost_phishing_model.pkl")
joblib.dump(scaler, "models/xgb_scaler.pkl")

# === Define Hybrid Model with CatBoost Veto Logic ===
class HybridCatXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None):
        self.model1 = model1  # XGBoost
        self.scaler1 = scaler1
        self.model2 = model2  # CatBoost

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(X)[:, 1]
        return ((p1 + p2) / 2).reshape(-1, 1)

    def predict(self, X):
        cat_pred = self.model2.predict(X)
        xgb_pred = self.model1.predict(self.scaler1.transform(X))
        # If CatBoost says phishing (1), mark as phishing regardless
        final_pred = np.where(cat_pred == 1, 1, xgb_pred)
        return final_pred

# === Create Hybrid Model ===
hybrid_model = HybridCatXGB(model1=xgb_model, scaler1=scaler, model2=cat_model)
joblib.dump(hybrid_model, "models/hybrid_cat_xgb_model.pkl")

# === Evaluate ===
print("\n🔍 Evaluating Hybrid Voting Model")
y_pred = hybrid_model.predict(X_test)
y_proba = hybrid_model.predict_proba(X_test).flatten()

print("\n📊 Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_proba))

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Show where both models disagree ===
xgb_preds = xgb_model.predict(X_test_scaled)
cat_preds = cat_model.predict(X_test)
disagreements = (xgb_preds != cat_preds)
print(f"\n Disagreements between XGB & CatBoost: {np.sum(disagreements)} out of {len(y_test)} samples")
