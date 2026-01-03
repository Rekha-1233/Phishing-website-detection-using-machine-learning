import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report
)
from catboost import CatBoostClassifier

# ===========================
# Define HybridCatXGB Class
# ===========================
class HybridCatXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None):
        self.model1 = model1  # XGBoost
        self.scaler1 = scaler1  # Scaler for XGBoost
        self.model2 = model2  # CatBoost

    def fit(self, X, y):
        X_scaled = self.scaler1.transform(X)
        self.model1.fit(X_scaled, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        xgb_pred = self.model1.predict(self.scaler1.transform(X))
        cat_pred = self.model2.predict(X)
        
        final_pred = []
        for cp, xp in zip(cat_pred, xgb_pred):
            if cp == 1 or xp == 1:
                final_pred.append(1)
            else:
                final_pred.append(0)
        return final_pred

    def predict_proba(self, X):
        xgb_proba = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        cat_proba = self.model2.predict_proba(X)[:, 1]
        avg_proba = (xgb_proba + cat_proba) / 2
        return np.vstack([1 - avg_proba, avg_proba]).T


# ===========================
# Load and Prepare Dataset
# ===========================
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

X = df[features].values
y = df["Type"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===========================
# Load Base Models and Scalers
# ===========================
rf = joblib.load("models/random_forest_phishing_model.pkl")
xgb = joblib.load("models/xgboost_phishing_model.pkl")
lgbm = joblib.load("models/lgbm_phishing_model.pkl")

cat = CatBoostClassifier()
cat.load_model("models/catboost_phishing_model.cbm")

scaler_rf = joblib.load("models/rf_scaler.pkl")
scaler_xgb = joblib.load("models/xgboost_scaler.pkl")
scaler_lgbm = joblib.load("models/lgbm_scaler.pkl")
scaler_cat = joblib.load("models/catboost_scaler.pkl")

# ===========================
# Load Hybrid Models
# ===========================
hybrid_cat_xgb = joblib.load("models/hybrid_cat_xgb_model.pkl")

hybrid_cat_lgbm_model = joblib.load("models/hybrid_cat_lgbm_model.pkl")

# ===========================
# Define Models Dictionary
# ===========================
models = {
    "Random Forest": (rf, scaler_rf),
    "XGBoost": (xgb, scaler_xgb),
    "LightGBM": (lgbm, scaler_lgbm),
    "CatBoost": (cat, scaler_cat),
    "Hybrid Cat+XGB": (hybrid_cat_xgb, None),
    "Hybrid Cat+LGBM": (hybrid_cat_lgbm, None),
    "Hybrid Cat+LGBM Model": (hybrid_cat_lgbm_model, None)
}

# ===========================
# Evaluate and Compare Models
# ===========================
results = []
plt.figure(figsize=(10, 7))

for name, (model, scaler) in models.items():
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    try:
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    except Exception as e:
        print(f"[ERROR] Model '{name}' failed: {e}")
        continue

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC AUC": roc_auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

# ===========================
# Final Results Table & ROC Plot
# ===========================
df_results = pd.DataFrame(results).set_index("Model")
print("\n=== Model Comparison Table ===")
print(df_results.applymap(lambda x: f"{x:.4f}"))

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
plt.title("ROC Curve Comparison of All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
