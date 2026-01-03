import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# === Define Hybrid Model Classes ===
class HybridCatXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None):
        self.model1 = model1  # XGBoost
        self.scaler1 = scaler1  # Scaler for XGBoost
        self.model2 = model2  # CatBoost

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(X)[:, 1]
        avg = (p1 + p2) / 2
        return np.vstack([1 - avg, avg]).T

    def predict(self, X):
        p1 = self.model1.predict(self.scaler1.transform(X))
        p2 = self.model2.predict(X)
        return np.where((p1 == 1) | (p2 == 1), 1, 0)

class HybridCatLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, model2=None, scaler2=None):
        self.model1 = model1  # CatBoost
        self.model2 = model2  # LightGBM
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(X)[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        avg = (p1 + p2) / 2
        return np.vstack([1 - avg, avg]).T

    def predict(self, X):
        p1 = self.model1.predict(X)
        p2 = self.model2.predict(self.scaler2.transform(X))
        return np.where((p1 == 1) | (p2 == 1), 1, 0)

class HybridXGBLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None, scaler2=None):
        self.model1 = model1  # XGBoost
        self.scaler1 = scaler1
        self.model2 = model2  # LightGBM
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        avg = (p1 + p2) / 2
        return np.vstack([1 - avg, avg]).T

    def predict(self, X):
        p1 = self.model1.predict(self.scaler1.transform(X))
        p2 = self.model2.predict(self.scaler2.transform(X))
        return np.where((p1 == 1) | (p2 == 1), 1, 0)

# === Load Data ===
df = pd.read_csv("Data/phishing_cleaned_final.csv")
features = df.columns[df.columns != "Type"]
X = df[features]
y = df["Type"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Load Models & Scalers ===
cat = joblib.load("models/catboost_phishing_model.pkl")
xgb = joblib.load("models/xgboost_phishing_model.pkl")
lgbm = joblib.load("models/lgbm_phishing_model.pkl")
rf = joblib.load("models/random_forest_model.pkl")

scaler_xgb = joblib.load("models/xgboost_scaler.pkl")
scaler_lgbm = joblib.load("models/lgbm_scaler.pkl")
scaler_rf = joblib.load("models/rf_scaler.pkl")

# === Hybrid Models ===
hybrid_cat_xgb = HybridCatXGB(model1=xgb, scaler1=scaler_xgb, model2=cat)
hybrid_cat_lgbm = HybridCatLGBM(model1=cat, model2=lgbm, scaler2=scaler_lgbm)
hybrid_xgb_lgbm = HybridXGBLGBM(model1=xgb, scaler1=scaler_xgb, model2=lgbm, scaler2=scaler_lgbm)

# === Models Dictionary ===
models = {
    "CatBoost": (cat, X_test),
    "XGBoost": (xgb, scaler_xgb.transform(X_test)),
    "Random Forest": (rf, scaler_rf.transform(X_test)),
    "LightGBM": (lgbm, scaler_lgbm.transform(X_test)),
    "Hybrid Cat + XGB": (hybrid_cat_xgb, X_test),
    "Hybrid Cat + LGBM": (hybrid_cat_lgbm, X_test),
    "Hybrid XGB + LGBM": (hybrid_xgb_lgbm, X_test)
}

results = []
plt.figure(figsize=(10, 7))

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    try:
        y_proba = model.predict_proba(X_eval)[:, 1]
    except:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.3f})")

    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Results Table ===
df_results = pd.DataFrame(results).set_index("Model")
print("\n=== Final Model Comparison Table ===")
print(df_results.applymap(lambda x: f"{x:.4f}"))

# === Plot ROC ===
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
plt.title("ROC Curve Comparison of All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
