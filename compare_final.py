import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
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

# Train/test split to get your test set for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load models and scalers
rf = joblib.load("models/random_forest_model.pkl")
xgb = joblib.load("models/xgboost_phishing_model.pkl")
lgbm = joblib.load("models/lgbm_phishing_model.pkl")

cat = CatBoostClassifier()
cat.load_model("models/catboost_phishing_model.cbm")

scaler_rf = joblib.load("models/rf_scaler.pkl")
scaler_xgb = joblib.load("models/xgboost_scaler.pkl")
scaler_lgbm = joblib.load("models/lgbm_scaler.pkl")
scaler_cat = joblib.load("models/catboost_scaler.pkl")

models = {
    "Random Forest": (rf, scaler_rf),
    "XGBoost": (xgb, scaler_xgb),
    "LightGBM": (lgbm, scaler_lgbm),
    "CatBoost": (cat, scaler_cat)
}

# Store results for table and ROC plotting
results = []
plt.figure(figsize=(10, 7))

for name, (model, scaler) in models.items():
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions & probabilities
    y_pred = model.predict(X_test_scaled)
    if name == "CatBoost":
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
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
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    
    # Print classification report
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

# Print comparison table
df_results = pd.DataFrame(results)
df_results = df_results.set_index("Model")
print("\n=== Model Comparison Table ===")
print(df_results.applymap(lambda x: f"{x:.4f}"))

# Plot ROC curve
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.7)
plt.title("ROC Curve Comparison of Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
