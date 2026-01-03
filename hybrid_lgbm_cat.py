import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# === Load dataset ===
df = pd.read_csv("Data/phishing_cleaned_final.csv")

# === Define features and label ===
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

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Scale for LGBM ===
scaler_lgbm = StandardScaler()
X_train_lgbm = scaler_lgbm.fit_transform(X_train)
X_test_lgbm = scaler_lgbm.transform(X_test)

# === Train models ===
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_lgbm, y_train)

# === Define HybridCatLGBM class ===
class HybridCatLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, model2=None, scaler2=None):
        self.model1 = model1  # CatBoost
        self.model2 = model2  # LGBM
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(X)[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        return ((p1 + p2) / 2).reshape(-1, 1)

    def predict(self, X):
        return (self.predict_proba(X).flatten() >= 0.5).astype(int)

# === Save models ===
joblib.dump(cat_model, "models/catboost_model.pkl")
joblib.dump(lgbm_model, "models/lgbm_model.pkl")
joblib.dump(scaler_lgbm, "models/lgbm_scaler.pkl")

hybrid_cat_lgbm = HybridCatLGBM(model1=cat_model, model2=lgbm_model, scaler2=scaler_lgbm)
joblib.dump(hybrid_cat_lgbm, "models/hybrid_cat_lgbm.pkl")

print("✅ Hybrid CatBoost + LightGBM model trained and saved.")
