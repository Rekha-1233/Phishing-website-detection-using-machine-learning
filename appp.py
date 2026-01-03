from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from urllib.parse import urlparse

# ===== Trusted Domains =====
TRUSTED_DOMAINS = [
    "mail.google.com", "www.google.com", "web.whatsapp.com",
    "accounts.google.com", "www.wikipedia.org", "www.facebook.com",
    "www.instagram.com", "www.youtube.com", "twitter.com", "linkedin.com",
    "github.com", "stackoverflow.com", "microsoft.com", "login.microsoftonline.com"
]

def is_trusted_url(url):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return any(trusted in domain for trusted in TRUSTED_DOMAINS)
    except:
        return False

# ===== Hybrid Classes =====
class HybridCatXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None):
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(X)[:, 1]
        return ((p1 + p2) / 2).reshape(-1, 1)

    def predict(self, X):
        xgb_preds = self.model1.predict(self.scaler1.transform(X))
        cat_preds = self.model2.predict(X)
        return np.where((cat_preds == 1) | (xgb_preds == 1), 1, 0)

class HybridCatLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, model2=None, scaler2=None):
        self.model1 = model1
        self.model2 = model2
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(X)[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        return ((p1 + p2) / 2).reshape(-1, 1)

    def predict(self, X):
        cat_preds = self.model1.predict(X)
        lgbm_preds = self.model2.predict(self.scaler2.transform(X))
        return np.where((cat_preds == 1) | (lgbm_preds == 1), 1, 0)

class HybridXGBLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None, scaler2=None):
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        return ((p1 + p2) / 2).reshape(-1, 1)

    def predict(self, X):
        xgb_preds = self.model1.predict(self.scaler1.transform(X))
        lgbm_preds = self.model2.predict(self.scaler2.transform(X))
        return np.where((xgb_preds == 1) | (lgbm_preds == 1), 1, 0)

class ConsensusPhishing(BaseEstimator, ClassifierMixin):
    def __init__(self, *models):
        self.models = models

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.where(np.vstack(preds).sum(axis=0) >= 1, 1, 0)

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        return sum(probas) / len(probas)

# ===== Flask App Setup =====
app = Flask(__name__)

# ===== Load Models =====
rf_model = joblib.load("models/random_forest_model.pkl")
scaler_rf = joblib.load("models/rf_scaler.pkl")

xgb_model = joblib.load("models/xgboost_phishing_model.pkl")
scaler_xgb = joblib.load("models/xgb_scaler.pkl")

lgbm_model = joblib.load("models/lgbm_phishing_model.pkl")
scaler_lgbm = joblib.load("models/lgbm_scaler.pkl")

cat_model = joblib.load("models/catboost_model.pkl")
scaler_cat = joblib.load("models/catboost_scaler.pkl")

# ===== Hybrid Instances =====
hybrid_cat_xgb = HybridCatXGB(model1=xgb_model, scaler1=scaler_xgb, model2=cat_model)
hybrid_cat_lgbm = HybridCatLGBM(model1=cat_model, model2=lgbm_model, scaler2=scaler_lgbm)
hybrid_xgb_lgbm = HybridXGBLGBM(model1=xgb_model, scaler1=scaler_xgb, model2=lgbm_model, scaler2=scaler_lgbm)
consensus_model = ConsensusPhishing(hybrid_cat_xgb, hybrid_xgb_lgbm)

# ===== Default Features =====
DEFAULT_22 = [0,0,0,2,0,0,3,0,1,3,0,0,0,3,0,0,4.0104,2.7516,0,0,0,0]
columns = [
    "url_length","number_of_dots_in_url","having_repeated_digits_in_url","number_of_digits_in_url",
    "number_of_special_char_in_url","number_of_hyphens_in_url","number_of_slash_in_url",
    "number_of_questionmark_in_url","number_of_equal_in_url","number_of_at_in_url",
    "number_of_exclamation_in_url","number_of_percent_in_url","domain_length","number_of_dots_in_domain",
    "number_of_hyphens_in_domain","having_special_characters_in_domain","number_of_special_characters_in_domain",
    "having_digits_in_domain","having_repeated_digits_in_domain","number_of_subdomains",
    "having_dot_in_subdomain","having_hyphen_in_subdomain","average_subdomain_length",
    "average_number_of_hyphens_in_subdomain","having_special_characters_in_subdomain",
    "number_of_special_characters_in_subdomain","having_digits_in_subdomain","number_of_digits_in_subdomain",
    "having_path","path_length","having_query","having_anchor","entropy_of_url","entropy_of_domain"
]

# ===== Utility Functions =====
def normalize_url(raw_url):
    return re.sub(r'^(https?://)?(www\.)?', '', raw_url.strip().lower())

def extract_12_features(url):
    url = normalize_url(url)
    return [
        len(url), url.count('.'), int(bool(re.search(r'(\d)\1{1,}', url))),
        sum(c.isdigit() for c in url), len(re.findall(r'[^\w\s]', url)),
        url.count('-'), url.count('/'), url.count('?'), url.count('='),
        url.count('@'), url.count('!'), url.count('%')
    ]

def get_prediction(model, scaler, df, label):
    try:
        input_data = scaler.transform(df) if scaler else df
        pred = model.predict(input_data)[0]
        proba_all = model.predict_proba(input_data)[0]
        proba = proba_all[1] if len(proba_all) > 1 else proba_all[0]
        status = "🟢 Legitimate" if pred == 0 else "🔴 Phishing"
        return f"{status} ({label}, {proba*100:.2f}%)"
    except Exception as e:
        return f"❌ {label} Error: {e}"

# ===== Main Route =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    url_input = ""
    error = None

    if request.method == "POST":
        try:
            method = request.form.get("input_method")
            if method == "url":
                url_input = request.form.get("url", "").strip()
                if not url_input:
                    return render_template("indexx.html", error="Please enter a valid URL.")
                if is_trusted_url(url_input):
                    trusted = "🟢 Trusted (Whitelisted Domain)"
                    for key in ["rf_result","xgb_result","lgbm_result","cat_result","hybrid1_result","hybrid2_result","hybrid3_result","consensus_result"]:
                        result[key] = trusted
                    return render_template("indexx.html", **result, url=url_input)
                feats_12 = extract_12_features(url_input)
            elif method == "manual":
                feats_12 = [float(request.form.get(f"f{i}")) for i in range(1, 13)]
            else:
                return render_template("indexx.html", error="Invalid input method.")

            full_features = feats_12 + DEFAULT_22
            df_in = pd.DataFrame([full_features], columns=columns)

            result["rf_result"] = get_prediction(rf_model, scaler_rf, df_in, "Random Forest")
            result["xgb_result"] = get_prediction(xgb_model, scaler_xgb, df_in, "XGBoost")
            result["lgbm_result"] = get_prediction(lgbm_model, scaler_lgbm, df_in, "LightGBM")
            result["cat_result"] = get_prediction(cat_model, scaler_cat, df_in, "CatBoost")
            result["hybrid1_result"] = get_prediction(hybrid_xgb_lgbm, None, df_in, "Hybrid XGB + LGBM")
            result["hybrid2_result"] = get_prediction(hybrid_cat_xgb, None, df_in, "Hybrid CatBoost + XGBoost")
            result["hybrid3_result"] = get_prediction(hybrid_cat_lgbm, None, df_in, "Hybrid CatBoost + LightGBM")
            result["consensus_result"] = get_prediction(consensus_model, None, df_in, "Consensus Hybrid")

        except Exception as e:
            error = f"❌ Error: {e}"

    return render_template("indexx.html", **result, url=url_input, error=error)

# ===== Run App =====
if __name__ == "__main__":
    app.run(debug=True)
