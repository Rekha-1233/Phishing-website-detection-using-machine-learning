# train_hybrid_voting.py

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, ClassifierMixin


# 1. Load cleaned dataset
df = pd.read_csv("data/phishing_cleaned_final.csv")
X = df.drop("Type", axis=1).values
y = df["Type"].values

# 2. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Load or train base models
try:
    rf = joblib.load("models/random_forest_phishing_model.pkl")
    xgb = joblib.load("models/xgboost_phishing_model.pkl")
    lgbm = joblib.load("models/lgbm_phishing_model.pkl")
except FileNotFoundError:
    print("[INFO] Base model files not found, training base models...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)

    lgbm = LGBMClassifier(n_estimators=200, random_state=42)
    lgbm.fit(X_train_scaled, y_train)

    joblib.dump(rf, "models/random_forest_phishing_model.pkl")
    joblib.dump(xgb, "models/xgboost_phishing_model.pkl")
    joblib.dump(lgbm, "models/lgbm_phishing_model.pkl")
    print("[INFO] Base models trained and saved.")


# 5. Wrapper class for Keras CNN as sklearn classifier
class KerasCNNWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, model_path):
        self.model_path = model_path

    def fit(self, X=None, y=None):
        self.model_ = load_model(self.model_path)
        self.classes_ = np.array([0, 1])  # required by sklearn
        return self

    def predict(self, X):
        X_cnn = X.reshape(-1, X.shape[1], 1)
        probs = self.model_.predict(X_cnn, verbose=0)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        X_cnn = X.reshape(-1, X.shape[1], 1)
        probs = self.model_.predict(X_cnn, verbose=0)[:, 1]
        return np.vstack([1 - probs, probs]).T


# Instantiate and load Keras model
cnn = KerasCNNWrapper("models/final_cnn_model.h5").fit()

# 6. Build soft-voting ensemble
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm), ("cnn", cnn)],
    voting="soft",
    n_jobs=-1
)

# 7. Train ensemble on scaled training data
ensemble.fit(X_train_scaled, y_train)

# 8. Evaluate ensemble on test data
acc = ensemble.score(X_test_scaled, y_test)
print(f"✅ Hybrid Voting Ensemble Test Accuracy: {acc:.4f}")

# 9. Save scaler and ensemble for use in Flask app
joblib.dump(scaler, "processed/scaler.pkl")
joblib.dump(ensemble, "models/hybrid_voting_model.pkl")
print("✅ Saved scaler and hybrid voting model.")
