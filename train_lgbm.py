import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Data/phishing_cleaned_final.csv")
X = df.drop("Type", axis=1).values
y = df["Type"].astype(int).values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


lgbm = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
lgbm.fit(X_train_s, y_train)
y_pred = lgbm.predict(X_test_s)


print(f"LGBM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


os.makedirs("models", exist_ok=True)
joblib.dump(lgbm, "models/lgbm_phishing_model.pkl")
joblib.dump(scaler, "models/lgbm_scaler.pkl")