# full_pipeline.py
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ─────────────────────────────────────────────
# 1. Load & Clean Data
# ─────────────────────────────────────────────
df = pd.read_csv("Data/PhiUSIIL_Phishing_URL_Dataset.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

cols_to_drop = ["FILENAME","URL","Domain","TLD","Title"]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

if "label" not in df.columns:
    raise ValueError("❌ 'label' column missing!")
y = df["label"].astype(int)
X = df.drop(columns=["label"])

# Keep only numeric features
num_cols = X.select_dtypes(include="number").columns.tolist()
X = X[num_cols]

# ─────────────────────────────────────────────
# 2. Train/Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ─────────────────────────────────────────────
# 3. Preprocessing Pipeline
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(
    [("scale", StandardScaler(), num_cols)],
    remainder="drop"
)

# ─────────────────────────────────────────────
# 4. Define Models & Pipelines
# ─────────────────────────────────────────────
def make_pipeline(clf):
    return Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])

models = {
    "LogisticRegression": make_pipeline(
        LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced")
    ),
    "RandomForest": make_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    ),
    "GradientBoosting": make_pipeline(
        GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    ),
    "SVM-RBF": make_pipeline(
        SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)
    )
}

# ─────────────────────────────────────────────
# 5. Cross-Validation Evaluation
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True)
}

results = []
for name, pipe in models.items():
    cv_res = cross_validate(pipe, X_train, y_train,
                            cv=cv, scoring=scoring,
                            return_train_score=False, n_jobs=-1)
    acc_mean = cv_res["test_accuracy"].mean()
    auc_mean = cv_res["test_roc_auc"].mean()
    results.append((name, acc_mean, auc_mean))
    print(f"{name:18s} | CV Accuracy: {acc_mean:.4f} | CV ROC-AUC: {auc_mean:.4f}")

# ─────────────────────────────────────────────
# 6. Final Train & Test Evaluation
# ─────────────────────────────────────────────
print("\nFinal test-set evaluation:")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probas = pipe.predict_proba(X_test)[:,1]
    acc_test = (preds == y_test).mean()
    auc_test = roc_auc_score(y_test, probas)
    print(f"{name:18s} | Test Accuracy: {acc_test:.4f} | Test ROC-AUC: {auc_test:.4f}")

# ─────────────────────────────────────────────
# 7. Save CV Summary
# ─────────────────────────────────────────────
summary_df = pd.DataFrame(results, columns=["Model","CV_Accuracy","CV_ROC_AUC"])
os.makedirs("processed", exist_ok=True)
summary_df.to_csv(Path("processed")/"model_cv_summary.csv", index=False)
print("\n✅ All done. CV summary saved to processed/model_cv_summary.csv")
