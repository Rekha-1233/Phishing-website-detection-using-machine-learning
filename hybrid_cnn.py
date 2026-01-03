import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load data
df = pd.read_csv("Data/data_phish.csv")
df = df[['url', 'status']].dropna()
df = df[df['status'].isin(['phishing', 'legitimate'])]
df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# Feature engineering (extend with more features as needed)
def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        int('https' in url),
        int('@' in url),
        url.count('/'),
        url.count('='),
        url.count('&'),
        url.count('%'),
        url.count('?')
    ]

df['features'] = df['url'].apply(extract_features)
X_handcrafted = np.array(df['features'].tolist())

# Load tokenizer and prepare CNN input
with open("models/cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
sequences = tokenizer.texts_to_sequences(df['url'])
X_cnn_input = pad_sequences(sequences, maxlen=200)

# Load CNN model and extract features from intermediate layer
cnn_model = load_model("models/cnn_url_model.keras")
from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
cnn_features = intermediate_layer_model.predict(X_cnn_input, verbose=1)

# Normalize features before combining
scaler_handcrafted = StandardScaler()
X_handcrafted_scaled = scaler_handcrafted.fit_transform(X_handcrafted)

scaler_cnn = StandardScaler()
cnn_features_scaled = scaler_cnn.fit_transform(cnn_features)

# Combine features
X_combined = np.hstack([X_handcrafted_scaled, cnn_features_scaled])
y = df['label'].values

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f"✅ Best RF Params: {grid_search.best_params_}")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ Classification Report:\n{classification_report(y_test, y_pred)}")

# Save model and scalers
import joblib
joblib.dump(best_rf, "models/hybrid_rf_model.pkl")
joblib.dump(scaler_handcrafted, "models/scaler_handcrafted.pkl")
joblib.dump(scaler_cnn, "models/scaler_cnn.pkl")

print("✅ Hybrid model and scalers saved successfully.")
