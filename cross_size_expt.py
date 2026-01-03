import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -------------------- Load Data -------------------- #
df = pd.read_csv("Data/data_phish.csv")
df = df[df.status.isin(['phishing', 'legitimate'])]
df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
urls = df['url'].values
labels = df['label'].values

# -------------------- Load Tokenizer and CNN Embedder -------------------- #
with open("models/cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

cnn_model = load_model("models/cnn_url_model.keras")
cnn_embedder = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# -------------------- Feature Extractor -------------------- #
def extract_features(url_list):
    handcrafted = np.array([
        [len(u), u.count('.'), u.count('-'),
         int('https' in u), int('@' in u),
         u.count('/'), u.count('='), u.count('&'),
         u.count('%'), u.count('?')]
        for u in url_list
    ])
    sequences = tokenizer.texts_to_sequences(url_list)
    padded = pad_sequences(sequences, maxlen=200)
    cnn_feats = cnn_embedder.predict(padded, verbose=0)
    return np.hstack([handcrafted, cnn_feats])

# -------------------- Create Small and Large Sets -------------------- #
total_rows = len(df)
print(f"✅ Total samples in dataset: {total_rows}")

# Set small sample size to a fraction of dataset
small_frac = 0.2  # 20% for small dataset
df_small = df.sample(frac=small_frac, random_state=42)
df_large = df.copy()  # full dataset

# -------------------- Extract Features -------------------- #
print("✅ Extracting features (this may take some time)...")
X_small = extract_features(df_small['url'].values)
y_small = df_small['label'].values

X_large = extract_features(df_large['url'].values)
y_large = df_large['label'].values

# Cross-size (CNN trained on small, RF on combined)
X_cross = np.vstack([X_small, X_large])
y_cross = np.hstack([y_small, y_large])

# -------------------- Train and Evaluate -------------------- #
settings = {
    "Both-Small": (X_small, y_small),
    "Both-Large": (X_large, y_large),
    "Cross-Size": (X_cross, y_cross)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n✅ Running 5-Fold Evaluation...")
for name, (X, y) in settings.items():
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

print("\n✅ Cross-Size Experiment Completed!")
