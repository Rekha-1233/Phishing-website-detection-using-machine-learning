# preprocess.py
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_22 = [0, 0, 0, 2, 0, 0, 3, 0, 1, 3, 0, 0, 0, 3, 0, 0,
              4.0104, 2.7516, 0, 0, 0, 0]

def normalize_url(raw_url):
    url = raw_url.strip().lower()
    url = re.sub(r'^(https?://)?(www\.)?', '', url)
    return url

def extract_12_features(url):
    url = normalize_url(url)
    return [
        len(url),
        url.count('.'),
        int(bool(re.search(r'(\d)\1{1,}', url))),
        sum(c.isdigit() for c in url),
        len(re.findall(r'[^\w\s]', url)),
        url.count('-'),
        url.count('/'),
        url.count('?'),
        url.count('='),
        url.count('@'),
        url.count('!'),
        url.count('%'),
    ]

# Load data
df = pd.read_csv("data/phishing_cleaned.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# Map labels to 0/1 if needed
df['Type'] = df['Type'].map({1: 1, -1: 0, 0: 0}).astype(int)

features = []
for _, row in df.iterrows():
    url = row.get('url', '') or row.get('URL', '')
    feats12 = extract_12_features(url)
    full_feats = feats12 + DEFAULT_22
    features.append(full_feats)

X = np.array(features, dtype=float)
y = df['Type'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("processed", exist_ok=True)
np.savez("processed/features_labels.npz", X=X_scaled, y=y)

import joblib
joblib.dump(scaler, "processed/scaler.pkl")

print("Preprocessing done: features and labels saved, scaler saved.")