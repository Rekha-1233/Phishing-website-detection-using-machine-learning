# make_no.py - Final Stable Version for Controlled Noise Injection

import pandas as pd
import numpy as np
import os

# Load the original dataset
df = pd.read_csv("Data/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"✅ Original shape: {df.shape}")

# Drop rows with missing values
df.dropna(inplace=True)

# Keep only numeric columns (including label)
numeric_df = df.select_dtypes(include=["number"]).copy()
print(f"✅ Numeric features shape: {numeric_df.shape}")

# Separate features and label
X = numeric_df.drop(columns=["label"])
y = numeric_df["label"]

# Convert all features to float to allow adding noise
X = X.astype(float)

# Add Gaussian noise to 10% of the data
num_samples = int(0.1 * len(X))
noise_indices = np.random.choice(X.index, size=num_samples, replace=False)

for col in X.columns:
    noise = np.random.normal(loc=0, scale=X[col].std() * 0.2, size=num_samples)
    X.loc[noise_indices, col] += noise

# Combine back the label
X["label"] = y

# Save to new CSV
os.makedirs("Data", exist_ok=True)
X.to_csv("Data/noisy_phiusiil_dataset.csv", index=False)
print("✅ Saved noisy dataset to: Data/noisy_phiusiil_dataset.csv")
