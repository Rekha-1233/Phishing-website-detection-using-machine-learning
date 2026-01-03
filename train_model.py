# train_model.py
# Purpose: Preprocess, summarize, and prepare final cleaned phishing dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the raw dataset (already feature-cleaned)
df = pd.read_csv("data/phishing_cleaned.csv")
print("Original dataset shape:", df.shape)

# Preview first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset info
print("\nDataset Info:")
df.info()

# Count and remove missing values
missing_before = df.isnull().sum().sum()
df.dropna(inplace=True)
missing_after = df.isnull().sum().sum()

# Count and remove duplicates
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()

# Print dataset shape after cleaning
print(f"\nDataset shape after cleaning: {df.shape}")
print(f"Missing values removed: {missing_before - missing_after}")
print(f"Duplicate rows removed: {duplicates_before - duplicates_after}")
print(f"Total rows removed: {missing_before + duplicates_before - missing_after - duplicates_after}")

# Ensure label column 'Type' is integer and properly mapped (optional)
if df['Type'].dtype != int:
    df['Type'] = df['Type'].astype(int)

# Label distribution
label_counts = df['Type'].value_counts().sort_index()
label_percent = df['Type'].value_counts(normalize=True).sort_index() * 100

print("\nFinal Label Distribution:")
for label, count in label_counts.items():
    label_name = "Legitimate" if label == 0 else "Phishing"
    print(f"{label_name} ({label}): {count} ({label_percent[label]:.2f}%)")

# Feature and target split
X = df.drop("Type", axis=1)
y = df["Type"]

print(f"\nNumber of features used: {X.shape[1]}")
print(f"Total records after cleaning: {df.shape[0]}")

# Optional: train-test split preview (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Save the cleaned full dataset for downstream training
df.to_csv("data/phishing_cleaned_final.csv", index=False)
print("\nCleaned dataset saved as 'data/phishing_cleaned_final.csv'")

# (Optional) Save train and test splits if needed separately
# X_train.to_csv("data/X_train.csv", index=False)
# X_test.to_csv("data/X_test.csv", index=False)
# y_train.to_csv("data/y_train.csv", index=False)
# y_test.to_csv("data/y_test.csv", index=False)
