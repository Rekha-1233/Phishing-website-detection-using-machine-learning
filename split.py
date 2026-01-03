# split_data.py

import pandas as pd
import os

# Step 1: Load the original dataset
df = pd.read_csv("Data/PhiUSIIL_Phishing_URL_Dataset.csv")  # 🔁 Change to your actual CSV filename
print("✅ Original dataset shape:", df.shape)

# Step 2: Extract only URL and label for CNN
if 'URL' in df.columns and 'label' in df.columns:
    url_df = df[['URL', 'label']].dropna()
    print("✅ URL dataset shape:", url_df.shape)
else:
    raise KeyError("❌ 'URL' or 'label' column not found in the dataset.")

# Step 3: Extract only numeric columns for ML models
numeric_df = df.select_dtypes(include=['number'])  # includes 'label' if it's numeric
print("✅ Numeric feature dataset shape:", numeric_df.shape)

# Step 4: Save both files
os.makedirs("processed", exist_ok=True)
url_df.to_csv("processed/url_data.csv", index=False)
numeric_df.to_csv("processed/numeric_data.csv", index=False)

print("\n✅ Splitting complete. Files saved to: /processed/")
