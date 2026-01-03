import pandas as pd

# ----------------------------
# STEP 1: Read CSV safely
# ----------------------------
bad_lines = []

def handle_bad_line(line):
    bad_lines.append(line)
    return None  # Skip the bad line

try:
    df = pd.read_csv(
        "Data/urlset.csv",
        encoding='latin1',
        on_bad_lines=handle_bad_line,
        engine='python'  # ✅ Required when using on_bad_lines as a function
    )
except Exception as e:
    print("❌ Error reading the dataset:", e)
    exit()

print(f"\n✅ Loaded dataset shape: {df.shape}")
print(f"⚠️ Skipped {len(bad_lines)} bad lines\n")

# ----------------------------
# STEP 2: Data Cleaning
# ----------------------------
print("✅ Checking for missing values...\n", df.isnull().sum())

df = df.drop_duplicates()
print("\n✅ After removing duplicates:", df.shape)

print("\n✅ Label distribution:\n", df['label'].value_counts())

# ----------------------------
# STEP 3: Prepare for Models
# ----------------------------

# For Structured Models
X_struct = df.drop(columns=['domain', 'label'], errors='ignore')
y_struct = df['label']

print("\n📊 Structured Features Shape:", X_struct.shape)
print("🎯 Target Shape:", y_struct.shape)

# Option 1: Drop rows with missing values (simpler)
df = df.dropna()

# Option 2: Fill with mean (more advanced)
# df = df.fillna(df.mean(numeric_only=True))

# Recalculate splits after cleaning
X_struct = df.drop(columns=['domain', 'label'], errors='ignore')
y_struct = df['label']
df_cnn = df[['domain', 'label']].copy()


# For CNN Model (URL string-based)
df_cnn = df[['domain', 'label']].copy()
print("\n🔤 CNN Data Shape:", df_cnn.shape)
print("🔍 Example URL:", df_cnn['domain'].iloc[0])
