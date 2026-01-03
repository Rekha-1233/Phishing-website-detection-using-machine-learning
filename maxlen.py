import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
tokenizer = joblib.load("model/cnn_tokenizer.pkl")

# Example dummy URL list (to simulate your dataset)
urls = ["google.com", "example-phishing-site.net", "secure-login.com"]

# Convert to sequences
sequences = tokenizer.texts_to_sequences(urls)

# Find max length
max_len = max(len(seq) for seq in sequences)

# Save max_len
joblib.dump(max_len, "model/cnn_maxlen.pkl")

print(f"✅ Saved cnn_maxlen.pkl with value: {max_len}")
