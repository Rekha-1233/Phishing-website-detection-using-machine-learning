import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Step 1: Load dataset
csv_path = os.path.join("Data", "data_phish.csv")
df = pd.read_csv(csv_path)

# ✅ Step 2: Filter & clean
df = df[['url', 'status']]
df.dropna(inplace=True)
df = df[df['status'].isin(['phishing', 'legitimate'])]
df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})

print(f"✅ Total cleaned records: {len(df)}")

# ✅ Step 3: Tokenization (character-level)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['url'])
sequences = tokenizer.texts_to_sequences(df['url'])

# ✅ Step 4: Padding
maxlen = 200
X = pad_sequences(sequences, maxlen=maxlen)
y = df['label'].values

# ✅ Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Build CNN Model
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# ✅ Step 7: Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

# ✅ Step 8: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# ✅ Step 9: Save model & tokenizer
model.save("cnn_url_model.keras")  # safer format than .h5
with open("cnn_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\n✅ CNN model and tokenizer saved successfully!")

