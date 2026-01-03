from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load tokenizer
with open("cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load CNN model (use .keras format!)
model = load_model("cnn_url_model.keras")

# ✅ Define prediction function
def predict_url(url):
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded)[0][0]
    return "Phishing" if prediction > 0.5 else "Legitimate"

# ✅ Test
url = "http://secure-paypal-login.com"
print("Prediction:", predict_url(url))
