from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model & tokenizer
model = load_model("cnn_url_model.keras")
with open("cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

# Prediction function
def predict_url(url):
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=200)
    pred = model.predict(padded)[0][0]
    return "Phishing" if pred >= 0.5 else "Legitimate"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            prediction = predict_url(url)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
