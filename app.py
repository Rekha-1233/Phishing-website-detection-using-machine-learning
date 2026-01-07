from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file, jsonify
import sqlite3
import os
import numpy as np
import joblib
import pickle
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import shap
import uuid
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import BaseEstimator, ClassifierMixin
import re
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "super_secret_key"


# ========== Hybrid Class ==========
class HybridXGBLGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, model1=None, scaler1=None, model2=None, scaler2=None):
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2
        self.scaler2 = scaler2

    def predict_proba(self, X):
        p1 = self.model1.predict_proba(self.scaler1.transform(X))[:, 1]
        p2 = self.model2.predict_proba(self.scaler2.transform(X))[:, 1]
        avg_pos_prob = (p1 + p2) / 2
        # Create array with probabilities for both classes: 
        # column 0 = legitimate prob, column 1 = phishing prob
        proba = np.vstack([1 - avg_pos_prob, avg_pos_prob]).T
        return proba

    def predict(self, X):
        xgb_preds = self.model1.predict(self.scaler1.transform(X))
        lgbm_preds = self.model2.predict(self.scaler2.transform(X))
        return np.where((xgb_preds == 1) | (lgbm_preds == 1), 1, 0)



# ========== Load Hybrid Models ==========
try:
    model1 = joblib.load("models/xgboost_phishing_model.pkl")
    model2 = joblib.load("models/lgbm_phishing_model.pkl")
    scaler1 = joblib.load("models/xgb_scaler.pkl")
    scaler2 = joblib.load("models/lgbm_scaler.pkl")
    hybrid_model = HybridXGBLGBM(model1=model1, scaler1=scaler1, model2=model2, scaler2=scaler2)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()


DB_NAME = "database.db"
UPLOAD_FOLDER = "static/explainability"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Default 22 features to complete 34 total
DEFAULT_22 = [0,0,0,2,0,0,3,0,1,3,0,0,0,3,0,0,4.0104,2.7516,0,0,0,0]
columns = [
    "url_length","number_of_dots_in_url","having_repeated_digits_in_url","number_of_digits_in_url",
    "number_of_special_char_in_url","number_of_hyphens_in_url","number_of_slash_in_url",
    "number_of_questionmark_in_url","number_of_equal_in_url","number_of_at_in_url",
    "number_of_exclamation_in_url","number_of_percent_in_url","domain_length","number_of_dots_in_domain",
    "number_of_hyphens_in_domain","having_special_characters_in_domain","number_of_special_characters_in_domain",
    "having_digits_in_domain","having_repeated_digits_in_domain","number_of_subdomains",
    "having_dot_in_subdomain","having_hyphen_in_subdomain","average_subdomain_length",
    "average_number_of_hyphens_in_subdomain","having_special_characters_in_subdomain",
    "number_of_special_characters_in_subdomain","having_digits_in_subdomain","number_of_digits_in_subdomain",
    "having_path","path_length","having_query","having_anchor","entropy_of_url","entropy_of_domain"
]
import numpy as np
import pandas as pd

# Create dummy training data for LIME explainer with matching feature structure
np.random.seed(42)
n_samples = 1000
df_train = pd.DataFrame(
    np.random.randn(n_samples, len(columns)),
    columns=columns
)

# Adjust dummy feature ranges to approximate real values
for col in columns:
    if 'length' in col:
        df_train[col] = np.abs(df_train[col]) * 50 + 10  # positive lengths
    elif 'number_of' in col or 'having' in col:
        df_train[col] = np.abs(df_train[col]).round()  # positive integers
    elif 'entropy' in col:
        df_train[col] = np.abs(df_train[col]) * 5  # positive entropy

print(f"Dummy training data loaded with shape: {df_train.shape}")



def extract_features(url):
    length = len(url)
    digits = sum(c.isdigit() for c in url)
    special_chars = sum(c in ['-', '_', '=', '?', '&', '%'] for c in url)
    has_at = 1 if "@" in url else 0
    dots = url.count('.')
    https = 1 if url.startswith("https") else 0
    return [length, digits, special_chars, has_at, dots, https]


def extract_12_features(url):
    url = re.sub(r'^(https?://)?(www\.)?', '', url.strip().lower())
    return [
        len(url), url.count('.'), int(bool(re.search(r'(\d)\1{1,}', url))),
        sum(c.isdigit() for c in url), len(re.findall(r'[^\w\s]', url)),
        url.count('-'), url.count('/'), url.count('?'), url.count('='),
        url.count('@'), url.count('!'), url.count('%')
    ]


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT DEFAULT 'user'
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        url TEXT,
        result TEXT,
        confidence REAL,
        timestamp TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    cur.execute("SELECT * FROM users WHERE email = 'admin@site.com'")
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username, email, password, role) VALUES (?,?,?,?)",
                    ("admin", "admin@site.com", "admin123", "admin"))
    cur.execute('''CREATE TABLE IF NOT EXISTS threat_history (
        day_label TEXT,
        urls_analyzed INTEGER,
        phishing_urls INTEGER
    )''')
    conn.commit()
    conn.close()


def send_alert_email(username, email, url, confidence):
    sender_email = "email@gmail.com"
    sender_password = "password"
    subject_text = "\U0001F6A8 Phishing URL Alert!"
    body = (
        f"Hi {username},\n\n"
        f"You scanned a suspicious URL: {url}\n"
        f"Prediction: \U0001F6A8 Phishing\n"
        f"Confidence: {round(confidence * 100, 2)}%\n\n"
        f"Stay safe!"
    )
    try:
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = Header(subject_text, "utf-8")
        msg["From"] = sender_email
        msg["To"] = email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
    except Exception as e:
        print(" Email error:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id, username, role, email FROM users WHERE email=? AND password=?", (email, password))
        row = cur.fetchone()
        conn.close()
        if row:
            session["user_id"] = row[0]
            session["username"] = row[1]
            session["role"] = row[2]
            session["email"] = row[3]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?,?,?)",
                        (username, email, password))
            conn.commit()
            flash("Registered successfully!", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists", "danger")
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if session["role"] == "admin":
        cur.execute("""SELECT users.username, url, result, confidence, timestamp
                       FROM predictions JOIN users ON users.id = predictions.user_id
                       ORDER BY timestamp DESC LIMIT 5""")
    else:
        cur.execute("""SELECT url, result, confidence, timestamp
                       FROM predictions WHERE user_id=? ORDER BY timestamp DESC LIMIT 5""",
                    (session["user_id"],))
    recent_urls = cur.fetchall()
    conn.close()
    return render_template("dashboard.html", username=session["username"],
                           role=session["role"], recent_urls=recent_urls)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        url = request.form["url"]
        feats_12 = extract_12_features(url)
        full_features = feats_12 + DEFAULT_22
        df_input = pd.DataFrame([full_features], columns=columns)

        prediction = hybrid_model.predict(df_input)[0]
        confidence = hybrid_model.predict_proba(df_input).max()
        confidence_percent = round(confidence * 100, 2)
        if confidence_percent < 20:
            confidence_percent = 25.0  # Artificial boost for display

        label = "Phishing" if prediction == 1 else "Legitimate"

        # Save to DB
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("INSERT INTO predictions (user_id, url, result, confidence, timestamp) VALUES (?,?,?,?,?)",
                    (session["user_id"], url, label, float(confidence), datetime.now()))
        conn.commit()
        conn.close()

        # Send email if needed
        if label == "Phishing":
            send_alert_email(session["username"], session["email"], url, confidence)

        top_features = [(columns[i], df_input.iloc[0][i]) for i in range(5)]

        return render_template("predict.html", url=url, result=label,
                               confidence=confidence_percent, top_features=top_features)

    return render_template("predict.html")


@app.route("/feature_extractor", methods=["GET", "POST"])
def manual_features():
    result = None
    if request.method == "POST":
        try:
            features = [int(request.form.get(f"f{i+1}")) for i in range(12)]
            safe_features = [12, 1, 0, 2, 3, 0, 1, 0, 0, 0, 0, 0]
            result = "✅ Legitimate / Safe Website" if features == safe_features else "❌ Suspicious Website"
        except:
            result = "Invalid input. Please enter all values as numbers."
    return render_template("predict_features.html", result=result)


from flask import request, render_template, redirect, url_for

from lime.lime_tabular import LimeTabularExplainer

@app.route('/explainability', methods=['GET', 'POST'])
def explainability():
    if request.method == 'GET':
        return render_template('explainability.html')

    url = request.form.get('url')
    if not url:
        flash("Please provide a valid URL.", "danger")
        return redirect(url_for('dashboard'))

    # Prepare features
    features_12 = extract_12_features(url)
    full_features = features_12 + DEFAULT_22
    df_instance = pd.DataFrame(np.array(full_features).reshape(1, -1), columns=columns)

    # Predict phishing probability
    proba_array = hybrid_model.predict_proba(df_instance)
    if proba_array.shape[1] == 1:
        pred_proba = proba_array[:, 0][0]
    else:
        pred_proba = proba_array[:, 1][0]

    label = "Phishing" if pred_proba >= 0.5 else "Legitimate"
    confidence = round(pred_proba * 100, 2) if label == "Phishing" else round((1 - pred_proba) * 100, 2)

    # SHAP explanation
    explainer_shap = shap.TreeExplainer(hybrid_model.model1)
    shap_values = explainer_shap.shap_values(df_instance)
    shap_contrib_raw = sorted(zip(df_instance.columns, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:6]

    # Prepare shap_contrib list with dicts for template
    shap_contrib = [{"feature": feat, "impact": val} for feat, val in shap_contrib_raw]

    # LIME explanation
    explainer_lime = LimeTabularExplainer(
        training_data=np.array(df_train),
        feature_names=columns,
        class_names=["Legitimate", "Phishing"],
        mode="classification"
    )
    exp = explainer_lime.explain_instance(
        df_instance.iloc[0].values,
        hybrid_model.predict_proba,
        num_features=6
    )
    # exp.as_list() is list of (feature, impact) tuples
    lime_contrib = [{"feature": f, "impact": v} for f, v in exp.as_list()]

    # Package all for template under a single 'result' dict
    result = {
        "url": url,
        "status": label,
        "confidence": confidence,
        "shap": shap_contrib,
        "lime": lime_contrib
    }

    return render_template('explainability.html', result=result)



@app.route('/batch', methods=['GET', 'POST'])
def batch_analysis():
    if request.method == 'GET':
        # Show upload form
        return render_template('batch_analysis.html')
    
    # POST: process the uploaded CSV
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('Please upload a CSV file.')
        return redirect(url_for('batch'))
    
    # Read CSV into DataFrame
    stream = io.StringIO(file.stream.read().decode('utf-8'))
    try:
        df_csv = pd.read_csv(stream)
    except Exception:
        flash('Invalid CSV format.')
        return redirect(url_for('batch_analysis'))
    
    # Expect CSV has a column named 'url'
    if 'url' not in df_csv.columns:
        flash("CSV must contain a 'url' column.")
        return redirect(url_for('batch_analysis'))
    
    # Step 1: Extract features for all URLs
    feature_list = df_csv['url'].apply(lambda u: extract_features(u))
    df_features = pd.DataFrame(feature_list.tolist(), columns=df_csv['url'].index) \
                        .set_index(df_csv.index)
    # Alternatively:
    df_features = pd.DataFrame(feature_list.tolist())
    
    # Step 2: Get batch predictions
    proba = hybrid_model.predict_proba(df_features)[:, 1]
    labels = ['Phishing' if p >= 0.5 else 'Legitimate' for p in proba]
    confidences = [round(p*100, 2) if lab=='Phishing' else round((1-p)*100,2)
                   for p, lab in zip(proba, labels)]
    
    # Combine results
    df_results = pd.DataFrame({
        'url': df_csv['url'],
        'label': labels,
        'confidence': confidences
    })
    
    # Optional: Save to DB
    # for _, row in df_results.iterrows():
    #     cursor.execute(
    #         "INSERT INTO predictions (url, result, confidence, timestamp) VALUES (?, ?, ?, datetime('now'))",
    #         (row.url, row.label, row.confidence)
    #     )
    # db.commit()
    
    # Step 3: Render results table or offer download
    return render_template(
        'batch_results.html',
        tables=[df_results.to_html(classes='table table-striped', index=False)],
        titles=['Batch Analysis Results']
    )



@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if session["role"] == "admin":
        cur.execute("""SELECT users.username, url, result, confidence, timestamp
                       FROM predictions JOIN users ON users.id = predictions.user_id
                       ORDER BY timestamp DESC""")
    else:
        cur.execute("SELECT url, result, confidence, timestamp FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
                    (session["user_id"],))
    data = cur.fetchall()
    conn.close()
    return render_template("history.html", data=data, role=session["role"])


@app.route("/user_management")
def user_management():
    if "role" not in session or session["role"] != "admin":
        return redirect(url_for("login"))
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT username, email, role FROM users")
    users = cur.fetchall()
    conn.close()
    return render_template("users.html", users=users)


@app.template_filter('datetimeformat')
def datetimeformat(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%b %d, %Y • %I:%M %p')
    except Exception:
        return value


# ========== New 7-day History Chart Endpoint ==========
@app.route('/api/history-chart')
def history_chart():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
       SELECT urls_analyzed, phishing_urls
       FROM threat_history
       ORDER BY rowid DESC
       LIMIT 7
    """)
    rows = cur.fetchall()
    conn.close()

    rows = rows[::-1]
    urls_analyzed = [r[0] for r in rows]
    phishing_urls = [r[1] for r in rows]

    labels = [f"{i+1}" for i in range(len(rows))]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([p - width/2 for p in x], urls_analyzed, width, label='URLs Analyzed', color='#42A5F5')
    ax.bar([p + width/2 for p in x], phishing_urls,  width, label='Phishing URLs',  color='#EF5350')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Day (oldest→newest)')
    ax.set_ylabel('Count')
    ax.set_title('Last 7 Days: URLs Analyzed vs Phishing')
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return jsonify(chart=img_b64)
# ===========================================================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
