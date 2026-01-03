import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
df = pd.read_csv("Data/phishing_cleaned_final.csv")  # Replace with your real CSV file name

# Use only the selected 34 features
final_features = [
    "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url", "number_of_digits_in_url",
    "number_of_special_char_in_url", "number_of_hyphens_in_url", "number_of_slash_in_url",
    "number_of_questionmark_in_url", "number_of_equal_in_url", "number_of_at_in_url",
    "number_of_exclamation_in_url", "number_of_percent_in_url", "domain_length", "number_of_dots_in_domain",
    "number_of_hyphens_in_domain", "having_special_characters_in_domain", "number_of_special_characters_in_domain",
    "having_digits_in_domain", "having_repeated_digits_in_domain", "number_of_subdomains",
    "having_dot_in_subdomain", "having_hyphen_in_subdomain", "average_subdomain_length",
    "average_number_of_hyphens_in_subdomain", "having_special_characters_in_subdomain",
    "number_of_special_characters_in_subdomain", "having_digits_in_subdomain", "number_of_digits_in_subdomain",
    "having_path", "path_length", "having_query", "having_anchor", "entropy_of_url", "entropy_of_domain"
]

X = df[final_features]
y = df["Type"]  # Your target column (0 = Legit, 1 = Phishing)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and Scaler saved successfully!")
