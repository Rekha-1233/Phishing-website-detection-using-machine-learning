import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")
X_test = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv").values.ravel()

pred = model.predict(X_test[:5])
print("Predictions:", pred)
print("Actual     :", y_test[:5])
