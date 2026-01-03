import pandas as pd
df = pd.read_csv("Data/phishing_cleaned_final.csv")   # Adjust the path as needed


print("Label values before mapping:", df['Type'].unique())
# Keep only known values
df = df[df['Type'].isin([-1, 1])]
df['Type'] = df['Type'].map({-1: 0, 1: 1}).astype(int)
print("Label distribution after mapping:", df['Type'].value_counts())
