import pandas as pd

df = pd.read_csv('Data/Dataset.csv')

columns_to_drop = [
    'number_of_underline_in_url',
    'number_of_dollar_in_url',
    'number_of_hashtag_in_url',
    'having_repeated_digits_in_subdomain',
    'average_number_of_dots_in_subdomain',
    'having_fragment',
    'number_of_digits_in_domain'
]

df.drop(columns=columns_to_drop, inplace=True)
df.dropna(inplace=True)

# Save cleaned dataset
df.to_csv('Data/phishing_cleaned.csv', index=False)
print("✅ Cleaned dataset saved!")
