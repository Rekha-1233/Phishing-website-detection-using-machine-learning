# url_features.py
from urllib.parse import urlparse
import re
import numpy as np

def extract_features(url):
    features = []

    # Parsed components
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    scheme = parsed.scheme

    # Feature 1: Length of the URL
    features.append(len(url))

    # Feature 2: Count of '.'
    features.append(url.count('.'))

    # Feature 3: Use of HTTPS
    features.append(1 if scheme == 'https' else 0)

    # Feature 4: Count of '@' symbol
    features.append(url.count('@'))

    # Feature 5: Count of '?' symbol
    features.append(url.count('?'))

    # Feature 6: Count of '-' symbol
    features.append(url.count('-'))

    # Feature 7: Count of '=' symbol
    features.append(url.count('='))

    # Feature 8: Count of digits
    digits = len(re.findall(r'\d', url))
    features.append(digits)

    # Feature 9: URL contains IP address
    ip_match = re.match(r'http[s]?://(\d{1,3}\.){3}\d{1,3}', url)
    features.append(1 if ip_match else 0)

    # Feature 10: Length of domain
    features.append(len(domain))

    # Feature 11: Count of subdomains
    features.append(domain.count('.'))

    # Feature 12: Suspicious keywords
    suspicious_words = ['login', 'verify', 'bank', 'secure', 'account', 'update']
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)

    # Feature 13: Shortening service
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co']
    features.append(1 if any(short in url for short in shorteners) else 0)

    # Feature 14: Path length
    features.append(len(path))

    # Feature 15–34: Dummy 0.1 values to complete 34 features (you can replace with real ones later)
    while len(features) < 34:
        features.append(0.1)

    return np.array(features)
