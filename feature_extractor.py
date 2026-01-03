# feature_extractor.py
import re
from urllib.parse import urlparse
import numpy as np

def extract_top_features_from_url(url: str) -> list:
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = [
        len(url),                         # url_length
        url.count('-'),                   # number_of_hyphens_in_url
        len(re.findall(r'\W', url)),      # number_of_special_char_in_url
        len(domain),                      # domain_length
        domain.count('.'),               # number_of_dots_in_domain
        len(re.findall(r'\d', url)),      # number_of_digits_in_url
        path.count('/'),                  # number_of_slash_in_url
        len(path),                        # path_length
        url.count('='),                   # number_of_equal_in_url
        url.count('@'),                   # number_of_at_in_url
        1 if '#' in url else 0,           # having_anchor
        sum(1 for c in url if c.isupper())  # number_of_uppercase
    ]
    return np.array(features).reshape(1, -1)
