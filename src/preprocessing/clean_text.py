# src/preprocessing/clean_text.py
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure you downloaded NLTK resources:
# python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def basic_clean(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)         # remove URLs
    text = re.sub(r"[^a-z0-9\s']", " ", text)            # keep simple apostrophes if any
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_simple(text: str) -> list[str]:
    return text.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _stopwords and len(t) > 1]

def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]

def clean_text(text: str) -> str:
    """Full cleaning pipeline returns a cleaned string."""
    c = basic_clean(text)
    toks = tokenize_simple(c)
    toks = remove_stopwords(toks)
    toks = lemmatize(toks)
    return " ".join(toks)
