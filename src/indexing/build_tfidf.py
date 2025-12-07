# src/indexing/build_tfidf.py
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_DIR = Path("data/processed")
TFIDF_DIR = Path("data/tfidf")
TFIDF_DIR.mkdir(parents=True, exist_ok=True)

def load_processed_texts():
    texts = []
    filenames = []
    for p in sorted(PROCESSED_DIR.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name):
        texts.append(p.read_text(encoding="utf-8"))
        filenames.append(p.name)
    return texts, filenames

def build_tfidf(max_features=None):
    texts, filenames = load_processed_texts()
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    # persist
    pickle.dump(vectorizer, open(TFIDF_DIR / "vectorizer.pkl", "wb"))
    pickle.dump(tfidf_matrix, open(TFIDF_DIR / "tfidf_matrix.pkl", "wb"))
    pickle.dump(filenames, open(TFIDF_DIR / "filenames.pkl", "wb"))
    print("Saved TF-IDF artifacts to", TFIDF_DIR)

if __name__ == "__main__":
    build_tfidf()
