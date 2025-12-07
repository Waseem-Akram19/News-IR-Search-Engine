# src/indexing/search_tfidf.py
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

TFIDF_DIR = Path("data/tfidf")

def load_tfidf():
    vec = pickle.load(open(TFIDF_DIR / "vectorizer.pkl", "rb"))
    mat = pickle.load(open(TFIDF_DIR / "tfidf_matrix.pkl", "rb"))
    filenames = pickle.load(open(TFIDF_DIR / "filenames.pkl", "rb"))
    return vec, mat, filenames

def search_tfidf(query: str, top_k: int = 10):
    vec, mat, filenames = load_tfidf()
    qv = vec.transform([query])
    scores = cosine_similarity(qv, mat).reshape(-1)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(filenames[i], float(scores[i])) for i in top_idx]
