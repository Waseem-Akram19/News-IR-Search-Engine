# src/indexing/bm25_search.py

from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
import numpy as np

PROCESSED_DIR = Path("data/processed")
BM25_DIR = Path("data/tfidf")
BM25_DIR.mkdir(parents=True, exist_ok=True)

BM25_FILE = BM25_DIR / "bm25.pkl"
FILENAMES_FILE = BM25_DIR / "bm25_filenames.pkl"


# ------------------------------------------------------------
# Build BM25 index (your original function preserved)
# ------------------------------------------------------------
def build_bm25():
    docs_tokens = []
    filenames = []

    for p in sorted(PROCESSED_DIR.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name):
        text = p.read_text(encoding="utf-8")
        tokens = text.split()
        docs_tokens.append(tokens)
        filenames.append(p.name)

    bm25 = BM25Okapi(docs_tokens)

    pickle.dump(bm25, open(BM25_FILE, "wb"))
    pickle.dump(filenames, open(FILENAMES_FILE, "wb"))

    print("BM25 saved to", BM25_DIR)
    return bm25, filenames


# ------------------------------------------------------------
# Load BM25 model if exists
# ------------------------------------------------------------
def load_bm25():
    if BM25_FILE.exists() and FILENAMES_FILE.exists():
        bm25 = pickle.load(open(BM25_FILE, "rb"))
        filenames = pickle.load(open(FILENAMES_FILE, "rb"))
        return bm25, filenames
    else:
        raise FileNotFoundError("BM25 model not found.")


# ------------------------------------------------------------
# Load or Build BM25 (required by Streamlit)
# ------------------------------------------------------------
def load_or_build_bm25():
    try:
        return load_bm25()
    except:
        return build_bm25()


# ------------------------------------------------------------
# Search BM25 (your original function, renamed to fit Streamlit)
# ------------------------------------------------------------
def search_bm25(query: str, top_k: int = 10):
    bm25, filenames = load_or_build_bm25()

    q_tokens = query.split()
    scores = bm25.get_scores(q_tokens)

    idx = np.argsort(scores)[::-1][:top_k]
    return [(filenames[i], float(scores[i])) for i in idx]
