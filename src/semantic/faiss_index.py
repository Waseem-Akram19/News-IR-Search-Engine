# src/semantic/faiss_index.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from .embedder import embed_query

EMB_DIR = Path("data/embeddings")

def load_faiss():
    index = faiss.read_index(str(EMB_DIR / "faiss.index"))
    filenames = pickle.load(open(EMB_DIR / "filenames.pkl", "rb"))
    return index, filenames

def search_faiss(query: str, top_k=10):
    index, filenames = load_faiss()

    qv = embed_query(query).astype(np.float32).reshape(1, -1)

    distances, indices = index.search(qv, top_k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for idx, dist in zip(indices, distances):
        fname = filenames[idx]
        results.append((fname, float(dist)))

    return results
