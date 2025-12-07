# src/semantic/embedder.py
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER", "all-MiniLM-L6-v2")
_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts, batch_size=64):
    """Return numpy array (n_docs, dim)."""
    return _model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def embed_query(text):
    """Return 1 vector (dim,)"""
    return _model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
