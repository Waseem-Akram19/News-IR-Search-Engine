# src/semantic/semantic_search.py
from .faiss_index import search_faiss

def semantic_search(query: str, top_k=10):
    return search_faiss(query, top_k)
