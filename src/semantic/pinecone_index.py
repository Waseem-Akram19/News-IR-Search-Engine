# src/semantic/pinecone_index.py
import os
import pinecone
from .embedder import embed_texts

def init_pinecone(api_key: str = None, env: str = None):
    api_key = api_key or os.environ.get("PINECONE_API_KEY")
    env = env or os.environ.get("PINECONE_ENV", "us-west1-gcp")
    if api_key is None:
        raise RuntimeError("Set PINECONE_API_KEY env var to use Pinecone.")
    pinecone.init(api_key=api_key, environment=env)

def pinecone_upsert(index_name: str, ids, texts):
    init_pinecone()
    idx = pinecone.Index(index_name)
    vectors = embed_texts(texts).tolist()
    items = [(str(i), vectors[i]) for i in range(len(vectors))]
    idx.upsert(items)

def pinecone_query(query, top_k=10, index_name="news-semantic-index"):
    init_pinecone()
    idx = pinecone.Index(index_name)
    qv = embed_texts([query])[0].tolist()
    res = idx.query(qv, top_k=top_k, include_metadata=True)
    return res["matches"]
