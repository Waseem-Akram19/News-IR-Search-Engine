# src/semantic/build_embeddings_faiss.py
from pathlib import Path
import numpy as np
import pickle
import faiss
from .embedder import embed_texts

PROCESSED_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

def build_faiss_index():
    texts = []
    filenames = []

    # Load processed files
    for p in sorted(PROCESSED_DIR.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name):
        texts.append(p.read_text(encoding="utf-8"))
        filenames.append(p.name)

    # Embed all documents
    print("Embedding documents...")
    embeddings = embed_texts(texts)
    d = embeddings.shape[1]  # vector dimension

    # Save embeddings + filenames
    np.save(EMB_DIR / "embeddings.npy", embeddings)
    pickle.dump(filenames, open(EMB_DIR / "filenames.pkl", "wb"))

    # Build FAISS Index (L2)
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))

    # Save index
    faiss.write_index(index, str(EMB_DIR / "faiss.index"))
    print("FAISS index built and saved to data/embeddings/faiss.index")

if __name__ == "__main__":
    build_faiss_index()
