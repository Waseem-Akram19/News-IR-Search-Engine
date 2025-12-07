# src/semantic/build_embeddings.py
from pathlib import Path
import numpy as np
import pickle
from .embedder import embed_texts
import os

PROCESSED_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

def build_and_save_embeddings():
    texts = []
    filenames = []
    for p in sorted(PROCESSED_DIR.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name):
        texts.append(p.read_text(encoding="utf-8"))
        filenames.append(p.name)
    embs = embed_texts(texts)
    np.save(EMB_DIR / "embeddings.npy", embs)
    pickle.dump(filenames, open(EMB_DIR / "filenames.pkl", "wb"))
    np.save(EMB_DIR / "example_vec.npy", embs[0])
    print("Saved embeddings to", EMB_DIR, "shape=", embs.shape)

if __name__ == "__main__":
    build_and_save_embeddings()
