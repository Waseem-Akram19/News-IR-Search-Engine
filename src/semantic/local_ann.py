# src/semantic/local_ann.py
from annoy import AnnoyIndex
import numpy as np
import pickle
from pathlib import Path

EMB_DIR = Path("data/embeddings")
ANNOY_PATH = EMB_DIR / "annoy_index.ann"
ID_MAP_PATH = EMB_DIR / "filenames.pkl"

def build_ann(n_trees=50, metric="angular"):
    embs = np.load(EMB_DIR / "embeddings.npy")
    filenames = pickle.load(open(EMB_DIR / "filenames.pkl", "rb"))
    dims = embs.shape[1]
    idx = AnnoyIndex(dims, metric)
    for i, vec in enumerate(embs):
        idx.add_item(i, vec.tolist())
    idx.build(n_trees)
    idx.save(str(ANNOY_PATH))
    print("Annoy index saved to", ANNOY_PATH)

def query_ann(query_vec, top_k=10):
    dims = np.load(EMB_DIR / "example_vec.npy").shape[0]
    idx = AnnoyIndex(dims, "angular")
    idx.load(str(ANNOY_PATH))
    filenames = pickle.load(open(ID_MAP_PATH, "rb"))
    ids, dists = idx.get_nns_by_vector(query_vec.tolist(), top_k, include_distances=True)
    return [(filenames[i], float(dists[j])) for j, i in enumerate(ids)]
