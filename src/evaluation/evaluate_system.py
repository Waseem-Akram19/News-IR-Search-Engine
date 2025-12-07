# src/evaluation/evaluate_system.py
import csv
from pathlib import Path
from ..indexing.search_tfidf import search_tfidf
from ..semantic.semantic_search import semantic_search
from .metrics import mean_average_precision

# Expects a qrels CSV at data/processed/qrels.csv with columns:
# query_id,query_text,relevant_filenames (pipe-separated e.g. 12.txt|34.txt)

QRELS_PATH = Path("data/processed/qrels.csv")

def load_qrels(path=QRELS_PATH):
    queries = []
    qrels = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row["query_text"])
            rel = row["relevant_filenames"].split("|") if row["relevant_filenames"] else []
            qrels.append(set(rel))
    return queries, qrels

def evaluate():
    queries, qrels = load_qrels()
    retrieved_all_tfidf = []
    retrieved_all_sem = []
    for q in queries:
        tf = search_tfidf(q, top_k=10)
        tf_fnames = [t[0] for t in tf]
        sem = semantic_search(q, top_k=10)
        if isinstance(sem, list) and sem and isinstance(sem[0], tuple):
            sem_fnames = [s[0] for s in sem]
        else:
            sem_fnames = [str(m["id"]) for m in sem]  # pinecone path (if used)
        retrieved_all_tfidf.append(tf_fnames)
        retrieved_all_sem.append(sem_fnames)
    print("MAP TF-IDF:", mean_average_precision(qrels, retrieved_all_tfidf))
    print("MAP SEMANTIC:", mean_average_precision(qrels, retrieved_all_sem))

if __name__ == "__main__":
    evaluate()
