import csv
from pathlib import Path
from ..indexing.search_tfidf import search_tfidf
from ..indexing.bm25_search import search_bm25
from ..semantic.faiss_index import search_faiss
from .metrics import mean_average_precision, precision_at_k, recall_at_k, ndcg_at_k

QRELS_PATH = Path("data/processed/qrels.csv")

def load_qrels():
    queries = []
    qrels = []
    with QRELS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row["query_text"])
            rel = row["relevant_filenames"].split("|") if row["relevant_filenames"] else []
            qrels.append(set(rel))
    return queries, qrels


def evaluate_model(name, queries, qrels, search_fn):
    all_retrieved = []
    for q in queries:
        results = search_fn(q, top_k=10)
        fnames = [r[0] for r in results]  # extract filenames
        all_retrieved.append(fnames)

    map_score = mean_average_precision(qrels, all_retrieved)
    p5 = sum(precision_at_k(qr, ret, 5) for qr, ret in zip(qrels, all_retrieved)) / len(qrels)
    r5 = sum(recall_at_k(qr, ret, 5) for qr, ret in zip(qrels, all_retrieved)) / len(qrels)
    ndcg5 = sum(ndcg_at_k(qr, ret, 5) for qr, ret in zip(qrels, all_retrieved)) / len(qrels)

    return {
        "MAP": map_score,
        "Precision@5": p5,
        "Recall@5": r5,
        "NDCG@5": ndcg5
    }


def evaluate():
    queries, qrels = load_qrels()

    print("\n========= Evaluation Results =========")

    # TF-IDF
    tfidf_scores = evaluate_model("TF-IDF", queries, qrels, search_tfidf)
    print("\nTF-IDF:", tfidf_scores)

    # BM25
    bm25_scores = evaluate_model("BM25", queries, qrels, search_bm25)
    print("\nBM25:", bm25_scores)

    # Semantic
    semantic_scores = evaluate_model("Semantic", queries, qrels, search_faiss)
    print("\nSemantic:", semantic_scores)

    print("\n======================================")

if __name__ == "__main__":
    evaluate()
