import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import evaluation + search
from src.evaluation.evaluate_system import load_qrels
from src.indexing.search_tfidf import search_tfidf
from src.indexing.bm25_search import search_bm25
from src.semantic.faiss_index import load_faiss, search_faiss
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
)

# --- Page Config ---
st.set_page_config(page_title="IR System Evaluation Dashboard", layout="wide")
st.title("📊 IR System Evaluation Dashboard")

# Load Qrels
queries, qrels = load_qrels()

st.subheader("Ground Truth Queries (Qrels)")
qrel_df = pd.DataFrame({
    "Query": queries,
    "Relevant Docs": [", ".join(list(q)) for q in qrels]
})
st.dataframe(qrel_df, use_container_width=True)

st.write("---")

top_k = st.slider("Top-K for Evaluation Metrics", 3, 20, 10)

# Load FAISS
faiss_index, faiss_filenames = load_faiss()

# --- RUN EVALUATION BUTTON ---
if st.button("Run Evaluation"):

    st.subheader("🔍 Evaluating Retrieval Models...")

    results = {
        "Model": [],
        "MAP": [],
        "Avg Precision@K": [],
        "Avg Recall@K": [],
        "Avg NDCG@K": [],
    }

    def evaluate_model(model_name, retrieve_fn):
        prec_list, rec_list, ndcg_list, ap_list = [], [], [], []

        for q, rel in zip(queries, qrels):
            retrieved = retrieve_fn(q)
            retrieved_ids = [d[0] for d in retrieved]

            prec_list.append(precision_at_k(rel, retrieved_ids, top_k))
            rec_list.append(recall_at_k(rel, retrieved_ids, top_k))
            ndcg_list.append(ndcg_at_k(rel, retrieved_ids, top_k))
            ap_list.append(average_precision(rel, retrieved_ids))

        return (
            sum(ap_list)/len(ap_list),
            sum(prec_list)/len(prec_list),
            sum(rec_list)/len(rec_list),
            sum(ndcg_list)/len(ndcg_list)
        )

    # TF-IDF Evaluation
    map_, p_, r_, ndcg_ = evaluate_model(
        "TF-IDF",
        lambda q: search_tfidf(q, top_k=top_k)
    )
    results["Model"].append("TF-IDF")
    results["MAP"].append(map_)
    results["Avg Precision@K"].append(p_)
    results["Avg Recall@K"].append(r_)
    results["Avg NDCG@K"].append(ndcg_)

    # BM25 Evaluation
    map_, p_, r_, ndcg_ = evaluate_model(
        "BM25",
        lambda q: search_bm25(q, top_k=top_k)
    )
    results["Model"].append("BM25")
    results["MAP"].append(map_)
    results["Avg Precision@K"].append(p_)
    results["Avg Recall@K"].append(r_)
    results["Avg NDCG@K"].append(ndcg_)

    # Semantic (FAISS)
    map_, p_, r_, ndcg_ = evaluate_model(
        "Semantic",
        lambda q: search_faiss(q, top_k=top_k)
    )
    results["Model"].append("Semantic")
    results["MAP"].append(map_)
    results["Avg Precision@K"].append(p_)
    results["Avg Recall@K"].append(r_)
    results["Avg NDCG@K"].append(ndcg_)

    # Display Metrics Table
    df = pd.DataFrame(results)
    st.subheader("📌 Evaluation Results")
    st.dataframe(df, use_container_width=True)

    # Plot Charts
    st.subheader("📈 Performance Comparison")
    st.bar_chart(df.set_index("Model")["MAP"])
    st.bar_chart(df.set_index("Model")["Avg Precision@K"])
    st.bar_chart(df.set_index("Model")["Avg Recall@K"])
    st.bar_chart(df.set_index("Model")["Avg NDCG@K"])

    st.success("Evaluation Completed ✓")
