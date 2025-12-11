#############################################################
#   STREAMLIT APP WITH SEARCH ENGINE + EVALUATION DASHBOARD  #
#############################################################

import sys
from pathlib import Path
import re
import csv
import json
import pickle
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Add project root so "src" imports work
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ---- Project Imports ----
from src.preprocessing.clean_text import clean_text
from src.indexing.search_tfidf import load_tfidf, search_tfidf
from src.indexing.bm25_search import load_or_build_bm25, search_bm25
from src.semantic.faiss_index import load_faiss, search_faiss
from src.topic_modeling.lda_inference import load_lda
from src.evaluation.evaluate_system import load_qrels
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, mean_average_precision, ndcg_at_k
)

# ---- Paths ----
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
LDA_DIR = ROOT_DIR / "data" / "lda"

#############################################################
#                         CACHES
#############################################################

@st.cache_resource
def cached_load_tfidf():
    try:
        return load_tfidf()
    except:
        return None, None, None

@st.cache_resource
def cached_load_bm25():
    try:
        return load_or_build_bm25()
    except:
        return None

@st.cache_resource
def cached_load_faiss():
    try:
        return load_faiss()
    except:
        return None, None

@st.cache_resource
def cached_load_lda():
    try:
        return load_lda()
    except:
        return None, None, None

#############################################################
#                   THEME HANDLING (DARK/LIGHT)
#############################################################

def apply_theme(dark: bool):
    if dark:
        css = """
        <style>
        .stApp { background:#0e1117; color:#e6e6e6; }
        section[data-testid="stSidebar"] { background:#161b22 !important; }
        .stMarkdown, p, span, li { color:#e8e8e8 !important; }
        div[data-baseweb="input"] > input { background:#1c2128; color:#e6e6e6 !important; }
        div[role="tab"] { background:#1c2128; color:#e6e6e6 !important; }
        div[role="tab"][aria-selected="true"] { background:#238636 !important; color:white !important; }
        mark { background:#f39c12 !important; color:black; padding:2px 4px; border-radius:4px; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background:white; color:#1f1f1f; }
        section[data-testid="stSidebar"] { background:#f5f5f5; }
        div[data-baseweb="input"] > input { background:white; color:black !important; }
        div[role="tab"][aria-selected="true"] { background:#4b8df8 !important; color:white !important; }
        mark { background:#ffec99 !important; color:black; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

#############################################################
#                UTILITIES FOR SEARCH PAGE
#############################################################

def highlight_query_terms(text, query, max_len=500):
    cleaned_q = clean_text(query)
    tokens = [t for t in cleaned_q.split() if len(t) > 1]
    if not tokens:
        return text[:max_len]
    pattern = r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b"
    return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", text, flags=re.I)[:max_len]


#############################################################
#                     STREAMLIT LAYOUT
#############################################################

st.set_page_config(page_title="News IR System", layout="wide")

st.title("📚 News IR Search Engine — TF-IDF | BM25 | Semantic | Evaluation")

# ---- Sidebar ----
with st.sidebar:

    st.header("Options")

    dark_mode = st.checkbox("🌙 Dark Mode", value=False)
    apply_theme(dark_mode)

    mode = st.radio(
        "Select Mode:",
        ["🔍 Search Engine", "📊 Evaluation Dashboard"]
    )

    st.markdown("---")
    st.write(f"Documents Loaded: {len(list((DATA_PROCESSED).glob('*.txt')))}")


#############################################################
#                     MODE 1 → SEARCH ENGINE
#############################################################
if mode == "🔍 Search Engine":

    st.subheader("Enter your search query:")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Query", placeholder="e.g. economy inflation government crisis")
    with col2:
        search_model = st.selectbox("Model", ["TF-IDF", "BM25", "Semantic"])

    top_k = st.slider("Top K", 3, 20, 5)
    enable_highlight = st.checkbox("Highlight Query Terms", True)

    vectorizer, tfidf_matrix, tfidf_filenames = cached_load_tfidf()
    bm25_obj = cached_load_bm25()
    faiss_index, faiss_filenames = cached_load_faiss()

    if st.button("Search"):
        cleaned_query = clean_text(query)

        results = []
        if search_model == "TF-IDF":
            results = search_tfidf(cleaned_query, top_k)
        elif search_model == "BM25":
            results = search_bm25(query, top_k)
        elif search_model == "Semantic":
            results = search_faiss(query, top_k)

        st.subheader(f"{search_model} Results")

        for rank, (fname, score) in enumerate(results, start=1):
            st.markdown(f"### {rank}. {fname} — **{score:.4f}**")
            path = DATA_PROCESSED / fname
            text = path.read_text(encoding="utf-8")
            preview = highlight_query_terms(text, query, 1200) if enable_highlight else text[:1200]
            st.markdown(preview, unsafe_allow_html=True)
            st.markdown("---")


#############################################################
#                      MODE 2 → EVALUATION
#############################################################
else:
    st.subheader("📊 Retrieval Evaluation Dashboard")

    try:
        queries, qrels = load_qrels()
    except:
        st.error("❌ qrels.csv not found in data/processed/")
        st.stop()

    top_k = st.slider("Evaluation Top-K", 5, 20, 10)

    if st.button("Run Evaluation"):

        vectorizer, tfidf_matrix, tfidf_filenames = cached_load_tfidf()
        bm25_obj = cached_load_bm25()
        faiss_index, faiss_filenames = cached_load_faiss()

        all_tf = []
        all_bm = []
        all_sem = []

        st.write("Running evaluation...")

        for q in queries:
            cq = clean_text(q)

            # TF-IDF
            tf = search_tfidf(cq, top_k)
            all_tf.append([x[0] for x in tf])

            # BM25
            bm = search_bm25(q, top_k)
            all_bm.append([x[0] for x in bm])

            # Semantic
            sem = search_faiss(q, top_k)
            all_sem.append([x[0] for x in sem])

        # ---- Compute Metrics ----
        def build_metrics(model_name, retrieved):
            return {
                "Model": model_name,
                "MAP": mean_average_precision(qrels, retrieved),
                "P@K": np.mean([precision_at_k(r, ret, top_k) for r, ret in zip(qrels, retrieved)]),
                "R@K": np.mean([recall_at_k(r, ret, top_k) for r, ret in zip(qrels, retrieved)]),
                "NDCG@K": np.mean([ndcg_at_k(r, ret, top_k) for r, ret in zip(qrels, retrieved)]),
            }

        results_table = [
            build_metrics("TF-IDF", all_tf),
            build_metrics("BM25", all_bm),
            build_metrics("Semantic", all_sem),
        ]

        st.success("Evaluation Complete ✓")

        st.table(results_table)

        st.markdown("### Interpretation:")
        st.write("- **MAP** reflects ranking quality.")
        st.write("- **NDCG** measures ordering of relevant documents.")
        st.write("- **Precision/Recall** measure retrieval coverage.")

