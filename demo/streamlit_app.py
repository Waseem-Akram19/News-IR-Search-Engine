# demo/streamlit_app.py
import sys
from pathlib import Path
import re
import json
import pickle
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Add project root to path so "src" imports work when Streamlit runs inside demo/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# --- Imports from your package (cached loaders used below) ---
from src.preprocessing.clean_text import clean_text
from src.indexing.search_tfidf import load_tfidf, search_tfidf
from src.indexing.bm25_search import build_bm25, load_or_build_bm25, search_bm25
from src.semantic.faiss_index import load_faiss, search_faiss
from src.topic_modeling.lda_inference import load_lda, doc_topic_vector

# --- Paths ---
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
EMB_DIR = ROOT_DIR / "data" / "embeddings"
LDA_DIR = ROOT_DIR / "data" / "lda"
TFIDF_DIR = ROOT_DIR / "data" / "tfidf"

# ---------- Helpers and cached loaders ----------

@st.cache_resource
def cached_load_tfidf():
    try:
        vectorizer, tfidf_matrix, filenames = load_tfidf()
        return vectorizer, tfidf_matrix, filenames
    except Exception as e:
        st.warning("TF-IDF artifacts not found in data/tfidf. Run TF-IDF build.")
        return None, None, None

@st.cache_resource
def cached_load_bm25():
    """
    Attempt to load BM25 if saved; if not, build from processed files.
    We assume build_bm25() creates necessary files in data/tfidf.
    """
    try:
        # We provided a helper build_or_load in earlier code; if not, call build_bm25
        bm25_obj = load_or_build_bm25()
        return bm25_obj
    except Exception:
        try:
            build_bm25()
            bm25_obj = load_or_build_bm25()
            return bm25_obj
        except Exception as e:
            st.warning("BM25 is not available and could not be built.")
            return None

@st.cache_resource
def cached_load_faiss():
    try:
        idx, filenames = load_faiss()
        return idx, filenames
    except Exception:
        st.warning("FAISS index not found. Build embeddings & FAISS first.")
        return None, None

@st.cache_resource
def cached_load_lda():
    try:
        lda, dictionary, filenames = load_lda()
        return lda, dictionary, filenames
    except Exception:
        # LDA not built
        return None, None, None

@st.cache_data
def load_metadata():
    meta_path = DATA_PROCESSED / "metadata.csv"
    if not meta_path.exists():
        return {}
    import csv
    md = {}
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            md[row["filename"]] = {"doc_id": row["doc_id"], "category": row["category"], "source_path": row["source_path"]}
    return md

# ------------ Category → Emoji Mapping ------------
CATEGORY_ICONS = {
    "business": "💼 business",
    "politics": "🏛 politics",
    "sport": "⚽ sport",
    "entertainment": "🎬 entertainment",
    "tech": "💻 tech",
    "unknown": "📄 unknown"
}

def category_with_icon(cat: str):
    cat = cat.lower().strip()
    return CATEGORY_ICONS.get(cat, CATEGORY_ICONS["unknown"])


def highlight_query_terms(text, query, max_len=500):
    """Highlight tokens from the cleaned query in the original text.
       We'll match on word boundaries, case-insensitive."""
    if not query:
        return text[:max_len] + ("..." if len(text) > max_len else "")
    # take cleaned tokens (space-separated)
    cleaned_q = clean_text(query)
    tokens = list(dict.fromkeys([t for t in cleaned_q.split() if len(t) > 1]))  # preserve order, unique
    if not tokens:
        return text[:max_len] + ("..." if len(text) > max_len else "")
    # highlight each token (escape tokens)
    def repl(m):
        return f"<mark>{m.group(0)}</mark>"
    pattern = r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b"
    try:
        highlighted = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        if len(highlighted) > max_len:
            # try to return first part but include highlights around first occurrences
            return highlighted[:max_len] + "..."
        return highlighted
    except re.error:
        return text[:max_len] + ("..." if len(text) > max_len else "")

def tfidf_top_keywords_for_doc(vectorizer, tfidf_matrix, filenames, fname, top_n=10):
    """Return top tf-idf feature names and their scores for a given processed filename."""
    try:
        if vectorizer is None or tfidf_matrix is None:
            return []
        idx = filenames.index(fname)
        # tfidf_matrix is sparse
        vec = tfidf_matrix[idx].toarray().ravel()
        top_idx = np.argsort(vec)[::-1][:top_n]
        feature_names = vectorizer.get_feature_names_out()
        return [(feature_names[i], float(vec[i])) for i in top_idx if vec[i] > 0]
    except Exception:
        return []

def cosine_sim_query_doc(vectorizer, tfidf_matrix, filenames, query_clean, fname):
    """Compute cosine similarity between query and a document using TF-IDF vectors."""
    try:
        qv = vectorizer.transform([query_clean])
        idx = filenames.index(fname)
        dv = tfidf_matrix[idx]
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(qv, dv)[0,0]
        return float(sim)
    except Exception:
        return None

def load_doc_text(fname):
    p = DATA_PROCESSED / fname
    if not p.exists():
        return "(Document not found)"
    return p.read_text(encoding="utf-8")

# ---------- UI layout ----------

st.set_page_config(page_title="News IR Search Engine (Enhanced)", layout="wide")
st.title("📚 News IR Search Engine — (TF-IDF | BM25 | Semantic)")

# Sidebar with explanations & controls
with st.sidebar:
    st.header("About this Project")
    st.write("End-to-end News IR system built for CS516 course project.")
    st.markdown("- **TF-IDF**: lexical cosine similarity (vector space).")
    st.markdown("- **BM25**: probabilistic ranking (document length normalization).")
    st.markdown("- **Semantic (FAISS)**: transformer embeddings + ANN (semantic similarity).")
    st.markdown("- **LDA**: topic modeling for topic distribution per doc.")
    st.markdown("---")

    st.subheader("Options")
    show_full_text = st.checkbox("Show full document text", value=False)
    enable_highlighting = st.checkbox("Highlight query terms in preview", value=True)
    show_topic_viz = st.checkbox("Show LDA topics (pyLDAvis)", value=False)
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)
    st.markdown("---")
    st.write("Data path: `data/processed`")
    st.write("Docs processed:", len(list((ROOT_DIR/"data"/"processed").glob("*.txt"))))

def apply_theme(dark: bool):
    if dark:
        # ---------------------- DARK MODE ----------------------
        css = """
        <style>
        .stApp {
            background-color: #0e1117 !important;
            color: #e6e6e6 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
            color: #e6e6e6 !important;
        }
        /* Inputs */
        div[data-baseweb="input"] > input {
            background-color: #1c2128 !important;
            color: #e6e6e6 !important;
        }
        .stSelectbox div, .stSlider, .stMultiSelect {
            color: #e6e6e6 !important;
        }
        /* Tabs */
        div[data-baseweb="tab-list"] {
            background-color: #161b22 !important;
        }
        div[role="tab"] {
            background-color: #1c2128 !important;
            color: #e6e6e6 !important;
        }
        div[role="tab"][aria-selected="true"] {
            background-color: #238636 !important;
            color: white !important;
            font-weight: bold;
        }
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #1c2128 !important;
            color: #f1f1f1 !important;
        }
        /* Markdown text */
        .stMarkdown, p, span, li {
            color: #e8e8e8 !important;
        }
        /* Highlighted query terms */
        mark {
            background-color: #f39c12 !important;
            color: black !important;
            padding: 2px 4px;
            border-radius: 4px;
        }
        /* Headers */
        h1, h2, h3, h4 {
            color: #f0f0f0 !important;
        }
        </style>
        """
    else:
        # ---------------------- LIGHT MODE ----------------------
        css = """
        <style>
        .stApp {
            background-color: #ffffff !important;
            color: #1f1f1f !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #f5f5f5 !important;
            color: #1f1f1f !important;
        }
        /* Inputs */
        div[data-baseweb="input"] > input {
            background-color: #ffffff !important;
            color: #1f1f1f !important;
        }
        /* Tabs */
        div[data-baseweb="tab-list"] {
            background-color: #efefef !important;
        }
        div[role="tab"] {
            background-color: #eaeaea !important;
            color: #1f1f1f !important;
        }
        div[role="tab"][aria-selected="true"] {
            background-color: #4b8df8 !important;
            color: white !important;
            font-weight: bold;
        }
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #ededed !important;
            color: #1f1f1f !important;
        }
        /* Document text */
        .stMarkdown, p, span, li {
            color: #1f1f1f !important;
        }
        /* Highlighted query terms */
        mark {
            background-color: #ffec99 !important;
            color: black !important;
            padding: 2px 4px;
            border-radius: 4px;
        }
        /* Headers */
        h1, h2, h3, h4 {
            color: #1f1f1f !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)
apply_theme(dark_mode)
# Top input area
col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input("🔎 Enter search query:", value="", placeholder="e.g. economy inflation government crisis")

with col2:
    model_choice = st.selectbox("Model:", ["TF-IDF", "BM25", "Semantic"])
    top_k = st.slider("Top K", min_value=3, max_value=20, value=5)

# Load cached resources
vectorizer, tfidf_matrix, tfidf_filenames = cached_load_tfidf()
bm25_obj = cached_load_bm25()
faiss_index, faiss_filenames = cached_load_faiss()
lda_model, lda_dictionary, lda_filenames = cached_load_lda()
metadata = load_metadata()

# Main action
if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    cleaned_query = clean_text(query)

    # run each retrieval model as needed and present results side-by-side in tabs
    tab_tf, tab_bm, tab_sem = st.tabs(["TF-IDF", "BM25", "Semantic"])

    # --- TF-IDF tab ---
    with tab_tf:
        st.subheader("TF-IDF Results")
        if vectorizer is None:
            st.error("TF-IDF artifacts not found. Build TF-IDF first.")
        else:
            try:
                tf_results = search_tfidf(cleaned_query, top_k=top_k, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, filenames=tfidf_filenames)
            except Exception:
                tf_results = search_tfidf(cleaned_query, top_k=top_k)

            for rank, (fname, score) in enumerate(tf_results, start=1):
                st.markdown(f"### {rank}. {fname} — score: {score:.4f}")
                text = load_doc_text(fname)
                # highlight if enabled
                display_text = highlight_query_terms(text, query, max_len=1500) if enable_highlighting else (text[:1500] + ("..." if len(text)>1500 else ""))
                st.markdown(display_text, unsafe_allow_html=True)

                # metadata panel
                with st.expander("Document metadata & analysis"):
                    # category
                    cat = metadata.get(fname, {}).get("category", "unknown")
                    st.write(f"**Category:** {category_with_icon(cat)}")

                    # TF-IDF top keywords
                    kw = tfidf_top_keywords_for_doc(vectorizer, tfidf_matrix, tfidf_filenames, fname, top_n=10)
                    if kw:
                        st.write("**Top TF-IDF terms (this doc):**")
                        st.write(", ".join(f"{t} ({s:.3f})" for t,s in kw))
                    # cosine similarity (tfidf)
                    sim = cosine_sim_query_doc(vectorizer, tfidf_matrix, tfidf_filenames, cleaned_query, fname)
                    if sim is not None:
                        st.write(f"**Cosine (query vs doc)**: {sim:.4f}")
                    # LDA topic dist
                    if lda_model is not None:
                        # tokens of doc
                        doc_toks = load_doc_text(fname).split()
                        try:
                            bow = lda_dictionary.doc2bow(doc_toks)
                            td = lda_model.get_document_topics(bow)
                            top_topics = sorted(td, key=lambda x: x[1], reverse=True)[:5]
                            st.write("**Top topics (LDA)**:")
                            for tid, prob in top_topics:
                                st.write(f"Topic {tid} — {prob:.3f}")
                        except Exception:
                            st.write("LDA topic info not available for this doc.")
                st.markdown("---")

    # --- BM25 tab ---
    with tab_bm:
        st.subheader("BM25 Results")
        if bm25_obj is None:
            st.error("BM25 index not available. Build BM25 index first.")
        else:
            bm_results = search_bm25(query, top_k=top_k)
            for rank, (fname, score) in enumerate(bm_results, start=1):
                st.markdown(f"### {rank}. {fname} — score: {score:.4f}")
                text = load_doc_text(fname)
                display_text = highlight_query_terms(text, query, max_len=1500) if enable_highlighting else (text[:1500] + ("..." if len(text)>1500 else ""))
                st.markdown(display_text, unsafe_allow_html=True)

                with st.expander("Document metadata & analysis"):
                    cat = metadata.get(fname, {}).get("category", "unknown")
                    st.write(f"**Category:** {category_with_icon(cat)}")

                    # BM25 does not expose per-doc tfidf; show TF-IDF keywords if available
                    if vectorizer is not None:
                        kw = tfidf_top_keywords_for_doc(vectorizer, tfidf_matrix, tfidf_filenames, fname, top_n=10)
                        if kw:
                            st.write("**Top TF-IDF terms (doc):**")
                            st.write(", ".join(f"{t} ({s:.3f})" for t,s in kw))
                    # LDA
                    if lda_model is not None:
                        doc_toks = load_doc_text(fname).split()
                        try:
                            bow = lda_dictionary.doc2bow(doc_toks)
                            td = lda_model.get_document_topics(bow)
                            top_topics = sorted(td, key=lambda x: x[1], reverse=True)[:5]
                            st.write("**Top topics (LDA)**:")
                            for tid, prob in top_topics:
                                st.write(f"Topic {tid} — {prob:.3f}")
                        except Exception:
                            st.write("LDA topic info not available for this doc.")
                st.markdown("---")

    # --- Semantic tab ---
    with tab_sem:
        st.subheader("Semantic (FAISS) Results")
        if faiss_index is None:
            st.error("FAISS index not available.")
        else:
            sem_results = search_faiss(query, top_k=top_k)
            for rank, (fname, dist) in enumerate(sem_results, start=1):
                st.markdown(f"### {rank}. {fname} — distance: {dist:.4f}")
                text = load_doc_text(fname)
                # For semantic results highlighting we still highlight query terms in preview
                display_text = highlight_query_terms(text, query, max_len=1500) if enable_highlighting else (text[:1500] + ("..." if len(text)>1500 else ""))
                st.markdown(display_text, unsafe_allow_html=True)

                with st.expander("Document metadata & analysis"):
                    cat = metadata.get(fname, {}).get("category", "unknown")
                    st.write(f"**Category:** {category_with_icon(cat)}")

                    # show tf-idf keywords (if available)
                    if vectorizer is not None:
                        kw = tfidf_top_keywords_for_doc(vectorizer, tfidf_matrix, tfidf_filenames, fname, top_n=10)
                        if kw:
                            st.write("**Top TF-IDF terms (doc):**")
                            st.write(", ".join(f"{t} ({s:.3f})" for t,s in kw))
                    # LDA topics
                    if lda_model is not None:
                        doc_toks = load_doc_text(fname).split()
                        try:
                            bow = lda_dictionary.doc2bow(doc_toks)
                            td = lda_model.get_document_topics(bow)
                            top_topics = sorted(td, key=lambda x: x[1], reverse=True)[:5]
                            st.write("**Top topics (LDA)**:")
                            for tid, prob in top_topics:
                                st.write(f"Topic {tid} — {prob:.3f}")
                        except Exception:
                            st.write("LDA topic info not available for this doc.")
                st.markdown("---")

# If user wants to see LDA viz, attempt to embed HTML
if show_topic_viz:
    lda_viz_path = LDA_DIR / "lda_viz.html"
    if lda_viz_path.exists():
        st.subheader("LDA Topic Model Visualization")
        html = lda_viz_path.read_text(encoding="utf-8")
        components.html(html, height=700)
    else:
        st.info("LDA visualization not found. Generate via pyLDAvis (src/topic_modeling/visualize_topics.py)")
# Footer tips
st.sidebar.markdown("---")
st.sidebar.write("Tip: Use the 'Show LDA topics' checkbox if you generated `data/lda/lda_viz.html`.")
st.sidebar.write("Tip: To rebuild indices, run scripts from the terminal (TF-IDF, BM25, embeddings).")
