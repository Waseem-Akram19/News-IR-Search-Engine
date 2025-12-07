# src/main.py

from src.preprocessing.clean_text import clean_text
from src.indexing.search_tfidf import search_tfidf
from src.indexing.bm25_search import bm25_search
from src.semantic.semantic_search import semantic_search

def interactive():
    print("Interactive News IR (type 'exit' to quit')")
    while True:
        q = input("\nQuery> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break

        q_clean = clean_text(q)

        print("\nTF-IDF results:")
        for fname, score in search_tfidf(q_clean, top_k=5):
            print(f" - {fname}\t{score:.4f}")

        print("\nBM25 results:")
        try:
            for fname, score in bm25_search(q_clean, top_k=5):
                print(f" - {fname}\t{score:.4f}")
        except Exception:
            print("  BM25 not built. Run build_bm25() first.")

        print("\nSemantic results:")
        for fname, dist in semantic_search(q, top_k=5):
            print(f" - {fname}\t{dist:.4f}")

if __name__ == "__main__":
    interactive()
