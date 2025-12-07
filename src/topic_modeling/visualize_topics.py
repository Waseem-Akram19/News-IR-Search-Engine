# src/topic_modeling/visualize_topics.py

import pickle
from pathlib import Path
from gensim import models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

LDA_DIR = Path("data/lda")
OUTPUT_HTML = LDA_DIR / "lda_viz.html"

def main():
    print("\n📌 Loading LDA model, dictionary, corpus...")

    lda = models.LdaModel.load(str(LDA_DIR / "lda.model"))
    dictionary = pickle.load(open(LDA_DIR / "dictionary.pkl", "rb"))
    corpus = pickle.load(open(LDA_DIR / "corpus.pkl", "rb"))

    print("⏳ Generating pyLDAvis visualization...")

    vis = gensimvis.prepare(
        lda,
        corpus,
        dictionary,
        sort_topics=False
    )

    pyLDAvis.save_html(vis, str(OUTPUT_HTML))

    print("\n✅ Saved LDA visualization to:", OUTPUT_HTML)

if __name__ == "__main__":
    main()
