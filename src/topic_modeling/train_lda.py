# src/topic_modeling/train_lda.py
from pathlib import Path
from gensim import corpora, models
import pickle

PROCESSED_DIR = Path("data/processed")
LDA_DIR = Path("data/lda")
LDA_DIR.mkdir(parents=True, exist_ok=True)

def build_texts_and_train(num_topics=10, passes=10):
    texts = []
    filenames = []
    for p in sorted(PROCESSED_DIR.glob("*.txt"), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name):
        toks = p.read_text(encoding="utf-8").split()
        texts.append(toks)
        filenames.append(p.name)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    lda.save(str(LDA_DIR / "lda.model"))
    pickle.dump(dictionary, open(LDA_DIR / "dictionary.pkl", "wb"))
    pickle.dump(corpus, open(LDA_DIR / "corpus.pkl", "wb"))
    pickle.dump(filenames, open(LDA_DIR / "filenames.pkl", "wb"))
    print("LDA trained and saved to", LDA_DIR)

if __name__ == "__main__":
    build_texts_and_train()
