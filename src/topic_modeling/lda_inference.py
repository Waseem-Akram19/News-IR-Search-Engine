# src/topic_modeling/lda_inference.py
from pathlib import Path
import pickle
from gensim import models
from math import sqrt

LDA_DIR = Path("data/lda")

def load_lda():
    lda = models.LdaModel.load(str(LDA_DIR / "lda.model"))
    dictionary = pickle.load(open(LDA_DIR / "dictionary.pkl", "rb"))
    corpus = pickle.load(open(LDA_DIR / "corpus.pkl", "rb"))
    filenames = pickle.load(open(LDA_DIR / "filenames.pkl", "rb"))
    return lda, dictionary, corpus, filenames

def doc_topic_vector(lda, bow, num_topics=None):
    # returns dense list of topic probabilities
    if num_topics is None:
        num_topics = lda.num_topics
    dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
    return [dist.get(i, 0.0) for i in range(num_topics)]

def query_by_topic_similarity(query_tokens, top_k=10):
    lda, dictionary, corpus, filenames = load_lda()
    qbow = dictionary.doc2bow(query_tokens)
    qvec = doc_topic_vector(lda, qbow)
    scores = []
    for i, doc_bow in enumerate(corpus):
        dvec = doc_topic_vector(lda, doc_bow)
        # cosine similarity
        num = sum(a*b for a,b in zip(qvec, dvec))
        denom = sqrt(sum(a*a for a in qvec)) * sqrt(sum(b*b for b in dvec))
        score = num / denom if denom != 0 else 0.0
        scores.append((filenames[i], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
