# src/evaluation/metrics.py
import numpy as np

def precision_at_k(relevant_set:set, retrieved_list:list, k:int):
    if k == 0: return 0.0
    retrieved_k = retrieved_list[:k]
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / k

def recall_at_k(relevant_set:set, retrieved_list:list, k:int):
    retrieved_k = retrieved_list[:k]
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

def average_precision(relevant_set:set, retrieved_list:list):
    if not relevant_set: return 0.0
    score = 0.0
    hits = 0
    for i, doc in enumerate(retrieved_list, start=1):
        if doc in relevant_set:
            hits += 1
            score += hits / i
    return score / len(relevant_set)

def mean_average_precision(list_qrels, list_retrieved):
    ap_list = [average_precision(qrel, ret) for qrel, ret in zip(list_qrels, list_retrieved)]
    return float(np.mean(ap_list))

def ndcg_at_k(relevant_set:set, retrieved_list:list, k:int):
    dcg = 0.0
    for i, doc in enumerate(retrieved_list[:k], start=1):
        if doc in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    ideal_len = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0
