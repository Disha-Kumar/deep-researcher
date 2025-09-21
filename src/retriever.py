"""retriever.py
Brute-force nearest neighbors using numpy cosine similarity (fast enough at small scale).
"""
import numpy as np
from typing import List, Tuple




def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (d,), b: (n, d) -> returns (n,)
    a = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = b_norm.dot(a)
    return sims




def retrieve(query_emb: np.ndarray, embeddings: np.ndarray, metadata: List[dict], top_k: int = 5):
    sims = cosine_sim(query_emb, embeddings)
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append((float(sims[i]), metadata[i]))
    return results