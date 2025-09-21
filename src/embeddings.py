"""embeddings.py
Wrapper around sentence-transformers local model.
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingsModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)


    def embed(self, texts):
    # texts: list[str] or single str
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        if single:
            return embeddings[0]
        return np.array(embeddings)