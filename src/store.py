"""store.py
Simple file-based vector store using numpy + metadata json.
"""
import os
import json
import numpy as np
from typing import List, Dict


class VectorStore:
    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.vectors_file = os.path.join(path, "vectors.npz")
        self.meta_file = os.path.join(path, "metadata.json")
        self._load()


    def _load(self):
        if os.path.exists(self.vectors_file) and os.path.exists(self.meta_file):
            data = np.load(self.vectors_file)
            self.embeddings = data["embeddings"]
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.metadata = []


    def save(self):
        np.savez_compressed(self.vectors_file, embeddings=self.embeddings)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)


    def add(self, embeddings: np.ndarray, metas: List[Dict]):
        if self.embeddings.shape[0] == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.metadata.extend(metas)
        self.save()


    def size(self):
        return len(self.metadata)


    def get_all(self):
        return self.embeddings, self.metadata