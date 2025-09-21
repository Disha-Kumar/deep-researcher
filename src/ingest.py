"""ingest.py
Ingest documents from a folder; split into chunks, embed, and store.
Run as: python -m src.ingest --data_dir ./data --store_dir ./store
"""
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


from .embeddings import EmbeddingsModel
from .store import VectorStore

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")




def read_pdf(path: Path) -> str:
    text = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def main(data_dir: str, store_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    emb = EmbeddingsModel(model_name)
    store = VectorStore(store_dir)


    files = list(Path(data_dir).glob("**/*"))
    docs = [f for f in files if f.suffix.lower() in [".txt", ".pdf"]]
    metas = []
    all_embeddings = []


    for f in tqdm(docs, desc="documents"):
        try:
            if f.suffix.lower() == ".txt":
                text = read_txt(f)
            else:
                text = read_pdf(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue


        if not text.strip():
            continue


    chunks = chunk_text(text, chunk_size=400, overlap=40)
    embeddings = emb.embed(chunks)
    for i, chunk in enumerate(chunks):
        metas.append({
            "source": str(f),
            "chunk_index": i,
            "text": chunk[:2000]
        })
    all_embeddings.append(embeddings)


    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        store.add(all_embeddings, metas)
        print(f"Stored {store.size()} chunks in {store_dir}")
    else:
        print("No new data to ingest.")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--store_dir", default="./store")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args.data_dir, args.store_dir, args.model)