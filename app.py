import streamlit as st
from pathlib import Path
import time
from src.embeddings import EmbeddingsModel
from src.store import VectorStore
from src.retriever import retrieve
from src.summarizer import top_sentences_from_docs

st.title("🔍 DeepResearcher — Local Research Agent")

query = st.text_input("Enter your research question:")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Run Query") and query:
    emb = EmbeddingsModel()
    store = VectorStore("./store")
    if store.size() == 0:
        st.error("⚠️ No data found! Run ingestion first.")
    else:
        query_emb = emb.embed(query)
        embeddings, metadata = store.get_all()
        hits = retrieve(query_emb, embeddings, metadata, top_k=top_k)
        docs = [m["text"] for _, m in hits]
        summary = top_sentences_from_docs(docs, n_sentences=6)

        st.subheader("📑 Summary")
        st.write(summary)

        st.subheader("📂 Evidence")
        for score, meta in hits:
            with st.expander(f"{meta['source']} (score={score:.4f})"):
                st.write(meta["text"][:1000])
