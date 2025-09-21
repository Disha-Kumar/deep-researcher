# DeepResearcher — Local Embeddings Researcher

A lightweight, **fully local** Python research agent that:

* Ingests local documents (TXT, PDF)
* Generates embeddings locally with `sentence-transformers`
* Stores embeddings and metadata on disk
* Retrieves relevant passages using cosine similarity
* Synthesizes answers with a local extractive summarizer (TF–IDF based)
* Exports results to Markdown (optional PDF conversion)

This project is simple, fully deployable, and **does not rely on external APIs**.

---

## 📂 Folder structure

```
deep-researcher/
├─ README.md
├─ requirements.txt
├─ data/                # input documents (.txt or .pdf)
│  └─ example.txt
├─ store/               # generated embeddings + metadata
├─ outputs/             # generated query results (Markdown/PDF)
├─ src/
│  ├─ __init__.py
│  ├─ embeddings.py     # local embedding model wrapper
│  ├─ ingest.py         # document ingestion pipeline
│  ├─ store.py          # vector + metadata storage
│  ├─ retriever.py      # similarity search
│  ├─ summarizer.py     # TF-IDF summarizer
│  └─ agent.py          # CLI query agent
└─ .gitignore
```

---

## ⚡ Quickstart (VSCode)

1. Clone or download this repo, then open it in VSCode.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # mac/linux
   source .venv/bin/activate
   # windows (powershell)
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Add documents to the `data/` folder.
5. Ingest documents into the store:

   ```bash
   python -m src.ingest --data_dir ./data --store_dir ./store
   ```
6. Run a query:

   ```bash
   python -m src.agent --query "What are embeddings used for?" --top_k 5
   ```
7. View the generated report in `outputs/`.

---

## 🛠 Dependencies

Listed in `requirements.txt`:

```
sentence-transformers>=2.2.2
numpy>=1.24
scikit-learn>=1.2
PyPDF2>=3.0.0
tqdm>=4.65
```

---

## 🚀 Features

* Local embeddings with `sentence-transformers`
* Brute-force cosine similarity retrieval (can be upgraded to FAISS)
* TF-IDF based summarization of retrieved chunks
* Markdown report export with evidence snippets

---

## 📖 Example

1. Place `example.txt` in `data/`.
2. Run:

   ```bash
   python -m src.ingest --data_dir ./data --store_dir ./store
   python -m src.agent --query "Summarize the main ideas" --top_k 5
   ```
3. Open `outputs/result_*.md` in VSCode.

---

## 🔮 Next steps (optional improvements)

* Swap brute-force search with FAISS for scalability.
* Integrate a local LLM (e.g., Llama.cpp) for more fluent synthesis.
* Enhance PDF parsing (layout-aware extraction).
* Build an interactive web UI (Streamlit/FastAPI).
* Add Markdown → PDF export (`pandoc`).
