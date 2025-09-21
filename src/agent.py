"""agent.py
CLI to query the local store, retrieve relevant chunks, synthesize, and export a Markdown report.
Run: python -m src.agent --query "..." --top_k 5
"""
import argparse
import time
import json
from pathlib import Path
from .embeddings import EmbeddingsModel
from .store import VectorStore
from .retriever import retrieve
from .summarizer import top_sentences_from_docs




def make_report(question: str, hits, summary_text: str, out_path: Path):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    md = []
    md.append(f"# Researcher Report\n\n**Question:** {question}\n\n**Generated:** {ts}\n\n---\n")
    md.append("## Summary\n")
    md.append(summary_text + "\n\n")
    md.append("---\n## Evidence (top hits)\n")
    for score, meta in hits:
        md.append(f"- **score**: {score:.4f} — **source**: `{meta.get('source')}` — chunk {meta.get('chunk_index')}\n\n")
        md.append(f"```\n{meta.get('text')[:1000]}\n```\n")