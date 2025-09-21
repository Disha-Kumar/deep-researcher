"""summarizer.py
Simple extractive summarizer: score sentences by TF-IDF and return top sentences.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np




def top_sentences_from_docs(docs_texts, n_sentences=5):
    # docs_texts: list[str] (each element is a chunk)
    # Join and split sentences
    all_text = "\n\n".join(docs_texts)
    sentences = [s.strip() for s in all_text.replace('\n', ' ').split('.') if s.strip()]
    if not sentences:
        return ""
    # TF-IDF over sentences
    vect = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vect.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top_idx = np.argsort(-scores)[:n_sentences]
    selected = [sentences[i] for i in sorted(top_idx)]
    return '. '.join(selected) + ('.' if selected else '')