"""Dense vector embedding extraction for semantic paper similarity and dense retrieval.

Provides feature representation capabilities using n-gram tfidf embeddings, TruncatedSVD / SVD dense projections,
and optional transformer embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["DensePaperEmbedder"]


class DensePaperEmbedder:
    """Dense vector embedding extractor for text documents."""

    def __init__(self, n_components: int = 64, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, texts: Sequence[str]) -> DensePaperEmbedder:
        """Fit the vectorizer and dense SVD projection on a text corpus."""
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, max(1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(tfidf)
        self.is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform texts into dense vector embeddings."""
        if not self.is_fitted:
            self.fit(texts)
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit and transform texts in a single pass."""
        self.fit(texts)
        return self.transform(texts)

    def cosine_similarity(self, query_text: str, corpus_texts: Sequence[str]) -> np.ndarray:
        """Compute cosine similarities between a query text and a corpus."""
        if not self.is_fitted:
            all_texts = [query_text, *list(corpus_texts)]
            self.fit(all_texts)

        query_vec = self.transform([query_text])[0]
        corpus_vecs = self.transform(corpus_texts)

        query_norm = np.linalg.norm(query_vec)
        corpus_norms = np.linalg.norm(corpus_vecs, axis=1)

        if query_norm == 0:
            return np.zeros(len(corpus_texts))

        denom = corpus_norms * query_norm
        denom[denom == 0] = 1e-10

        dots = np.dot(corpus_vecs, query_vec)
        return dots / denom

