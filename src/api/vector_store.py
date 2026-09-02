"""Persistent Vector Store & Semantic Embedding Index Engine.

Indexes paper documents into dense vector spaces, enabling semantic similarity search
across the entire paper corpus and supporting multi-document RAG retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.models.embeddings import DensePaperEmbedder
from src.schemas.paper import PaperDocument

__all__ = ["VectorStore"]


class VectorStore:
    """Persistent local vector database for paper embedding index."""

    def __init__(self, store_dir: str | Path = "data/vector_store") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = DensePaperEmbedder(n_components=64)
        self.paper_ids: list[str] = []
        self.documents: dict[str, PaperDocument] = {}
        self.vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        """Load vector index from disk if present."""
        index_file = self.store_dir / "index.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                self.paper_ids = data.get("paper_ids", [])
                if "vectors" in data and data["vectors"]:
                    self.vectors = np.array(data["vectors"], dtype=np.float32)
            except Exception:
                pass

    def save(self) -> None:
        """Persist vector index to disk."""
        index_file = self.store_dir / "index.json"
        payload = {
            "paper_ids": self.paper_ids,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        index_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_documents(self, documents: Sequence[PaperDocument]) -> None:
        """Index paper documents into the vector store."""
        for doc in documents:
            self.documents[doc.paper_id] = doc

        texts = [
            doc.full_text or doc.text_for(("title", "abstract"))
            for doc in self.documents.values()
        ]
        if not texts:
            return

        self.paper_ids = list(self.documents.keys())
        self.vectors = self.embedder.fit_transform(texts)
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[tuple[PaperDocument, float]]:
        """Search nearest paper documents by semantic vector similarity."""
        if not self.paper_ids or self.vectors is None or len(self.vectors) == 0:
            return []

        query_vec = self.embedder.transform([query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        vec_norms = np.linalg.norm(self.vectors, axis=1)
        vec_norms[vec_norms == 0] = 1e-10

        scores = np.dot(self.vectors, query_vec) / (vec_norms * query_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[PaperDocument, float]] = []
        for idx in top_indices:
            pid = self.paper_ids[idx]
            score = float(scores[idx])
            if pid in self.documents:
                results.append((self.documents[pid], round(score, 4)))

        return results

