"""Common classifier interface shared by every model in the project (spec §5).

Four model families sit behind one contract so training, evaluation, and the
serving layer treat them uniformly:

* ``TFIDFClassifier``      — TF-IDF + Logistic Regression  (Baseline 1)
* ``TFIDFSVMClassifier``   — TF-IDF + Linear SVM           (Baseline 2)
* ``TransformerClassifier``— SciBERT features + linear head (Baselines 3/4)
* ``HANClassifier``        — SciBERT + Hierarchical Attention (final model)

The interface is deliberately the *same shape* as a fitted scikit-learn
pipeline — ``fit``, ``predict``, ``predict_proba``, ``classes_`` — because the
whole evaluation layer, the run manifest, and the serving code already speak
that language. A model that satisfies this protocol can be swapped into any of
those paths without an adapter.

``predict_proba`` may be unavailable (the SVM case): the ``scores``/``kind``
pair from :func:`src.models.baselines.prediction_scores` is the contract the
rest of the code consumes, and ``confidence_kind`` travels with every served
response so a margin is never presented as a probability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["BaseClassifier", "ScoreKind", "ScoresProtocol"]


#: What a model's scores mean. Mirrors ``src.models.baselines.ScoreKind``.
ScoreKind = str


@runtime_checkable
class ScoresProtocol(Protocol):
    """Minimal read surface a served run needs from any model."""

    @property
    def classes_(self) -> list[str]: ...


class BaseClassifier(ABC):
    """Abstract base for every trainable classifier in the project.

    Subclasses implement the sklearn-pipeline-shaped surface. Anything that
    implements this protocol can be trained by the same training entry point,
    evaluated by the same metrics code, and served by the same run store.
    """

    @property
    @abstractmethod
    def classes_(self) -> list[str]:
        """Ordered class vocabulary; index i is the class of proba column i."""

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "BaseClassifier":
        """Fit on the training split. Returns ``self``."""

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> list[str]:
        """Return the predicted class label for each text."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n_samples, n_classes)`` scores.

        Raises:
            NotImplementedError: When the underlying model exposes no
                probability-like output (the SVM margin case).
        """

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Persist the fitted model. Returns the written path."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseClassifier":
        """Load a model previously written by :meth:`save`."""

    # -- shared conveniences -------------------------------------------------
    def predict_top_k(self, texts: Sequence[str], k: int = 3) -> list[list[dict[str, Any]]]:
        """Return the top-k ``{label, score}`` entries per text."""
        proba = self.predict_proba(texts)
        classes = self.classes_
        results: list[list[dict[str, Any]]] = []
        for row in proba:
            ranked = sorted(
                range(len(classes)), key=lambda i: float(row[i]), reverse=True
            )[:k]
            results.append(
                [{"label": classes[i], "score": float(row[i])} for i in ranked]
            )
        return results

    def scores_kind(self) -> ScoreKind:
        """What :meth:`predict_proba` returns, for honest UI labelling."""
        try:
            self.predict_proba(["probe"])
        except NotImplementedError:
            return "decision"
        return "probability"