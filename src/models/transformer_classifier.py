"""Transformer baselines: frozen SciBERT features + a trainable head (spec §5).

Two configured baselines share this class:

* **Baseline 3** — SciBERT embeddings + Logistic Regression (``head="logreg"``).
  Frozen encoder; the sklearn head is fast, CPU-friendly, and gives real
  probabilities.
* **Baseline 4** — SciBERT + a trained classification head (``head="torch"``).
  Frozen encoder, torch head trained with the ``neural_training`` config
  (class weights, early stopping).

Fine-tuning the *encoder* is intentionally out of scope here (spec §9 offers it
as a flag, but on CPU it is not a realistic baseline); the HAN module is where
the interesting trainable structure lives.

The class satisfies :class:`~src.models.base.BaseClassifier`, so the same
training entry point, metrics code, and run store serve it unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from src.models.base import BaseClassifier, ScoreKind
from src.models.transformer_embeddings import (
    EncoderUnavailableError,  # noqa: F401 - re-exported for callers
    SciBERTEncoder,
    resolve_device,  # noqa: F401 - re-exported for callers
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.config.settings import ModelConfig, Settings

__all__ = ["EncoderUnavailableError", "TransformerClassifier", "resolve_device"]

logger = get_logger(__name__)


class TransformerClassifier(BaseClassifier):
    """SciBERT feature extractor + a classification head.

    Attributes:
        encoder: The (frozen) sentence encoder.
        head_kind: ``"logreg"`` (Baseline 3) or ``"torch"`` (Baseline 4).
    """

    def __init__(
        self,
        encoder: SciBERTEncoder,
        *,
        head: str = "logreg",
        model_config: "ModelConfig | None" = None,
        seed: int = 42,
    ) -> None:
        """Wrap an already-constructed encoder; prefer :meth:`from_config`."""
        self.encoder = encoder
        self.head_kind = head
        self.model_config = model_config
        self.seed = seed
        self._classes: list[str] = []
        self._logreg = None
        self._torch_head = None

    @classmethod
    def from_config(cls, settings: "Settings", *, head: str = "logreg") -> "TransformerClassifier":
        """Build from resolved settings (uses ``model.encoder``)."""
        encoder = SciBERTEncoder(settings.model.encoder)
        return cls(
            encoder,
            head=head,
            model_config=settings.model,
            seed=settings.app.project.seed,
        )

    @property
    def classes_(self) -> list[str]:
        """Ordered class vocabulary; index i maps to proba column i."""
        return list(self._classes)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts with the frozen encoder (disk-cached)."""
        return self.encoder.embed(texts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TransformerClassifier":
        """Encode the corpus once, then fit the configured head."""
        from src.utils.seed import set_seed

        set_seed(self.seed)
        features = self.embed_texts(texts)
        unique = sorted(set(labels))
        index_of = {name: i for i, name in enumerate(unique)}
        targets = np.array([index_of[label] for label in labels])

        if self.head_kind == "logreg":
            self._fit_logreg(features, targets)
        elif self.head_kind == "torch":
            self._fit_torch_head(features, targets, n_classes=len(unique))
        else:
            raise ValueError(f"Unknown head {self.head_kind!r}; use 'logreg' or 'torch'")

        self._classes = unique
        logger.info(
            "transformer_classifier | fitted head=%s classes=%d features=%s",
            self.head_kind,
            len(unique),
            features.shape,
        )
        return self

    def _fit_logreg(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Baseline 3 head: multinomial logistic regression over embeddings."""
        from sklearn.linear_model import LogisticRegression

        self._logreg = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=4.0,
        )
        self._logreg.fit(features, targets)

    def _fit_torch_head(
        self, features: np.ndarray, targets: np.ndarray, *, n_classes: int
    ) -> None:
        """Baseline 4 head: a trained torch linear layer with weighted loss."""
        import torch
        from torch import nn

        neural = self.model_config.neural_training if self.model_config else None
        epochs = neural.epochs if neural else 10
        lr = neural.learning_rate if neural else 1e-3
        batch = neural.batch_size if neural else 32
        weight_decay = neural.weight_decay if neural else 0.0
        patience = neural.early_stopping_patience if neural else 5

        # Inverse-frequency class weights unless weighting is disabled.
        counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
        weight_tensor = None
        if neural is None or neural.class_weighting != "none":
            weights = counts.sum() / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            weight_tensor = torch.tensor(weights, dtype=torch.float32)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        head = nn.Linear(x.shape[1], n_classes)
        optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(self.seed)

        best_loss = float("inf")
        since_improve = 0
        for epoch in range(epochs):
            head.train()
            permutation = torch.randperm(x.shape[0], generator=generator)
            epoch_loss = 0.0
            for start in range(0, len(permutation), batch):
                batch_idx = permutation[start : start + batch]
                optimiser.zero_grad()
                loss = loss_fn(head(x[batch_idx]), y[batch_idx])
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss) * len(batch_idx)
            epoch_loss /= len(permutation)
            if epoch_loss < best_loss - 1e-5:
                best_loss, since_improve = epoch_loss, 0
            else:
                since_improve += 1
                if patience and since_improve >= patience:
                    logger.info(
                        "transformer_classifier | early stop at epoch %d (loss %.4f)",
                        epoch,
                        best_loss,
                    )
                    break

        head.eval()
        self._torch_head = head

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict class labels for each text."""
        proba = self.predict_proba(texts)
        return [self._classes[i] for i in proba.argmax(axis=1)]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-class probabilities from the fitted head.

        Raises:
            RuntimeError: When called before :meth:`fit`.
        """
        if not self._classes:
            raise RuntimeError("TransformerClassifier must be fitted before predicting.")
        features = self.embed_texts(texts)

        if self.head_kind == "logreg":
            return np.asarray(self._logreg.predict_proba(features))

        import torch

        with torch.no_grad():
            logits = self._torch_head(torch.tensor(features, dtype=torch.float32))
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def scores_kind(self) -> ScoreKind:
        """Both heads produce probability-shaped outputs."""
        return "probability"

    def predict_top_k(self, texts: Sequence[str], k: int = 3) -> list[list[dict]]:
        """Return the top-k ``{label, score}`` entries per text."""
        proba = self.predict_proba(texts)
        results: list[list[dict]] = []
        for row in proba:
            ranked = sorted(
                range(len(self._classes)), key=lambda i: float(row[i]), reverse=True
            )
            results.append(
                [{"label": self._classes[i], "score": float(row[i])} for i in ranked[:k]]
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        """Persist head + class vocabulary (the encoder reloads from config).

        The torch head is stored as a state_dict inside the joblib bundle, so
        the artifact is self-contained without pickling the whole encoder.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "head_kind": self.head_kind,
            "classes": self._classes,
            "seed": self.seed,
            "logreg": self._logreg,
            "torch_state": (
                self._torch_head.state_dict() if self._torch_head is not None else None
            ),
            "torch_shape": (
                list(self._torch_head.weight.shape) if self._torch_head is not None else None
            ),
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path, *, settings: "Settings | None" = None) -> "TransformerClassifier":
        """Load a model written by :meth:`save`.

        Raises:
            EncoderUnavailableError: When the encoder cannot be constructed.
        """
        payload = joblib.load(path)
        if settings is None:
            from src.config.settings import load_settings

            settings = load_settings()
        instance = cls(
            SciBERTEncoder(settings.model.encoder),
            head=payload["head_kind"],
            model_config=settings.model,
            seed=payload["seed"],
        )
        instance._classes = payload["classes"]
        instance._logreg = payload["logreg"]
        if payload.get("torch_state") is not None:
            import torch
            from torch import nn

            in_dim, n_classes = payload["torch_shape"]
            head = nn.Linear(int(in_dim), int(n_classes))
            head.load_state_dict(payload["torch_state"])
            head.eval()
            instance._torch_head = head
        return instance