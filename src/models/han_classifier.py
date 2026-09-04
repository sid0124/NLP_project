"""SciBERT + Hierarchical Attention Network classifier (spec §5/§7/§37).

The final model. Pipeline::

    text -> sections -> sentences  (section_parser + longdoc bounds)
         -> SciBERT embeddings      (frozen, disk-cached)
         -> HAN                     (sentence + section attention)
         -> softmax over domains

Inference returns the network's **real** attention weights alongside the
prediction, which is what the dashboard's attention panel and the
``GET /papers/{id}/attention`` endpoint render. The weights are evidence
visualisation, never a causal explanation (master spec §14).

The class satisfies :class:`~src.models.base.BaseClassifier`; the extra
:func:`predict_with_attention` is where the hierarchical explainability lives.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from src.models.base import BaseClassifier, ScoreKind
from src.models.hierarchical_attention import (
    HANAttentionResult,
    build_han_network,
    collate_documents,
)
from src.models.transformer_embeddings import (
    EncoderUnavailableError,  # noqa: F401 - re-exported
    SciBERTEncoder,
)
from src.preprocessing.section_parser import sections_to_sentence_units
from src.preprocessing.sections import parse_text_into_sections
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.config.settings import ModelConfig, Settings

__all__ = ["EncoderUnavailableError", "HANAttentionResult", "HANClassifier"]

logger = get_logger(__name__)


class HANClassifier(BaseClassifier):
    """SciBERT features + Hierarchical Attention Network head."""

    def __init__(
        self,
        encoder: SciBERTEncoder,
        *,
        model_config: "ModelConfig",
        seed: int = 42,
    ) -> None:
        """Wrap a constructed encoder; prefer :meth:`from_config`."""
        self.encoder = encoder
        self.model_config = model_config
        self.seed = seed
        self._classes: list[str] = []
        self._network = None

    @classmethod
    def from_config(cls, settings: "Settings") -> "HANClassifier":
        """Build from resolved settings (encoder + HAN + longdoc config)."""
        encoder = SciBERTEncoder(settings.model.encoder)
        return cls(encoder, model_config=settings.model, seed=settings.app.project.seed)

    @property
    def classes_(self) -> list[str]:
        """Ordered class vocabulary; index i maps to proba column i."""
        return list(self._classes)

    # ------------------------------------------------------------------
    # Document structuring (spec §8)
    # ------------------------------------------------------------------
    def _document_to_sentence_matrix(self, text: str) -> list[list[np.ndarray]]:
        """Turn one paper's text into per-section sentence embeddings.

        Applies the ``longdoc`` bounds (max sections, max sentences per
        section) so a 20-page paper contributes the same tensor rank as a
        4-page one — and never feeds a long paper as one Transformer sequence.
        """
        longdoc = self.model_config.longdoc
        sections = parse_text_into_sections(text)
        units = sections_to_sentence_units(
            sections,
            max_sentences_per_section=longdoc.max_sentences_per_section,
        )

        ordered_sections = sorted(
            {(u.section_order, u.section_name, u.canonical_name) for u in units},
            key=lambda t: t[0],
        )[: longdoc.max_sections]

        by_section: dict[int, list[str]] = {}
        for unit in units:
            by_section.setdefault(unit.section_order, []).append(unit.text)

        all_sentences: list[str] = []
        spans: list[tuple[int, int]] = []
        for order, _name, _canonical in ordered_sections:
            sentences = by_section.get(order, [])
            start = len(all_sentences)
            all_sentences.extend(sentences)
            spans.append((start, len(all_sentences)))

        vectors = self.encoder.embed(all_sentences)
        matrix = [list(vectors[start:end]) for start, end in spans]
        # A document with zero sentences still needs one empty section so the
        # collator sees a valid entry rather than dropping the paper.
        return matrix or [[]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[list[np.ndarray]]]:
        """Structure + encode every document into section/sentence matrices."""
        return [self._document_to_sentence_matrix(text) for text in texts]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "HANClassifier":
        """Encode documents hierarchically, then train the HAN head.

        The encoder stays frozen; only the HAN (attention + classifier) trains,
        which is what makes this tractable on CPU.
        """
        import torch
        from src.utils.seed import set_seed

        set_seed(self.seed)
        documents = self.embed_documents(texts)
        unique = sorted(set(labels))
        index_of = {name: i for i, name in enumerate(unique)}
        targets = torch.tensor([index_of[label] for label in labels], dtype=torch.long)

        config = self.model_config.han
        neural = self.model_config.neural_training
        network = build_han_network(
            input_dim=self.encoder.embedding_dim,
            sentence_hidden=config.sentence_hidden_size,
            section_hidden=config.section_hidden_size,
            n_classes=len(unique),
            dropout=config.dropout,
        )

        # Inverse-frequency class weights per the configured strategy.
        counts = np.bincount(targets.numpy(), minlength=len(unique)).astype(np.float64)
        weight_tensor = None
        if neural.class_weighting != "none":
            weights = counts.sum() / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        optimiser = torch.optim.AdamW(
            network.parameters(),
            lr=neural.learning_rate,
            weight_decay=neural.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed)

        best_loss = float("inf")
        since_improve = 0
        for epoch in range(neural.epochs):
            network.train()
            order = torch.randperm(len(documents), generator=generator).tolist()
            epoch_loss = 0.0
            for start in range(0, len(order), neural.batch_size):
                chunk_indices = order[start : start + neural.batch_size]
                chunk = [documents[i] for i in chunk_indices]
                batch = collate_documents(chunk)
                optimiser.zero_grad()
                logits = network(batch)
                loss = loss_fn(logits, targets[chunk_indices])
                loss.backward()
                if neural.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), neural.gradient_clip_norm
                    )
                optimiser.step()
                epoch_loss += float(loss) * len(chunk)
            epoch_loss /= len(order)

            if epoch_loss < best_loss - 1e-5:
                best_loss, since_improve = epoch_loss, 0
            else:
                since_improve += 1
                if (
                    neural.early_stopping_patience
                    and since_improve >= neural.early_stopping_patience
                ):
                    logger.info(
                        "han | early stop at epoch %d (loss %.4f)", epoch, best_loss
                    )
                    break

        network.eval()
        self._network = network
        self._classes = unique
        logger.info(
            "han | fitted classes=%d docs=%d best_loss=%.4f",
            len(unique),
            len(documents),
            best_loss,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict class labels for each text."""
        result = self.predict_with_attention(texts)
        return [self._classes[i] for i in result.probabilities.argmax(axis=1)]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-class probabilities from the HAN."""
        return self.predict_with_attention(texts).probabilities

    def scores_kind(self) -> ScoreKind:
        """The HAN ends in softmax, so outputs are probabilities."""
        return "probability"

    def predict_with_attention(self, texts: Sequence[str]) -> HANAttentionResult:
        """Run the HAN and return predictions **plus real attention weights**.

        The returned weights are the trained network's own additive-attention
        outputs — one per sentence (per section) and one per section — which is
        what the dashboard's attention panel visualises. They are model
        evidence, not a causal explanation (master spec §14).

        Raises:
            RuntimeError: When called before :meth:`fit`.
        """
        import torch

        if self._network is None or not self._classes:
            raise RuntimeError("HANClassifier must be fitted before predicting.")

        documents = self.embed_documents(texts)
        results = HANAttentionResult(
            probabilities=np.zeros((len(documents), len(self._classes))),
            sentence_weights=[],
            section_weights=[],
        )

        batch_size = self.model_config.neural_training.batch_size
        for start in range(0, len(documents), batch_size):
            chunk = documents[start : start + batch_size]
            batch = collate_documents(chunk)
            with torch.no_grad():
                logits, sentence_w, section_w = self._network(batch, return_attention=True)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            results.probabilities[start : start + len(chunk)] = probs
            results.sentence_weights.extend(sentence_w)
            results.section_weights.extend(section_w)

        return results

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
        """Persist the HAN state dict + class vocabulary (encoder reloads from config)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": self._classes,
            "seed": self.seed,
            "state_dict": (
                self._network.state_dict() if self._network is not None else None
            ),
            "architecture": {
                "input_dim": self.encoder.embedding_dim,
                "sentence_hidden": self.model_config.han.sentence_hidden_size,
                "section_hidden": self.model_config.han.section_hidden_size,
                "n_classes": len(self._classes),
                "dropout": self.model_config.han.dropout,
            },
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path, *, settings: "Settings | None" = None) -> "HANClassifier":
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
            model_config=settings.model,
            seed=payload["seed"],
        )
        instance._classes = payload["classes"]
        if payload.get("state_dict") is not None:
            import torch

            arch = payload["architecture"]
            network = build_han_network(
                input_dim=arch["input_dim"],
                sentence_hidden=arch["sentence_hidden"],
                section_hidden=arch["section_hidden"],
                n_classes=arch["n_classes"],
                dropout=arch["dropout"],
            )
            network.load_state_dict(payload["state_dict"])
            network.eval()
            instance._network = network
        return instance