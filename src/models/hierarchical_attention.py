"""Hierarchical Attention Network in PyTorch (spec §7).

The architecture, level by level::

    Sentence embeddings (SciBERT, frozen)
        -> Level 1: sentence encoder   (BiGRU over a section's sentences)
        -> Level 2: sentence attention (one weight per sentence, per section)
        -> section representation
        -> Level 3: section encoder    (BiGRU over the paper's sections)
        -> Level 4: section attention  (one weight per section)
        -> Level 5: document representation
        -> dropout -> Level 6: linear classifier -> softmax

Two properties matter for the project's claims:

* **Real attention.** Weights come from additive (Bahdanau-style) scoring
  inside the trained network — ``softmax(v^T tanh(W h))`` over encoder states —
  and are returned by the network at inference. Nothing here is heuristic; the
  UI's "model attention / evidence" panel renders exactly these numbers.
* **Honest scope.** Attention weights are an evidence *visualisation*, not a
  causal explanation (master spec §14).

Batching layout: a batch of papers is a padded 4-D tensor
``(batch, sections, sentences, dim)`` plus validity masks, built by
:func:`collate_documents`. Variable section/sentence counts are handled by
masking, never by truncating the hierarchy away.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.utils.logging import get_logger

__all__ = [
    "AttentionOutput",
    "DocumentBatch",
    "HANAttentionResult",
    "HierarchicalAttentionNetwork",
    "collate_documents",
]

logger = get_logger(__name__)


def _torch():
    """Import torch lazily with a clear error when missing."""
    try:
        import torch
        from torch import nn

        return torch, nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for the Hierarchical Attention Network. "
            "Install it with: pip install -r requirements-ml.txt"
        ) from exc


class DocumentBatch:
    """A padded batch of documents as sentence-embedding tensors.

    Attributes:
        embeddings: ``(batch, max_sections, max_sentences, dim)`` float tensor.
        sentence_mask: ``(batch, max_sections, max_sentences)`` True = real.
        section_mask: ``(batch, max_sections)`` True = section exists.
    """


def collate_documents(
    documents: Sequence[Sequence[Sequence[np.ndarray]]],
) -> DocumentBatch:
    """Pad a batch of documents into tensors for the HAN.

    Args:
        documents: One document per paper; a document is a sequence of
            sections; a section is a sequence of sentence embeddings.

    Returns:
        A :class:`DocumentBatch` with CPU tensors (caller moves to device).
    """
    torch, _ = _torch()

    max_sections = max(len(doc) for doc in documents) if documents else 0
    max_sentences = max((len(sec) for doc in documents for sec in doc), default=0)
    dim = next((len(v) for doc in documents for sec in doc for v in sec), 0)

    embeddings = torch.zeros(
        len(documents), max_sections, max_sentences, dim, dtype=torch.float32
    )
    sentence_mask = torch.zeros(
        len(documents), max_sections, max_sentences, dtype=torch.bool
    )
    section_mask = torch.zeros(len(documents), max_sections, dtype=torch.bool)

    for b, doc in enumerate(documents):
        for s, section in enumerate(doc):
            section_mask[b, s] = True
            for t, vector in enumerate(section):
                embeddings[b, s, t] = torch.as_tensor(np.asarray(vector), dtype=torch.float32)
                sentence_mask[b, s, t] = True

    batch = DocumentBatch()
    batch.embeddings = embeddings
    batch.sentence_mask = sentence_mask
    batch.section_mask = section_mask
    return batch


class HANAttentionResult:
    """Inference output: predictions plus the network's own attention weights.

    Attributes:
        probabilities: ``(n_documents, n_classes)`` softmax output.
        sentence_weights: per document, per section, one weight per sentence
            (lists shaped like the input document, unpadded).
        section_weights: per document, one weight per section (unpadded).
    """

    def __init__(
        self,
        probabilities: np.ndarray,
        sentence_weights: list[list[list[float]]],
        section_weights: list[list[float]],
    ) -> None:
        self.probabilities = probabilities
        self.sentence_weights = sentence_weights
        self.section_weights = section_weights


#: Alias for type annotations at call sites.
AttentionOutput = HANAttentionResult


def build_han_network(input_dim, sentence_hidden, section_hidden, n_classes, dropout):
    """Construct the HAN as a torch module (kept at module level for testing).

    Returns the un-initialised :class:`torch.nn.Module`.
    """
    torch, nn = _torch()

    class _AdditiveAttention(nn.Module):
        """Additive attention producing real, differentiable weights."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.project = nn.Linear(dim, dim, bias=False)
            self.score = nn.Linear(dim, 1, bias=False)

        def forward(self, states, mask):
            """Weight encoder states.

            Args:
                states: ``(batch, items, dim)`` encoder outputs.
                mask: ``(batch, items)`` bool; True where the item is real.

            Returns:
                ``(context, weights)`` with shapes ``(batch, dim)`` and
                ``(batch, items)``.
            """
            energies = self.score(torch.tanh(self.project(states))).squeeze(-1)
            energies = energies.masked_fill(~mask, torch.finfo(energies.dtype).min)
            weights = torch.softmax(energies, dim=-1)
            context = (weights.unsqueeze(-1) * states).sum(dim=1)
            return context, weights

    class _HAN(nn.Module):
        """The six-level Hierarchical Attention Network (spec §7)."""

        def __init__(self) -> None:
            super().__init__()
            # Level 1: sentence encoder — BiGRU over each section's sentences.
            self.sentence_encoder = nn.GRU(
                input_dim, sentence_hidden // 2, batch_first=True, bidirectional=True
            )
            # Level 2: sentence attention -> section representation.
            self.sentence_attention = _AdditiveAttention(sentence_hidden)
            # Level 3: section encoder — BiGRU over the paper's sections.
            self.section_encoder = nn.GRU(
                sentence_hidden, section_hidden // 2, batch_first=True, bidirectional=True
            )
            # Level 4: section attention -> document representation.
            self.section_attention = _AdditiveAttention(section_hidden)
            # Levels 5-6: document representation -> classifier.
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(section_hidden, n_classes)

        def _encode_sections(self, embeddings, sentence_mask):
            """Levels 1-2 for every section at once.

            Returns:
                ``(section_states, sentence_weights)`` with shapes
                ``(batch, max_sections, hidden)`` and
                ``(batch, max_sections, max_sentences)``.
            """
            batch, max_sections, max_sentences, dim = embeddings.shape
            flat = embeddings.reshape(batch * max_sections, max_sentences, dim)
            flat_mask = sentence_mask.reshape(batch * max_sections, max_sentences)

            # Empty sections would make softmax over all-masked rows NaN.
            has_sentences = flat_mask.any(dim=1)
            # Guarantee at least one valid slot per row for the GRU.
            safe_mask = flat_mask.clone()
            safe_mask[~has_sentences, 0] = True

            encoded, _ = self.sentence_encoder(flat)
            contexts, weights = self.sentence_attention(encoded, safe_mask)

            hidden = contexts.shape[-1]
            contexts = contexts.reshape(batch, max_sections, hidden)
            weights = weights.reshape(batch, max_sections, max_sentences)
            return contexts, weights

        def forward(self, batch: DocumentBatch, return_attention: bool = False):
            """Run the full hierarchy.

            Args:
                batch: A :class:`DocumentBatch` from :func:`collate_documents`.
                return_attention: Also return the attention weights.

            Returns:
                ``logits`` of shape ``(batch, n_classes)``, or
                ``(logits, sentence_weights, section_weights)`` when
                ``return_attention`` is set.
            """
            batch_size, max_sections = batch.embeddings.shape[:2]

            section_states, sentence_weights = self._encode_sections(
                batch.embeddings, batch.sentence_mask
            )

            # Mask out padded sections for the section encoder/attention.
            safe_section_mask = batch.section_mask.clone()
            empty = ~safe_section_mask.any(dim=1)
            safe_section_mask[empty, 0] = True

            encoded_sections, _ = self.section_encoder(section_states)
            doc_representation, section_weights = self.section_attention(
                encoded_sections, safe_section_mask
            )

            logits = self.classifier(self.dropout(doc_representation))

            if not return_attention:
                return logits

            # Unpad the weights to per-document lists.
            sentence_out: list[list[list[float]]] = []
            section_out: list[list[float]] = []
            for b in range(batch_size):
                doc_sentences: list[list[float]] = []
                for s in range(max_sections):
                    if not batch.section_mask[b, s]:
                        continue
                    count = int(batch.sentence_mask[b, s].sum())
                    doc_sentences.append(
                        [float(w) for w in sentence_weights[b, s, :count]]
                    )
                sentence_out.append(doc_sentences)
                section_out.append(
                    [float(w) for w in section_weights[b, : int(batch.section_mask[b].sum())]]
                )
            return logits, sentence_out, section_out

    return _HAN()