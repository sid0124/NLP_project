"""Scientific Transformer embeddings (SciBERT) with an on-disk cache (spec §6).

The single place the project touches Hugging Face transformers, so device
selection, tokenisation, truncation, padding, pooling, and caching stay
consistent everywhere.

Design decisions, and why:

* **Sentences in, vectors out.** Callers pass *sentence units* (see
  :mod:`src.preprocessing.section_parser`), never whole papers — a standard
  Transformer truncates at ~512 tokens, which would silently discard most of a
  long academic paper. Hierarchical processing (Paper -> Sections ->
  Sentences -> embeddings) is how the HAN handles long documents (spec §8).
* **Mean pooling by default.** Mean over non-padded tokens is the standard,
  robust BERT sentence embedding; CLS is available for comparison runs.
* **Disk cache keyed by content hash.** Encoding is the expensive step
  (seconds per hundred sentences on CPU). The cache key hashes the model name,
  pooling mode, and sentence text, so identical text is never re-encoded
  across runs — including repeated inference on the dashboard.
* **Graceful degradation.** Missing weights, no network, and CPU-only boxes
  are all expected states: construction raises a clear
  :class:`EncoderUnavailableError`, and consumers treat that as
  "feature unavailable with a reason", never as a crash.

The legacy :class:`src.models.embeddings.DensePaperEmbedder` (TF-IDF + SVD)
is unchanged and still backs the vector store; this module is the
Transformer path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.config.settings import EncoderConfig
from src.utils.io import resolve_path
from src.utils.logging import get_logger

__all__ = [
    "EncoderUnavailableError",
    "SciBERTEncoder",
    "resolve_device",
]

logger = get_logger(__name__)


class EncoderUnavailableError(RuntimeError):
    """Raised when the transformer encoder cannot be constructed or loaded.

    ``message`` names the prerequisite (package, model weights, or network) so
    the UI can render "unavailable because ..." rather than a stack trace.
    """


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference to hardware that actually exists.

    Args:
        preference: ``"auto"``, ``"cpu"``, or ``"cuda"``. ``"cuda"`` is
            honoured only when a CUDA device is present.

    Returns:
        A torch device string (``"cuda"`` or ``"cpu"``).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EncoderUnavailableError(
            "PyTorch is not installed. Install the ML stack with: "
            "pip install -r requirements-ml.txt"
        ) from exc

    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("transformer | cuda requested but unavailable; falling back to cpu")
        return "cpu"
    if preference == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SciBERTEncoder:
    """Sentence embedding with a scientific Transformer, cached on disk."""

    def __init__(self, config: EncoderConfig, *, cache_dir: str | Path | None = None) -> None:
        """Load the tokenizer and model onto the resolved device.

        Raises:
            EncoderUnavailableError: When torch/transformers are missing or
                the model weights cannot be downloaded or loaded.
        """
        self.config = config
        self.device = resolve_device(config.device)
        self.cache_dir = resolve_path(cache_dir or config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EncoderUnavailableError(
                "The 'transformers' package is not installed. "
                "Install it with: pip install -r requirements-ml.txt"
            ) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
        except Exception as exc:  # noqa: BLE001 - network / missing weights / OOM
            raise EncoderUnavailableError(
                f"Could not load encoder '{config.model_name}': {exc}. "
                "Check network access to huggingface.co, or pre-download the "
                "model weights."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "transformer | encoder=%s device=%s pooling=%s max_seq=%d",
            config.model_name,
            self.device,
            self.config.pooling,
            self.config.max_seq_length,
        )

    # ------------------------------------------------------------------
    # Tokenisation / batching / pooling
    # ------------------------------------------------------------------
    def _tokenize(self, texts: Sequence[str]):
        """Tokenize a batch with truncation and padding to the configured length."""
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

    def _pool(self, outputs, attention_mask) -> np.ndarray:
        """Pool the final hidden state into one vector per input."""
        hidden = outputs.last_hidden_state  # (batch, seq, dim)
        if self.config.pooling == "cls":
            return hidden[:, 0, :].detach().cpu().numpy()

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (batch, seq, 1)
        summed = (hidden * mask).sum(dim=1)  # (batch, dim)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        return (summed / counts).detach().cpu().numpy()

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode one batch on the selected device, in inference mode."""
        import torch

        inputs = self._tokenize(texts)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._pool(outputs, inputs["attention_mask"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        """Encode texts into an ``(n, dim)`` float32 array, using the cache.

        Args:
            texts: Sentences or short passages. Long documents must already
                have been decomposed (spec §8).
            batch_size: Override the configured batch size.

        Returns:
            One vector per input, in input order.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        texts = list(texts)
        results = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        pending: list[int] = []

        for index, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[index] = cached
            else:
                pending.append(index)

        batch = batch_size or self.config.batch_size
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            vectors = self._encode_batch([texts[i] for i in chunk])
            for i, vector in zip(chunk, vectors, strict=True):
                results[i] = vector
                self._cache_put(texts[i], vector)

        return results

    def transform(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        """Alias for :meth:`embed` with sklearn-style naming."""
        return self.embed(texts, **kwargs)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of one output vector (768 for SciBERT-base)."""
        return int(self.model.config.hidden_size)

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        """Stable cache key: model + pooling + text, hashed."""
        material = f"{self.config.model_name}|{self.config.pooling}|{text}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Shard the cache directory so no single folder balloons."""
        return self.cache_dir / key[:2] / f"{key}.npy"

    def _cache_get(self, text: str) -> np.ndarray | None:
        """Return the cached vector, or ``None`` on any miss."""
        try:
            return np.load(self._cache_path(self._cache_key(text)))
        except (OSError, ValueError):
            return None

    def _cache_put(self, text: str, vector: np.ndarray) -> None:
        """Best-effort cache write; a full disk must never break inference."""
        try:
            path = self._cache_path(self._cache_key(text))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector.astype(np.float32))
        except OSError as exc:  # pragma: no cover - disk-full path
            logger.debug("transformer | cache write failed: %s", exc)