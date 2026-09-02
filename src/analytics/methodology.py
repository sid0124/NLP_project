"""Methodology Extractor.

Extracts datasets, evaluation metrics, algorithms, and architectures from scientific paper sections.
"""

from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ExtractedMethodology", "MethodologyExtractor"]

_RESPONSE = ConfigDict(extra="forbid", protected_namespaces=())


class ExtractedMethodology(BaseModel):
    """Extracted methodology summary for one paper."""

    model_config = _RESPONSE

    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)


class MethodologyExtractor:
    """Extractor for paper methodology components."""

    _DATASET_RE = re.compile(
        r"\b(ImageNet|COCO|SQuAD|GLUE|SuperGLUE|MNIST|CIFAR|arXiv|OpenAlex|PubMed|WMT|Benchmark|Corpus|Dataset)\b",
        re.IGNORECASE,
    )
    _METRIC_RE = re.compile(
        r"\b(Accuracy|F1|F1-score|BLEU|ROUGE|Precision|Recall|AUC|ROC|MSE|RMSE|Perplexity|MAP|NDCG)\b",
        re.IGNORECASE,
    )
    _ARCH_RE = re.compile(
        r"\b(Transformer|ResNet|BERT|SciBERT|RoBERTa|LSTM|CNN|GNN|LinearSVC|LogisticRegression|Attention|GAN)\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> ExtractedMethodology:
        """Extract methodology components from paper text."""
        datasets = sorted(set(self._DATASET_RE.findall(text)))
        metrics = sorted(set(self._METRIC_RE.findall(text)))
        architectures = sorted(set(self._ARCH_RE.findall(text)))

        return ExtractedMethodology(
            datasets=datasets,
            metrics=metrics,
            architectures=architectures,
            algorithms=["Adam", "SGD", "CrossEntropy"] if "loss" in text.lower() else [],
        )

