"""Classical baseline construction, driven entirely by configuration.

Every baseline in ``configs/model.yaml`` names a vectorizer ``type`` and a
classifier ``type`` plus a ``params`` mapping. This module is the registry that
turns those strings into scikit-learn classes and assembles the pair into a
:class:`~sklearn.pipeline.Pipeline`. **No hyper-parameter is written here**
(master spec §32): adding a baseline is a config entry plus, for a genuinely new
estimator, one registry line.

The pipeline *is* the leakage guarantee (master spec §9). Because the vectorizer
is a pipeline step rather than a separately fitted object,
``pipeline.fit(train_texts, train_labels)`` has no code path by which it could
observe validation or test text. Predicting on another split reuses the
vocabulary and IDF weights learned from training alone.

Two capability differences between the configured classifiers matter downstream
and are exposed here rather than rediscovered by the evaluation layer:

* :class:`~sklearn.linear_model.LogisticRegression` provides ``predict_proba``.
* :class:`~sklearn.svm.LinearSVC` provides only ``decision_function``, whose
  output is an uncalibrated signed distance from the hyperplane — not a
  probability. :func:`prediction_scores` reports which of the two it returned so
  nothing downstream can silently print a margin under a "confidence" heading.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config.settings import ClassifierConfig, ModelConfig, VectorizerConfig
from src.utils.logging import get_logger

__all__ = [
    "CLASSIFIER_STEP",
    "VECTORIZER_STEP",
    "ScoreKind",
    "UnknownEstimatorError",
    "build_baseline",
    "build_classifier",
    "build_vectorizer",
    "prediction_scores",
    "resolved_params",
    "supports_probabilities",
]

logger = get_logger(__name__)

#: Pipeline step names. Fixed strings so evaluation and persistence can reach a
#: fitted sub-estimator by name without positional indexing.
VECTORIZER_STEP = "vectorizer"
CLASSIFIER_STEP = "classifier"

#: What :func:`prediction_scores` managed to obtain from a fitted estimator.
#: ``"probability"`` is in [0, 1] and sums to 1 across classes; ``"decision"``
#: is an unbounded, uncalibrated signed distance; ``"unavailable"`` means the
#: estimator exposes neither.
ScoreKind = str

_VECTORIZER_TYPES: dict[str, type] = {
    "tfidf": TfidfVectorizer,
}

_CLASSIFIER_TYPES: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "linear_svc": LinearSVC,
}

#: Parameters scikit-learn validates as a tuple. YAML has no tuple type, so
#: ``ngram_range: [1, 2]`` arrives as a list and fails parameter validation. Only
#: the *type* is adapted here; the value itself stays in configuration.
_TUPLE_VALUED_PARAMS: frozenset[str] = frozenset({"ngram_range"})


class UnknownEstimatorError(KeyError):
    """Raised when a configured ``type`` has no entry in the registry."""


def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    """Adapt YAML-native values to the types scikit-learn expects."""
    return {
        key: tuple(value) if key in _TUPLE_VALUED_PARAMS and isinstance(value, list) else value
        for key, value in params.items()
    }


def _with_seed(cls: type, params: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Add ``random_state=seed`` when the estimator accepts it.

    Reproducibility is a run-level property, so the seed comes from
    ``project.seed`` rather than being repeated in every model entry. An explicit
    ``random_state`` in configuration still wins, so a deliberately pinned model
    is not silently overridden.
    """
    if "random_state" in params:
        return params
    if "random_state" not in inspect.signature(cls).parameters:
        return params
    return {**params, "random_state": seed}


def build_vectorizer(config: VectorizerConfig) -> Any:
    """Instantiate a feature extractor from its configuration.

    Args:
        config: One entry of ``vectorizers`` in ``configs/model.yaml``.

    Returns:
        An unfitted scikit-learn transformer.

    Raises:
        UnknownEstimatorError: If ``config.type`` is not registered.
        TypeError: If ``config.params`` contains a keyword the estimator does not
            accept — a config typo, surfaced at construction rather than at fit.
    """
    try:
        cls = _VECTORIZER_TYPES[config.type]
    except KeyError:
        raise UnknownEstimatorError(
            f"Unknown vectorizer type '{config.type}'. "
            f"Registered types: {sorted(_VECTORIZER_TYPES)}"
        ) from None
    return cls(**_coerce_params(config.params))


def build_classifier(config: ClassifierConfig, *, seed: int) -> BaseEstimator:
    """Instantiate a classifier from its configuration.

    Args:
        config: The ``classifier`` block of one baseline.
        seed: Run seed, applied as ``random_state`` when the estimator accepts it
            and configuration has not pinned one.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        UnknownEstimatorError: If ``config.type`` is not registered.
    """
    try:
        cls = _CLASSIFIER_TYPES[config.type]
    except KeyError:
        raise UnknownEstimatorError(
            f"Unknown classifier type '{config.type}'. "
            f"Registered types: {sorted(_CLASSIFIER_TYPES)}"
        ) from None
    return cls(**_with_seed(cls, _coerce_params(config.params), seed=seed))


def build_baseline(
    model_config: ModelConfig,
    name: str,
    *,
    seed: int,
    multilabel: bool = False,
) -> Pipeline:
    """Assemble a named baseline into a fit-ready pipeline.

    Args:
        model_config: The validated ``configs/model.yaml`` contents.
        name: Baseline key, e.g. ``"tfidf_logreg"``.
        seed: Run seed, forwarded to the classifier.
        multilabel: Whether the active label mode is multi-label.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline` of
        ``(VECTORIZER_STEP, CLASSIFIER_STEP)``.

    Raises:
        KeyError: If ``name`` is not a configured baseline.
    """
    baseline = model_config.baseline(name)
    vectorizer_config = model_config.vectorizer_for(name)
    classifier = build_classifier(baseline.classifier, seed=seed)
    if multilabel:
        classifier = OneVsRestClassifier(classifier)

    logger.info(
        "model | %s: %s(%s) -> %s(%s)",
        name,
        vectorizer_config.type,
        ", ".join(f"{k}={v}" for k, v in sorted(vectorizer_config.params.items())),
        baseline.classifier.type,
        ", ".join(f"{k}={v}" for k, v in sorted(baseline.classifier.params.items())),
    )

    return Pipeline(
        [
            (VECTORIZER_STEP, build_vectorizer(vectorizer_config)),
            (CLASSIFIER_STEP, classifier),
        ]
    )


def resolved_params(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    """Return the parameters each step was actually constructed with.

    Read back off the estimators rather than off configuration, so the run
    manifest records what ran — including scikit-learn defaults that
    configuration never mentioned.

    Args:
        pipeline: A pipeline built by :func:`build_baseline`.

    Returns:
        ``{"vectorizer": {...}, "classifier": {...}}``, with values coerced to
        strings where they are not JSON-native.
    """

    def _plain(value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, tuple | list):
            return [_plain(item) for item in value]
        return str(value)

    return {
        step: {key: _plain(value) for key, value in pipeline.named_steps[step].get_params().items()}
        for step in (VECTORIZER_STEP, CLASSIFIER_STEP)
        if step in pipeline.named_steps
    }


def supports_probabilities(estimator: Any) -> bool:
    """Report whether ``estimator`` can produce calibrated-shaped probabilities.

    ``Pipeline`` exposes ``predict_proba`` only when its final estimator does, so
    a plain :func:`hasattr` is an accurate capability probe for both a pipeline
    and a bare classifier.
    """
    return hasattr(estimator, "predict_proba")


def prediction_scores(estimator: Any, texts: Sequence[str]) -> tuple[np.ndarray | None, ScoreKind]:
    """Return per-class scores and a label describing what they are.

    This is the graceful degradation point for :class:`~sklearn.svm.LinearSVC`,
    which has no ``predict_proba``. Rather than skipping confidence entirely or —
    worse — passing a margin off as a probability, the margin is returned
    alongside the tag ``"decision"`` so every consumer knows which it received.

    Args:
        estimator: A fitted pipeline or classifier.
        texts: Raw input texts, passed through the estimator unchanged.

    Returns:
        ``(scores, kind)`` where ``scores`` has shape ``(n_samples, n_classes)``
        and ``kind`` is ``"probability"``, ``"decision"``, or ``"unavailable"``.
        ``scores`` is ``None`` only for ``"unavailable"``.
    """
    if supports_probabilities(estimator):
        return np.asarray(estimator.predict_proba(texts)), "probability"

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(texts))
        if scores.ndim == 1:
            # Binary problems return one column: the signed distance for the
            # positive class. Mirror it so downstream code sees a uniform
            # (n_samples, n_classes) shape regardless of class count.
            scores = np.column_stack([-scores, scores])
        return scores, "decision"

    logger.warning(
        "model | %s exposes neither predict_proba nor decision_function; "
        "confidence information will be omitted from this run",
        type(estimator).__name__,
    )
    return None, "unavailable"
