"""The no-leakage guarantee: TF-IDF sees the training split and nothing else.

This is the property master spec §9 exists to protect, and the one that most
quietly invalidates results when it breaks — a leaked vocabulary inflates
validation scores without producing any error.

Every assertion here is written to be **non-vacuous**. Asserting "the marker is
absent from the vocabulary" passes trivially if the marker could never have been
learned in the first place (too rare for ``min_df``, filtered as a stop word,
split by the tokenizer). So each test pairs the real fit with a *control* fit that
deliberately includes the held-out text, and asserts the marker appears there.
If the control ever stops finding it, the test is broken rather than the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.settings import Settings
from src.models.baselines import VECTORIZER_STEP, build_baseline, build_vectorizer
from src.training.dataset import ProcessedDataset

#: Injected into held-out text only. Lowercase alphabetic so the default token
#: pattern keeps it whole, and not an English stop word.
MARKER = "zzleakagesentineltoken"

#: Documents to inject it into. Must clear the configured ``min_df`` (2), or the
#: control fit would not learn the marker either and the test would prove nothing.
INJECTED_DOCS = 3


@pytest.fixture
def poisoned_val(processed_dataset: ProcessedDataset) -> list[str]:
    """Validation text with :data:`MARKER` appended to the first few documents."""
    texts = list(processed_dataset["val"].texts)
    for index in range(INJECTED_DOCS):
        texts[index] = f"{texts[index]} {MARKER}"
    return texts


def test_marker_is_learnable_when_fitting_sees_it(
    processed_dataset: ProcessedDataset, poisoned_val: list[str], settings: Settings
) -> None:
    """Control: the sentinel *does* enter the vocabulary when validation text is fitted.

    This test exists so the next one cannot pass for the wrong reason. It also
    pins the configured ``min_df`` against :data:`INJECTED_DOCS`.
    """
    train = processed_dataset["train"]
    vectorizer = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    vectorizer.fit(train.texts + poisoned_val)
    assert MARKER in vectorizer.vocabulary_


def test_marker_never_reaches_a_train_only_vocabulary(
    processed_dataset: ProcessedDataset, poisoned_val: list[str], settings: Settings
) -> None:
    """The real path: fitting on train leaves no trace of validation vocabulary."""
    train = processed_dataset["train"]
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed)
    pipeline.fit(train.texts, train.labels)

    vocabulary = pipeline.named_steps[VECTORIZER_STEP].vocabulary_
    assert MARKER not in vocabulary

    # Predicting on the poisoned split must not change that: transform maps unseen
    # terms to nothing, it does not extend the vocabulary.
    pipeline.predict(poisoned_val)
    assert MARKER not in pipeline.named_steps[VECTORIZER_STEP].vocabulary_


def test_vocabulary_contains_only_terms_present_in_train(
    processed_dataset: ProcessedDataset, settings: Settings
) -> None:
    """Every learned unigram traces back to a training document.

    Stronger than the sentinel check: it holds over the whole vocabulary rather
    than one planted token.
    """
    train = processed_dataset["train"]
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed)
    pipeline.fit(train.texts, train.labels)

    train_tokens = {token for text in train.texts for token in text.lower().split()}
    vocabulary = pipeline.named_steps[VECTORIZER_STEP].vocabulary_
    unigrams = [term for term in vocabulary if " " not in term]
    assert unigrams, "expected the vocabulary to contain unigrams"
    # Trailing punctuation is stripped by the tokenizer, so compare against a
    # punctuation-insensitive view of the training tokens.
    stripped = {token.strip(".,;:()[]") for token in train_tokens}
    assert set(unigrams) <= stripped


def test_predicting_does_not_refit_the_vectorizer(
    processed_dataset: ProcessedDataset, settings: Settings
) -> None:
    """The fitted state is frozen: no IDF weight moves when other splits arrive."""
    train, val = processed_dataset["train"], processed_dataset["val"]
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed)
    pipeline.fit(train.texts, train.labels)

    vectorizer = pipeline.named_steps[VECTORIZER_STEP]
    vocabulary_before = dict(vectorizer.vocabulary_)
    idf_before = np.array(vectorizer.idf_, copy=True)

    pipeline.predict(val.texts)
    pipeline.predict(processed_dataset["test"].texts)

    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(vectorizer.idf_, idf_before)


def test_pipeline_vocabulary_matches_a_standalone_train_only_fit(
    processed_dataset: ProcessedDataset, settings: Settings
) -> None:
    """The pipeline's learned features are exactly a train-only fit's features.

    This is the structural version of the guarantee: it holds for any input,
    without needing a planted sentinel, because the two objects saw the same text.
    """
    train = processed_dataset["train"]
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed)
    pipeline.fit(train.texts, train.labels)

    reference = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    reference.fit(train.texts)

    assert pipeline.named_steps[VECTORIZER_STEP].vocabulary_ == reference.vocabulary_
    np.testing.assert_allclose(pipeline.named_steps[VECTORIZER_STEP].idf_, reference.idf_)


def test_fitting_on_more_data_would_change_the_vocabulary(
    processed_dataset: ProcessedDataset, settings: Settings
) -> None:
    """Control for the test above: train-only and train+val fits genuinely differ.

    Without this, the equality assertion could hold simply because the splits
    contribute identical vocabularies, which would make it meaningless.
    """
    train, val = processed_dataset["train"], processed_dataset["val"]
    train_only = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    train_only.fit(train.texts)
    combined = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    combined.fit(train.texts + val.texts)

    assert train_only.vocabulary_ != combined.vocabulary_


@pytest.mark.parametrize("model_name", ["tfidf_logreg", "tfidf_svm"])
def test_classifier_never_sees_held_out_labels(
    processed_dataset: ProcessedDataset, settings: Settings, model_name: str
) -> None:
    """Fitting is called once, with training targets only.

    A classifier fitted on training labels can only know classes present there;
    if held-out labels had reached the fit, ``classes_`` would be the giveaway on
    a corpus where a class was missing from train.
    """
    train = processed_dataset["train"]
    pipeline = build_baseline(settings.model, model_name, seed=settings.seed)
    pipeline.fit(train.texts, train.labels)
    assert set(pipeline.named_steps["classifier"].classes_) == set(train.labels)
