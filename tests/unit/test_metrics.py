"""Evaluation metrics and the shape of an evaluation payload.

Two kinds of test here. The first checks numbers against values computed by hand,
because a metric function that is merely self-consistent can still be wrong. The
second checks payload *structure*, since ``metrics.json`` is a contract consumed
by the report writer today and by a results browser later.

The recurring theme is that a class absent from a split must keep its place. Its
metrics read 0 and its confusion-matrix row is zeros — that is the truth, and it
keeps row indices comparable between runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config.settings import Settings
from src.evaluation.metrics import (
    AVERAGE_NAMES,
    classification_report_dict,
    confidence_summary,
    confusion_matrix_data,
    evaluate_predictions,
    parse_primary_metric,
    primary_metric_value,
    required_averages,
)

CLASSES = ["ai", "networks", "vision"]

#: A few tests below feed ``evaluate_predictions`` a split holding a single
#: distinct label, to isolate one behaviour (an absent class keeping its row, a
#: disabled per-class table). ``balanced_accuracy_score`` builds its own confusion
#: matrix and — unlike every other call here — takes no ``labels`` argument, so it
#: warns about the degenerate shape. Production eval splits always carry every
#: class (stratification; see ``test_every_class_reaches_every_split``), so this
#: warning cannot fire there; it is expected only for these deliberately tiny
#: inputs, and silenced narrowly rather than by weakening production code.
_single_class_input = pytest.mark.filterwarnings("ignore:A single label was found")


# ---------------------------------------------------------------------------
# Metric identifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("macro_f1", ("macro", "f1")),
        ("micro_precision", ("micro", "precision")),
        ("weighted_recall", ("weighted", "recall")),
        ("accuracy", (None, "accuracy")),
        ("balanced_accuracy", (None, "balanced_accuracy")),
    ],
)
def test_parse_primary_metric(name: str, expected: tuple[str | None, str]) -> None:
    assert parse_primary_metric(name) == expected


@pytest.mark.parametrize("name", ["f1", "macro", "geometric_f1", "macro_auc", ""])
def test_unsupported_primary_metric_lists_valid_options(name: str) -> None:
    """Almost always a config typo, so the message must be actionable."""
    with pytest.raises(ValueError, match="macro_f1"):
        parse_primary_metric(name)


def test_configured_primary_metric_is_supported(settings: Settings) -> None:
    """Guards against ``configs/model.yaml`` drifting away from what code accepts."""
    parse_primary_metric(settings.model.evaluation.primary_metric)


def test_required_averages_adds_what_the_selector_needs() -> None:
    """Model selection must always have a number to select on."""
    assert required_averages(["micro"], "macro_f1") == ["macro", "micro"]


def test_required_averages_uses_canonical_order() -> None:
    """Two runs' metrics files should diff cleanly, so order is fixed."""
    assert required_averages(["weighted", "micro", "macro"], "accuracy") == list(AVERAGE_NAMES)


def test_required_averages_deduplicates() -> None:
    assert required_averages(["macro", "macro"], "macro_f1") == ["macro"]


def test_required_averages_rejects_unknown_entries() -> None:
    with pytest.raises(ValueError, match="samples"):
        required_averages(["samples"], "macro_f1")


# ---------------------------------------------------------------------------
# Values, against hand computation
# ---------------------------------------------------------------------------
def test_perfect_predictions_score_one() -> None:
    labels = ["ai", "networks", "vision", "ai"]
    metrics = evaluate_predictions(labels, list(labels), classes=CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    for values in metrics["averages"].values():
        assert values == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_macro_f1_matches_hand_computation() -> None:
    """Worked by hand so the assertion is independent of the implementation.

    ai:       TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=1/2
    networks: TP=1, FP=0, FN=1 -> P=1,   R=1/2, F1=2/3
    vision:   TP=1, FP=1, FN=0 -> P=1/2, R=1,   F1=2/3
    macro F1 = (1/2 + 2/3 + 2/3) / 3
    """
    y_true = ["ai", "ai", "networks", "networks", "vision"]
    y_pred = ["ai", "vision", "networks", "ai", "vision"]
    metrics = evaluate_predictions(y_true, y_pred, classes=CLASSES)

    expected = (0.5 + 2 / 3 + 2 / 3) / 3
    assert metrics["averages"]["macro"]["f1"] == pytest.approx(expected)
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    # Micro-averaged F1 equals accuracy in single-label multi-class.
    assert metrics["averages"]["micro"]["f1"] == pytest.approx(metrics["accuracy"])
    assert metrics["primary_metric"] == {"name": "macro_f1", "value": pytest.approx(expected)}


def test_per_class_metrics_carry_support() -> None:
    y_true = ["ai", "ai", "networks", "vision"]
    y_pred = ["ai", "networks", "networks", "vision"]
    per_class = evaluate_predictions(y_true, y_pred, classes=CLASSES)["per_class"]

    assert per_class["ai"]["support"] == 2
    assert per_class["ai"]["recall"] == pytest.approx(0.5)
    assert per_class["networks"]["precision"] == pytest.approx(0.5)
    assert per_class["vision"]["f1"] == 1.0
    assert sum(values["support"] for values in per_class.values()) == len(y_true)


@_single_class_input
def test_absent_class_keeps_its_row_and_scores_zero() -> None:
    """A class in the vocabulary but not in the split must not vanish."""
    y_true = ["ai", "ai"]
    metrics = evaluate_predictions(y_true, ["ai", "ai"], classes=CLASSES)

    assert list(metrics["per_class"]) == CLASSES
    assert metrics["per_class"]["vision"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert metrics["confusion_matrix"]["labels"] == CLASSES
    assert metrics["confusion_matrix"]["counts"][CLASSES.index("vision")] == [0, 0, 0]


@_single_class_input
def test_per_class_can_be_disabled() -> None:
    metrics = evaluate_predictions(["ai"], ["ai"], classes=CLASSES, per_class=False)
    assert metrics["per_class"] == {}


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def test_confusion_matrix_rows_are_true_classes() -> None:
    data = confusion_matrix_data(["ai", "ai"], ["ai", "vision"], classes=CLASSES)
    row = data["counts"][CLASSES.index("ai")]
    assert row[CLASSES.index("ai")] == 1
    assert row[CLASSES.index("vision")] == 1


def test_normalized_rows_sum_to_one_or_zero() -> None:
    """Empty rows normalise to zeros rather than NaN, so the payload stays JSON."""
    data = confusion_matrix_data(
        ["ai", "ai", "networks"], ["ai", "vision", "networks"], classes=CLASSES
    )
    for row in data["normalized"]:
        total = sum(row)
        assert total == pytest.approx(1.0) or total == 0.0
        assert all(math.isfinite(value) for value in row)
    assert data["normalized"][CLASSES.index("vision")] == [0.0, 0.0, 0.0]


def test_normalization_can_be_skipped() -> None:
    data = confusion_matrix_data(["ai"], ["ai"], classes=CLASSES, normalize=False)
    assert data["normalized"] is None


# ---------------------------------------------------------------------------
# Confidence, labelled by provenance
# ---------------------------------------------------------------------------
def test_probability_confidence_is_the_top_class_probability() -> None:
    scores = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    summary = confidence_summary(scores, "probability", correct=[True, False])

    assert summary["available"] is True
    assert summary["statistic"] == "max_probability"
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["max"] == pytest.approx(0.7)
    assert summary["mean_when_correct"] == pytest.approx(0.7)
    assert summary["mean_when_incorrect"] == pytest.approx(0.5)
    assert "Uncalibrated" in summary["caveat"]


def test_decision_confidence_is_the_top_two_margin_and_says_so() -> None:
    """An SVM margin must never be presentable as a probability (spec §15)."""
    scores = np.array([[2.0, 0.5, -1.0], [-0.5, 0.25, -2.0]])
    summary = confidence_summary(scores, "decision")

    assert summary["statistic"] == "top1_minus_top2_margin"
    assert summary["mean"] == pytest.approx((1.5 + 0.75) / 2)
    assert "NOT a probability" in summary["caveat"]
    assert "do not render it as a percentage" in summary["caveat"]


def test_unavailable_confidence_explains_itself_instead_of_faking_a_number() -> None:
    summary = confidence_summary(None, "unavailable")
    assert summary == {
        "kind": "unavailable",
        "available": False,
        "reason": summary["reason"],
    }
    assert "predict_proba" in summary["reason"]
    assert "statistic" not in summary


def test_all_correct_leaves_the_incorrect_mean_empty() -> None:
    """``None`` rather than 0.0: there is no such population, and 0.0 would read as one."""
    summary = confidence_summary(np.array([[0.9, 0.1]]), "probability", correct=[True])
    assert summary["mean_when_correct"] == pytest.approx(0.9)
    assert summary["mean_when_incorrect"] is None


def test_single_class_scores_do_not_index_a_missing_runner_up() -> None:
    summary = confidence_summary(np.array([[1.5], [-2.0]]), "decision")
    assert summary["mean"] == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Payload structure
# ---------------------------------------------------------------------------
def test_evaluation_payload_has_every_documented_key() -> None:
    metrics = evaluate_predictions(
        ["ai", "vision"],
        ["ai", "ai"],
        classes=CLASSES,
        scores=np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]),
        score_kind="probability",
    )
    assert set(metrics) == {
        "n_samples",
        "n_classes",
        "accuracy",
        "balanced_accuracy",
        "averages",
        "per_class",
        "confusion_matrix",
        "confidence",
        "hamming_loss",
        "primary_metric",
    }
    assert metrics["n_samples"] == 2
    assert metrics["n_classes"] == len(CLASSES)


@_single_class_input
def test_hamming_loss_is_acknowledged_not_fabricated() -> None:
    """In multi-class it reduces to 1 - accuracy, so a number would mislead."""
    metrics = evaluate_predictions(
        ["ai"], ["ai"], classes=CLASSES, hamming_loss_requested=True
    )
    assert metrics["hamming_loss"]["requested"] is True
    assert metrics["hamming_loss"]["value"] is None
    assert "multi-label metric" in metrics["hamming_loss"]["note"]


def test_payload_is_json_native() -> None:
    """No numpy scalars: the writer must not need a custom encoder."""
    import json

    metrics = evaluate_predictions(
        ["ai", "vision"],
        ["ai", "vision"],
        classes=CLASSES,
        scores=np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]),
        score_kind="probability",
    )
    assert json.loads(json.dumps(metrics)) == metrics


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="y_pred"):
        evaluate_predictions(["ai", "vision"], ["ai"], classes=CLASSES)


def test_empty_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty split"):
        evaluate_predictions([], [], classes=CLASSES)


def test_empty_class_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="class vocabulary"):
        evaluate_predictions(["ai"], ["ai"], classes=[])


@_single_class_input
def test_primary_metric_value_reports_a_missing_average() -> None:
    metrics = evaluate_predictions(["ai"], ["ai"], classes=CLASSES, averages=["micro"])
    del metrics["averages"]["macro"]
    with pytest.raises(KeyError, match="macro"):
        primary_metric_value(metrics, "macro_f1")


# ---------------------------------------------------------------------------
# scikit-learn's report, persisted verbatim
# ---------------------------------------------------------------------------
def test_classification_report_keeps_sklearn_layout() -> None:
    report = classification_report_dict(
        ["ai", "vision", "networks"], ["ai", "vision", "ai"], classes=CLASSES
    )
    assert {"accuracy", "macro avg", "weighted avg"} <= set(report)
    assert set(CLASSES) <= set(report)
    assert set(report["macro avg"]) == {"precision", "recall", "f1-score", "support"}
    assert isinstance(report["accuracy"], float)
