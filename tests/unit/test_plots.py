"""Figure rendering.

A plot's *appearance* is not usefully assertable, so these tests cover what is:
the input guards, that a real image file lands on disk, and that split identity
colours are keyed by name. The last one matters because a filter that drops a
split must not repaint the survivors — colour follows the entity, never its rank.

Every test here writes into ``tmp_path``, and the autouse ``_close_figures``
fixture in ``conftest.py`` disposes of the figures afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.plots import SPLIT_COLORS, plot_class_distribution, plot_confusion_matrix

#: First eight bytes of any PNG file. Asserting on these proves an image was
#: encoded, where a mere ``exists()`` would also pass on a zero-byte file.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

CLASSES = ["ai", "networks", "vision"]
COUNTS = [[4, 1, 0], [0, 5, 1], [1, 0, 6]]


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def test_confusion_matrix_writes_a_png(tmp_path: Path) -> None:
    destination = plot_confusion_matrix(
        COUNTS, CLASSES, tmp_path / "confusion_matrix.png", normalized=False
    )
    assert destination.is_file()
    assert destination.read_bytes()[:8] == PNG_MAGIC


def test_confusion_matrix_creates_missing_parent_directories(tmp_path: Path) -> None:
    """A run directory may not exist yet when the first figure is rendered."""
    destination = plot_confusion_matrix(
        COUNTS, CLASSES, tmp_path / "nested" / "deeper" / "cm.png", normalized=False
    )
    assert destination.is_file()


def test_normalized_matrix_is_accepted(tmp_path: Path) -> None:
    fractions = [[0.8, 0.2, 0.0], [0.0, 0.83, 0.17], [0.14, 0.0, 0.86]]
    destination = plot_confusion_matrix(
        fractions, CLASSES, tmp_path / "cm.png", normalized=True
    )
    assert destination.read_bytes()[:8] == PNG_MAGIC


def test_non_square_matrix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="labels"):
        plot_confusion_matrix([[1, 2, 3], [4, 5, 6]], CLASSES, tmp_path / "cm.png")


def test_label_count_mismatch_is_rejected(tmp_path: Path) -> None:
    """Silently plotting a mislabelled axis would misattribute every error."""
    with pytest.raises(ValueError, match="labels"):
        plot_confusion_matrix(COUNTS, ["ai", "networks"], tmp_path / "cm.png")


def test_long_class_names_do_not_prevent_rendering(tmp_path: Path) -> None:
    """Real subfield names are long; they are elided rather than overflowing."""
    labels = ["Computer Vision and Pattern Recognition", "Computer Networks and Communications"]
    destination = plot_confusion_matrix(
        [[3, 1], [0, 4]], labels, tmp_path / "cm.png", normalized=False
    )
    assert destination.read_bytes()[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------
def test_class_distribution_writes_a_png(tmp_path: Path) -> None:
    destination = plot_class_distribution(
        {"train": {"ai": 20, "vision": 14}, "val": {"ai": 4, "vision": 3}},
        tmp_path / "class_distribution.png",
    )
    assert destination.read_bytes()[:8] == PNG_MAGIC


def test_class_distribution_accepts_an_explicit_class_order(tmp_path: Path) -> None:
    """The dataset vocabulary order is passed through so figures match the matrix."""
    destination = plot_class_distribution(
        {"train": {"ai": 5, "networks": 9, "vision": 7}},
        tmp_path / "cd.png",
        classes=CLASSES,
    )
    assert destination.is_file()


def test_a_class_missing_from_one_split_is_plotted_as_zero(tmp_path: Path) -> None:
    """A gap in a split is data, so it renders rather than raising."""
    destination = plot_class_distribution(
        {"train": {"ai": 8, "vision": 6}, "val": {"ai": 2}},
        tmp_path / "cd.png",
        classes=["ai", "vision"],
    )
    assert destination.is_file()


def test_empty_splits_are_rejected(tmp_path: Path) -> None:
    """Nothing to plot is a caller error, not an empty figure."""
    with pytest.raises(ValueError, match="No split contains any labelled records"):
        plot_class_distribution({"train": {}, "val": {}}, tmp_path / "cd.png")


def test_single_split_still_renders(tmp_path: Path) -> None:
    """A one-series chart needs no legend, and must not divide by zero on width."""
    destination = plot_class_distribution({"train": {"ai": 3}}, tmp_path / "cd.png")
    assert destination.read_bytes()[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# Split identity colours
# ---------------------------------------------------------------------------
def test_split_colors_are_keyed_by_name_not_position() -> None:
    assert set(SPLIT_COLORS) == {"train", "val", "test"}


def test_split_colors_are_distinct() -> None:
    """Two splits sharing a hue would make the grouped bars unreadable."""
    assert len(set(SPLIT_COLORS.values())) == len(SPLIT_COLORS)
    assert all(value.startswith("#") and len(value) == 7 for value in SPLIT_COLORS.values())
