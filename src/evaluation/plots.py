"""Figures for a training run (master spec §13/§45).

Two figures, each answering a question a metrics table answers poorly:

* **Confusion matrix** — *where* the errors go. A macro-F1 of 0.72 says nothing
  about whether the misses are spread evenly or whether two classes are being
  systematically swapped.
* **Class distribution** — whether the splits are balanced against each other,
  and how much support each per-class number actually rests on.

Colour follows the encoding's job rather than taste. The confusion matrix encodes
*magnitude*, so it uses a single sequential hue, light to dark. The class
distribution encodes *identity* (which split), so it uses a fixed categorical
order drawn from this project's validated light-mode palette — matplotlib renders
on a white surface, which is the surface that palette was validated against.
Splits keep their colour whatever the class order, since colour follows the
entity and never its rank.

Matplotlib's ``Agg`` backend is selected at import so a run works headless — in
CI, over SSH, or from a scheduled job — without a display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.io import ensure_dir, resolve_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

__all__ = [
    "SPLIT_COLORS",
    "plot_class_distribution",
    "plot_confusion_matrix",
]

logger = get_logger(__name__)

#: Identity colours for the splits, taken in fixed order from the project's
#: validated light-mode categorical palette (see ``frontend/README.md``).
#: Assigned by split name, never by position, so adding or dropping a split
#: cannot repaint the others.
SPLIT_COLORS: dict[str, str] = {
    "train": "#2a78d6",
    "val": "#eb6834",
    "test": "#1baf7a",
}

#: Sequential single hue for magnitude. Never a rainbow: a rainbow ramp implies
#: category boundaries that a count does not have.
_MAGNITUDE_CMAP = "Blues"

#: Above this many classes, per-cell annotations overlap into illegibility and
#: the colour ramp alone carries the reading.
_MAX_ANNOTATED_CLASSES = 20

#: Recessive grid, so the data marks stay dominant.
_GRID_STYLE: dict[str, Any] = {"color": "#d5d9e0", "linewidth": 0.6, "alpha": 0.9}

_INK_PRIMARY = "#1c2128"
_INK_MUTED = "#5c6672"


def _shorten(label: str, limit: int = 22) -> str:
    """Truncate a long class name for axis display, marking the elision."""
    return label if len(label) <= limit else label[: limit - 1] + "…"


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    output_path: Path | str,
    *,
    title: str = "Confusion matrix",
    normalized: bool = True,
    dpi: int = 150,
) -> Path:
    """Render a confusion matrix heatmap.

    Args:
        matrix: Square matrix, rows = true class, columns = predicted class.
        labels: Class names in matrix order.
        output_path: Destination file; the extension selects the format.
        title: Figure title.
        normalized: Whether values are row-normalised fractions rather than raw
            counts. Controls the annotation format and the colour-bar label.
        dpi: Output resolution.

    Returns:
        The resolved output path.

    Raises:
        ValueError: If the matrix is not square or does not match ``labels``.
    """
    data = np.asarray(matrix, dtype=float)
    size = len(labels)
    if data.shape != (size, size):
        raise ValueError(f"Confusion matrix is {data.shape} but there are {size} labels")

    destination = resolve_path(output_path)
    ensure_dir(destination.parent)

    # Grow with the class count so tick labels stay legible instead of colliding.
    side = max(5.5, min(16.0, 2.6 + 0.62 * size))
    figure, axes = plt.subplots(figsize=(side + 1.2, side), dpi=dpi)

    # vmax pinned to 1.0 for fractions so colour intensity is comparable across
    # runs; left to the data for raw counts, where no fixed ceiling exists.
    image = axes.imshow(
        data, cmap=_MAGNITUDE_CMAP, vmin=0.0, vmax=1.0 if normalized else None, aspect="equal"
    )

    short = [_shorten(label) for label in labels]
    axes.set_xticks(range(size), short, rotation=45, ha="right", fontsize=9, color=_INK_PRIMARY)
    axes.set_yticks(range(size), short, fontsize=9, color=_INK_PRIMARY)
    axes.set_xlabel("Predicted class", fontsize=10, color=_INK_PRIMARY, labelpad=8)
    axes.set_ylabel("True class", fontsize=10, color=_INK_PRIMARY, labelpad=8)
    axes.set_title(title, fontsize=12, color=_INK_PRIMARY, pad=14)

    # Hairline separators between cells; no heavy frame competing with the data.
    axes.set_xticks(np.arange(-0.5, size, 1), minor=True)
    axes.set_yticks(np.arange(-0.5, size, 1), minor=True)
    axes.grid(which="minor", color="#ffffff", linewidth=1.2)
    axes.tick_params(which="minor", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    if size <= _MAX_ANNOTATED_CLASSES:
        # Threshold at mid-ramp: dark ink on light cells, light ink on dark ones,
        # so every annotation keeps contrast against its own background.
        cutoff = (1.0 if normalized else data.max() or 1.0) * 0.55
        for row in range(size):
            for column in range(size):
                value = data[row, column]
                text = f"{value:.2f}" if normalized else f"{int(round(value))}"
                if not normalized or value > 0.005:
                    axes.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#ffffff" if value > cutoff else _INK_PRIMARY,
                    )

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    bar.set_label(
        "Share of true class" if normalized else "Papers",
        fontsize=9,
        color=_INK_MUTED,
    )
    bar.ax.tick_params(labelsize=8, colors=_INK_MUTED)
    bar.outline.set_visible(False)

    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)
    logger.info("plot | wrote %s", destination.name)
    return destination


def plot_class_distribution(
    counts_by_split: Mapping[str, Mapping[str, int]],
    output_path: Path | str,
    *,
    classes: Sequence[str] | None = None,
    title: str = "Class distribution by split",
    dpi: int = 150,
) -> Path:
    """Render grouped bars of per-class support for each split.

    Args:
        counts_by_split: ``{split_name: {class_name: count}}``.
        output_path: Destination file; the extension selects the format.
        classes: Class order. Defaults to descending total frequency, which puts
            the long tail — where per-class metrics are least reliable — together
            at one end.
        title: Figure title.
        dpi: Output resolution.

    Returns:
        The resolved output path.

    Raises:
        ValueError: If no split contains any class.
    """
    splits = [name for name, counts in counts_by_split.items() if counts]
    if not splits:
        raise ValueError("No split contains any labelled records to plot.")

    if classes is None:
        totals: dict[str, int] = {}
        for counts in counts_by_split.values():
            for label, count in counts.items():
                totals[label] = totals.get(label, 0) + count
        classes = [label for label, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]

    destination = resolve_path(output_path)
    ensure_dir(destination.parent)

    positions = np.arange(len(classes))
    group_width = 0.8
    bar_width = group_width / len(splits)

    height = max(4.2, 1.9 + 0.42 * len(classes))
    figure, axes = plt.subplots(figsize=(max(7.0, 3.4 + 0.5 * len(classes)), height), dpi=dpi)

    for index, split in enumerate(splits):
        counts = counts_by_split[split]
        offset = -group_width / 2 + bar_width * (index + 0.5)
        axes.bar(
            positions + offset,
            [counts.get(label, 0) for label in classes],
            width=bar_width * 0.9,  # gap between adjacent bars, in surface colour
            label=split,
            color=SPLIT_COLORS.get(split, "#4a3aa7"),
            edgecolor="#ffffff",
            linewidth=0.8,
        )

    axes.set_xticks(positions, [_shorten(label) for label in classes], rotation=35, ha="right",
                    fontsize=9, color=_INK_PRIMARY)
    axes.set_ylabel("Papers", fontsize=10, color=_INK_PRIMARY, labelpad=8)
    # The legend sits above the plot area when present, so the title needs room.
    axes.set_title(
        title, fontsize=12, color=_INK_PRIMARY, pad=30 if len(splits) >= 2 else 14
    )
    axes.tick_params(axis="y", labelsize=9, colors=_INK_MUTED)
    axes.yaxis.grid(True, **_GRID_STYLE)
    axes.set_axisbelow(True)
    axes.xaxis.grid(False)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#c3c9d2")

    # A legend is always present for two or more series, so identity never rests
    # on colour alone. It is anchored *above* the axes rather than inside them:
    # a bar at full height overlaps an inside legend, and no fixed amount of
    # headroom avoids that for every dataset.
    if len(splits) >= 2:
        legend = axes.legend(
            frameon=False,
            fontsize=9,
            ncols=len(splits),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.0,
            labelcolor=_INK_MUTED,
        )
        legend.set_title(None)

    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)
    logger.info("plot | wrote %s", destination.name)
    return destination
