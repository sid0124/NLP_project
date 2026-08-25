"""Dashboard aggregates: the stat row, the domain donut, and the trends chart.

Every number here is computed from the active run's artifacts and the corpus it
was trained on. Nothing is estimated, and nothing is compared against a previous
period — there is only one run in view, so a trend arrow would have nothing behind
it. Where a panel in the original design needed data this system does not have,
the panel's payload says so rather than approximating.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import ActiveRun, SettingsDep
from src.api.runstore import HELD_OUT_SPLITS, LoadedRun
from src.api.schemas import (
    DistributionResponse,
    DistributionSlice,
    StatsResponse,
    StatTile,
    TrendSeries,
    TrendsResponse,
)
from src.utils.logging import get_logger

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["dashboard"])

#: Which categorical slot each tile borrows for its icon. Fixed per tile id, so a
#: tile keeps its colour when the row is reordered.
_TILE_HUES = {
    "papers": "--series-6",
    "domains": "--series-3",
    "score": "--series-1",
    "review": "--series-4",
}


def _format_count(value: int) -> str:
    """Group thousands with commas, which is what the tiles display."""
    return f"{value:,}"


def _review_counts(run: LoadedRun) -> tuple[int, int]:
    """Return ``(flagged, scored)`` over held-out predictions.

    Only held-out splits are counted. The model was fitted on the training split,
    so its confidence there is inflated and pooling it in would understate how
    much of the corpus actually needs a human look.
    """
    flagged = 0
    scored = 0
    for entry in run.entries(HELD_OUT_SPLITS):
        prediction = entry.prediction
        if prediction is None:
            continue
        confidence = prediction.get("confidence")
        if not isinstance(confidence, int | float):
            continue
        scored += 1
        if run.needs_review(float(confidence), prediction.get("confidence_kind")):
            flagged += 1
    return flagged, scored


@router.get("/stats", response_model=StatsResponse, summary="The four headline tiles")
def stats(run: ActiveRun) -> StatsResponse:
    """Return the stat row for the active run.

    The four tiles answer, in order: how much data, how many classes, how well it
    scored, and how much of it a human still needs to look at. The last is the one
    a dashboard of this kind usually omits and the one master spec §15 makes
    non-optional.
    """
    split = run.primary_split
    split_sizes = run.summary().split_sizes
    total = sum(split_sizes.values())
    held_out = sum(count for name, count in split_sizes.items() if name in HELD_OUT_SPLITS)

    tiles: list[StatTile] = [
        StatTile(
            id="papers",
            label="Papers in Corpus",
            value=_format_count(total),
            note=f"{_format_count(held_out)} held out for evaluation",
            icon="papers",
            hue=_TILE_HUES["papers"],
        ),
        StatTile(
            id="domains",
            label="Domains Classified",
            value=_format_count(len(run.classes)),
            note=f"{run.manifest.get('labels', {}).get('taxonomy_level', 'topic')} level",
            icon="pie",
            hue=_TILE_HUES["domains"],
        ),
    ]

    # The headline metric, named by the run rather than assumed to be accuracy.
    metrics = run.metrics.get(split, {})
    primary = metrics.get("primary_metric") or {}
    metric_name = str(primary.get("name") or "score")
    metric_value = primary.get("value")
    tiles.append(
        StatTile(
            id="score",
            label=metric_name.replace("_", " ").title(),
            value=f"{float(metric_value) * 100:.1f}%" if isinstance(metric_value, int | float)
            else "n/a",
            note=f"on the {split} split, {metrics.get('n_samples', 0)} papers",
            icon="trend",
            hue=_TILE_HUES["score"],
        )
    )

    flagged, scored = _review_counts(run)
    if scored:
        review_note = f"{flagged / scored * 100:.0f}% of {scored} scored predictions"
    elif run.confidence_kind == "unavailable":
        review_note = f"{run.model_display_name} exposes no confidence score"
    else:
        review_note = "no scored predictions yet"
    tiles.append(
        StatTile(
            id="review",
            label="Flagged for Review",
            value=_format_count(flagged) if scored else "n/a",
            note=review_note,
            icon="flag",
            hue=_TILE_HUES["review"],
        )
    )

    return StatsResponse(tiles=tiles, run_id=run.run_id, split=split)


@router.get(
    "/stats/domains",
    response_model=DistributionResponse,
    summary="Corpus composition by domain",
)
def domain_distribution(run: ActiveRun, settings: SettingsDep) -> DistributionResponse:
    """Return the donut slices.

    Counts **ground-truth** labels from the source taxonomy across the whole
    corpus, not model predictions. That is the composition of the data the model
    was trained on, which is what a reader needs in order to judge every per-class
    number elsewhere on the page. Counting predictions instead would show the
    model's view of itself, and the two are easy to confuse.

    Classes beyond the configured maximum collapse into one labelled overflow
    bucket rather than being dropped, so the slices always sum to the total.
    """
    config = settings.api.distribution
    counts = run.class_counts
    total = sum(counts.values())

    items = list(counts.items())
    head = items[: config.max_slices]
    tail = items[config.max_slices :]

    slices = [
        DistributionSlice(
            label=label, count=count, share=round(count / total, 4) if total else 0.0
        )
        for label, count in head
    ]
    if tail:
        overflow = sum(count for _, count in tail)
        slices.append(
            DistributionSlice(
                label=config.other_label,
                count=overflow,
                share=round(overflow / total, 4) if total else 0.0,
            )
        )

    note = None
    if tail:
        note = f"{len(tail)} smaller domain(s) grouped as {config.other_label}."
    return DistributionResponse(
        total=total,
        unit="papers",
        slices=slices,
        basis="ground-truth labels from the source taxonomy",
        note=note,
    )


@router.get(
    "/research/trends",
    response_model=TrendsResponse,
    summary="Papers per domain per publication year",
)
def trends(run: ActiveRun, settings: SettingsDep) -> TrendsResponse:
    """Return per-domain publication counts over time.

    This is corpus composition over time — how many papers of each domain the
    dataset holds per year — and not a measurement of research activity in the
    field. The corpus is a filtered sample, so a rising line means "this sample
    contains more such papers", which is a fact about the sample.

    Series are capped for legibility and the number omitted is reported, so a
    truncated chart cannot read as a complete one.
    """
    table = run.counts_by_year
    if not table:
        return TrendsResponse(
            years=[],
            series=[],
            basis="publication year recorded at dataset build time",
            note="No paper in this corpus recorded a publication year.",
        )

    years = sorted({year for counts in table.values() for year in counts})

    # Ranked by corpus frequency so the cap keeps the domains a reader is most
    # likely to be looking for. Colour is still assigned by name on the client,
    # so this ordering never repaints a series.
    ranked = sorted(table.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    limit = settings.api.trends.max_series
    kept, dropped = ranked[:limit], ranked[limit:]

    series = [
        TrendSeries(label=label, values=[counts.get(year, 0) for year in years])
        for label, counts in kept
    ]

    note = None
    if dropped:
        note = (
            f"Showing the {len(kept)} largest domains; {len(dropped)} smaller "
            f"one(s) are omitted to keep the chart legible."
        )
    return TrendsResponse(
        years=years,
        series=series,
        dropped_series=len(dropped),
        basis="publication year recorded at dataset build time",
        note=note,
    )
