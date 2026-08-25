"""Paper endpoints: listing, detail, classification, similarity, explanation, ask.

Two of these do real work rather than reading a file. ``POST /classify`` runs new
text through the run's fitted pipeline, and ``/similar`` computes cosine distance
in that pipeline's own TF-IDF space. The rest read what the training run already
wrote, so a number shown in the dashboard is the same number in ``report.md``.

``POST /{paper_id}/ask`` returns 501. It is here, rather than omitted, because the
dashboard has a research-assistant composer and the composer needs a definite
answer to submitting a question. A 501 carrying master spec §20's exact refusal
wording is that answer; a generated response with no retrieval behind it would be
the one outcome §20 forbids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from src.api.capabilities import (
    ATTENTION_UNAVAILABLE_REASON,
    EXPLANATION_CAVEAT,
    SIMILARITY_CAVEAT,
    SIMILARITY_METHOD,
    classification_caveat,
)
from src.api.deps import ActiveRun, Pagination, SettingsDep
from src.api.runstore import HELD_OUT_SPLITS, LoadedRun, PaperEntry, RunUnavailableError
from src.api.schemas import (
    AskRequest,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResult,
    ExplanationResponse,
    LabelScore,
    PaperDetail,
    PaperListResponse,
    PaperSummary,
    SectionAttention,
    SimilarItem,
    SimilarResponse,
    TermWeight,
)
from src.utils.logging import get_logger

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])

#: Accepted values for the ``split`` query parameter. ``held_out`` is the default
#: because a prediction on the training split is not evidence: the model was
#: fitted on those rows, so listing them beside held-out rows would quietly mix
#: training accuracy into what reads as a results table.
_SPLIT_FILTERS: dict[str, tuple[str, ...]] = {
    "held_out": HELD_OUT_SPLITS,
    "val": ("val",),
    "test": ("test",),
    "train": ("train",),
    "all": (),
}


def _summary(run: LoadedRun, entry: PaperEntry) -> PaperSummary:
    """Build the table row for one paper."""
    prediction = entry.prediction or {}
    confidence = prediction.get("confidence")
    confidence = float(confidence) if isinstance(confidence, int | float) else None
    kind = str(prediction.get("confidence_kind") or run.confidence_kind)

    return PaperSummary(
        paper_id=entry.paper_id,
        title=entry.record.title or entry.paper_id,
        authors_short=entry.authors_short,
        year=entry.year,
        split=entry.split,
        true_label=entry.record.label,
        predicted_label=prediction.get("predicted_label"),
        correct=prediction.get("correct"),
        confidence=confidence,
        confidence_kind=kind,
        # Derived here, never on the client: the threshold has one home
        # (master spec §15).
        needs_review=run.needs_review(confidence, kind) if entry.prediction else None,
    )


def _detail(run: LoadedRun, entry: PaperEntry) -> PaperDetail:
    """Build the preview payload for one paper."""
    base = _summary(run, entry)
    prediction = entry.prediction or {}
    scores = prediction.get("top_scores")
    meta = entry.record.meta

    return PaperDetail(
        **base.model_dump(),
        text=entry.record.text,
        labels=list(entry.record.labels),
        venue=meta.get("venue") if isinstance(meta.get("venue"), str) else None,
        n_authors=meta.get("n_authors") if isinstance(meta.get("n_authors"), int) else None,
        n_references=entry.n_references,
        predicted_scores=[
            LabelScore(label=str(item["label"]), score=float(item["score"]))
            for item in (scores or [])
            if isinstance(item, dict) and "label" in item and "score" in item
        ],
    )


def _require_paper(run: LoadedRun, paper_id: str) -> PaperEntry:
    """Look up a paper or raise 404.

    The lookup is a dictionary hit against the loaded corpus. No part of
    ``paper_id`` reaches the filesystem, so a traversal attempt returns 404 like
    any other unknown id (master spec §40).
    """
    entry = run.paper(paper_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No paper '{paper_id}' in run '{run.run_id}'.",
        )
    return entry


@router.get("", response_model=PaperListResponse, summary="Browse the corpus")
def list_papers(
    run: ActiveRun,
    page: Pagination,
    split: str = Query(
        "held_out",
        description="Which splits to include. 'held_out' covers val and test.",
    ),
    q: str | None = Query(None, max_length=200, description="Case-insensitive title match."),
    needs_review: bool | None = Query(
        None, description="Restrict to predictions above or below the review threshold."
    ),
) -> PaperListResponse:
    """Return a page of papers, filtered and searched.

    Raises:
        HTTPException: 422 when ``split`` is not one of the accepted values.
    """
    if split not in _SPLIT_FILTERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"split must be one of {sorted(_SPLIT_FILTERS)}; got '{split}'.",
        )

    wanted = _SPLIT_FILTERS[split]
    entries = run.entries(wanted or None)

    rows = [_summary(run, entry) for entry in entries]

    if q:
        needle = q.strip().lower()
        rows = [
            row
            for row in rows
            if needle in row.title.lower()
            or needle in row.paper_id.lower()
            or (row.true_label and needle in row.true_label.lower())
        ]
    if needs_review is not None:
        rows = [row for row in rows if row.needs_review is needs_review]

    total = len(rows)
    window = rows[page.offset : page.offset + page.limit]
    return PaperListResponse(
        items=window,
        total=total,
        limit=page.limit,
        offset=page.offset,
        splits=list(wanted) if wanted else sorted({row.split for row in rows}),
        query=q,
    )


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify new text with the active run's model",
)
def classify(payload: ClassifyRequest, run: ActiveRun, settings: SettingsDep) -> ClassifyResponse:
    """Run a real forward pass over submitted title and abstract.

    The fields are joined in the order ``text.fields`` specifies, which is the
    same composition the dataset build used, so an ad-hoc classification reaches
    the vectorizer in the form it was fitted on. Joining them in some other order
    would feed the model a distribution it never saw.

    Raises:
        HTTPException: 503 when the run has no saved model to load.
    """
    parts = {"title": payload.title, "abstract": payload.abstract}
    composed = "\n\n".join(
        parts[field] for field in settings.app.text.fields if parts.get(field)
    ).strip()
    if not composed:
        # Reachable when text.fields excludes every field the request supplied.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No usable text: this run was built from fields "
                f"{settings.app.text.fields}, and none of them were supplied."
            ),
        )

    try:
        outcome = run.classify([composed])[0]
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return ClassifyResponse(
        result=ClassifyResult(
            predicted_label=outcome["predicted_label"],
            confidence=outcome["confidence"],
            confidence_kind=outcome["confidence_kind"],
            scores=[LabelScore(**score) for score in outcome["scores"]],
            needs_review=outcome["needs_review"],
        ),
        run_id=run.run_id,
        model_display_name=run.model_display_name,
        caveat=classification_caveat(run),
    )


@router.get("/{paper_id}", response_model=PaperDetail, summary="One paper in full")
def get_paper(paper_id: str, run: ActiveRun) -> PaperDetail:
    """Return one paper's record and the run's prediction for it."""
    return _detail(run, _require_paper(run, paper_id))


@router.get(
    "/{paper_id}/similar",
    response_model=SimilarResponse,
    summary="Lexically nearest papers in the corpus",
)
def similar_papers(
    paper_id: str,
    run: ActiveRun,
    limit: int | None = Query(None, ge=1, le=50, description="Neighbours to return."),
) -> SimilarResponse:
    """Return the corpus papers with the most vocabulary in common.

    Cosine distance between TF-IDF vectors in the run's own fitted space. This is
    lexical overlap, and ``method`` plus ``caveat`` say so in the payload so the
    UI cannot present it as semantic similarity (master spec §17).

    Raises:
        HTTPException: 404 for an unknown paper; 503 when the model is unloadable.
    """
    _require_paper(run, paper_id)
    try:
        neighbours = run.similar(paper_id, top_k=limit)
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return SimilarResponse(
        query_paper_id=paper_id,
        items=[SimilarItem(**item) for item in neighbours],
        method=SIMILARITY_METHOD,
        caveat=SIMILARITY_CAVEAT,
    )


@router.get(
    "/{paper_id}/explanation",
    response_model=ExplanationResponse,
    summary="Which terms drove the predicted label",
)
def explanation(paper_id: str, run: ActiveRun) -> ExplanationResponse:
    """Decompose the model's decision for one paper into per-term contributions.

    ``section_attention`` is always the unavailable marker. The dashboard has a
    section-attention panel, and this endpoint is where it would be filled; it is
    not filled, because a bag-of-words model has no section representation to
    weight. Splitting the abstract into thirds and summing term weights would draw
    the same chart while meaning something else entirely.

    Raises:
        HTTPException: 404 for an unknown paper; 503 when the model is unloadable.
    """
    entry = _require_paper(run, paper_id)
    prediction = entry.prediction or {}
    label = prediction.get("predicted_label") or entry.record.label
    attention = SectionAttention(reason=ATTENTION_UNAVAILABLE_REASON)

    if not label:
        return ExplanationResponse(
            paper_id=paper_id,
            method="linear_term_contributions",
            section_attention=attention,
            caveat="This paper has no predicted or ground-truth label to explain.",
        )

    try:
        contributions = run.term_contributions(entry.record.text, label)
        decision = _decision_value(run, entry.record.text, label)
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    # Scaled on the server so every client draws identical bar widths from the
    # same numbers, rather than each inventing its own normalisation.
    largest = max((abs(item.contribution) for item in contributions), default=0.0)
    terms = [
        TermWeight(
            term=item.term,
            contribution=round(item.contribution, 6),
            tfidf=round(item.tfidf, 6),
            weight=round(abs(item.contribution) / largest, 4) if largest else 0.0,
        )
        for item in contributions
    ]

    return ExplanationResponse(
        paper_id=paper_id,
        predicted_label=label,
        method="linear_term_contributions",
        terms=terms,
        decision_value=decision,
        section_attention=attention,
        caveat=EXPLANATION_CAVEAT,
    )


def _decision_value(run: LoadedRun, text: str, label: str) -> float | None:
    """Return the raw decision value for ``label``, or ``None`` if unavailable.

    Reported alongside the term list so the contributions can be checked against
    the model's own arithmetic instead of taken on trust.
    """
    classifier = run.classifier
    if not hasattr(classifier, "decision_function"):
        return None
    classes = [str(name) for name in getattr(classifier, "classes_", run.classes)]
    if label not in classes:
        return None

    values = np.asarray(classifier.decision_function(run.vectorizer.transform([text])))
    row = values[0] if values.ndim > 1 else values
    if np.ndim(row) == 0:
        # Binary: one signed value, oriented toward classes_[1].
        return float(row) if classes.index(label) == 1 else -float(row)
    return float(row[classes.index(label)])


@router.post(
    "/{paper_id}/ask",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Not implemented — no retrieval index exists",
    responses={501: {"description": "Question answering is not built in this milestone."}},
)
def ask(paper_id: str, payload: AskRequest, run: ActiveRun) -> Any:
    """Refuse to answer questions about a paper, and say why.

    Master spec §20 requires that an answer with no supporting passage be the
    exact refusal below. With no retriever and no full text, *every* answer would
    be unsupported, so the endpoint refuses categorically rather than per
    question. The question itself is not stored.

    The machine-readable reason is deliberately a short slug; the sentence a user
    reads comes from the ``rag_ask`` capability in ``/api/meta``, so the
    explanation has one home.

    Raises:
        HTTPException: 501, always.
    """
    _require_paper(run, paper_id)
    logger.info("api | ask refused for %s (%d chars)", paper_id, len(payload.question))
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Information not found in the provided paper.",
        headers={"X-Unavailable-Reason": "retrieval-index-missing"},
    )
