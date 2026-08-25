"""Request and response models for the dashboard API (master spec §26).

Two jobs, and the second is the interesting one.

The first is ordinary: validate input. Every request model sets
``extra="forbid"`` and bounds every field, so a malformed or oversized body is a
422 produced by the schema rather than an exception raised somewhere inside
scikit-learn.

The second is to make *unavailability* a first-class part of the contract. This
system is partly built. Section attention needs the hierarchical model from
Milestone 3; grounded question answering needs a retrieval index; there is no PDF
parser yet. A response shape that could only express *answers* would force each
missing feature to be either omitted silently or filled with something
plausible-looking. So :class:`Capability` and the ``available``/``reason`` pairs
below exist to let the API say "not built, and here is why" in a form the UI can
render — which is the difference between a dashboard that is honest about its
state and one that looks finished.

Confidence carries its ``kind`` everywhere for the same reason. A logistic
regression's 0.82 is a probability; a LinearSVC's 0.82 is an uncalibrated margin
between the top two classes. Dropping the distinction would make the two
indistinguishable in the UI while meaning entirely different things.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AskRequest",
    "Capability",
    "ClassifyRequest",
    "ClassifyResponse",
    "ClassifyResult",
    "DatasetInfo",
    "DistributionResponse",
    "DistributionSlice",
    "ErrorResponse",
    "ExplanationResponse",
    "HealthResponse",
    "LabelScore",
    "MetaResponse",
    "PaperDetail",
    "PaperListResponse",
    "PaperSummary",
    "RunDetail",
    "RunListResponse",
    "RunSummaryOut",
    "SectionAttention",
    "SimilarItem",
    "SimilarResponse",
    "StatTile",
    "StatsResponse",
    "StorageInfo",
    "TermWeight",
    "TrendSeries",
    "TrendsResponse",
    "UnavailableFeature",
    "UserInfo",
]

#: Requests reject unknown keys: a typo in a client payload is a bug worth a 422
#: rather than a field silently ignored.
_REQUEST = ConfigDict(extra="forbid", str_strip_whitespace=True)

#: Responses are constructed only by this package, so they need no input
#: strictness — but they do forbid extras, which turns a renamed field into a
#: server-side error instead of a key the frontend quietly stops receiving.
#:
#: ``protected_namespaces`` is cleared because several fields legitimately begin
#: with ``model_`` — they describe the *trained model*, which is the domain
#: object here, not pydantic's own configuration namespace.
_RESPONSE = ConfigDict(extra="forbid", protected_namespaces=())


def _max_text_chars() -> int:
    """Return the configured ceiling for free-text input.

    Imported lazily: this module is imported while the settings module is still
    being wired up by :mod:`src.api.app`, and the limit is only needed when a
    request actually arrives.
    """
    from src.api.deps import get_settings

    return get_settings().api.security.max_text_chars


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------
class LabelScore(BaseModel):
    """One class and the model's score for it."""

    model_config = _RESPONSE

    label: str
    score: float


class UnavailableFeature(BaseModel):
    """A feature the UI expects that this build does not provide.

    Returned with HTTP 200 in place of the feature's own payload, so the panel
    can render an explicit "not built" state. A 404 would be wrong: the resource
    exists conceptually and the client asked correctly.
    """

    model_config = _RESPONSE

    available: Literal[False] = False
    #: Why it is unavailable, in words a user can act on.
    reason: str
    #: The milestone or prerequisite that would make it available.
    requires: str | None = None


class Capability(BaseModel):
    """Whether one named feature is implemented in this build."""

    model_config = _RESPONSE

    key: str
    label: str
    available: bool
    #: Populated when ``available`` is false. Rendered by the UI verbatim.
    reason: str | None = None


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness plus a summary of what the server managed to load.

    Deliberately unauthenticated and deliberately not a bare ``{"ok": true}``: a
    process that is up but has no run, no dataset, or an unloadable model is the
    failure mode worth reporting, and it is invisible to a plain liveness ping.
    """

    model_config = _RESPONSE

    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    environment: str
    run_id: str | None = None
    dataset_ready: bool = False
    model_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    """Provenance of the corpus behind a run."""

    model_config = _RESPONSE

    source: str | None = None
    directory: str | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)
    n_classes: int | None = None
    classes: list[str] = Field(default_factory=list)
    built_at: str | None = None
    #: True when the corpus is the generated test fixture rather than real
    #: papers. The UI must surface this: the fixture is separable by
    #: construction, so its metrics are a wiring check, not a research result.
    is_synthetic: bool = False
    #: True when the dataset on disk no longer hashes to what the run trained on.
    is_stale: bool = False
    integrity_findings: list[dict[str, Any]] = Field(default_factory=list)


class RunSummaryOut(BaseModel):
    """One run, as listed in the run picker."""

    model_config = _RESPONSE

    run_id: str
    model_name: str | None = None
    model_display_name: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    n_classes: int | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)
    is_complete: bool = False
    is_active: bool = False


class RunListResponse(BaseModel):
    """Every discoverable run, newest first."""

    model_config = _RESPONSE

    runs: list[RunSummaryOut]
    active_run_id: str | None = None


class RunDetail(BaseModel):
    """Full description of one run, including its headline metrics."""

    model_config = _RESPONSE

    run_id: str
    model_name: str
    model_display_name: str
    created_at: str | None = None
    finished_at: str | None = None
    seed: int | None = None
    git_commit: str | None = None
    label_mode: str | None = None
    taxonomy_level: str | None = None
    classes: list[str] = Field(default_factory=list)
    confidence_kind: str = "unavailable"
    primary_split: str = "val"
    #: ``{split: {accuracy, macro_f1, n_samples, ...}}`` — the headline numbers
    #: only, since the full ``metrics.json`` is large and mostly per-class.
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dataset: DatasetInfo
    model_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class UserInfo(BaseModel):
    """Who the dashboard says you are.

    There is no authentication layer yet (master spec §40 calls for
    authentication-*ready*, which is what the API-key dependency provides). So
    this describes a local operator, and ``is_authenticated`` is false — the UI
    must not imply a signed-in account that does not exist.
    """

    model_config = _RESPONSE

    first_name: str
    full_name: str
    role: str
    initials: str
    is_authenticated: bool = False


class StorageInfo(BaseModel):
    """Measured disk usage against the configured quota."""

    model_config = _RESPONSE

    used_bytes: int
    used_gb: float
    quota_gb: float
    percent: float
    measured: list[str] = Field(default_factory=list)


class MetaResponse(BaseModel):
    """Everything the dashboard needs before it can render anything.

    One request rather than five, because the shell — greeting, nav, run banner,
    storage meter — is useless in pieces, and five parallel fetches would each
    need their own failure state.
    """

    model_config = _RESPONSE

    app_name: str
    version: str
    environment: str
    user: UserInfo
    storage: StorageInfo
    run: RunDetail | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    #: Standing caveats the UI must keep visible (master spec §14/§17/§23).
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dashboard aggregates
# ---------------------------------------------------------------------------
class StatTile(BaseModel):
    """One headline number.

    ``note`` replaces the period-over-period delta a dashboard of this shape
    usually shows. There is only ever one run in view, so there is no previous
    period to compare against, and an invented trend arrow is exactly the kind of
    decoration that reads as data.
    """

    model_config = _RESPONSE

    id: str
    label: str
    value: str
    #: Short qualifier under the value: what it counts, or which split it is from.
    note: str | None = None
    icon: str
    hue: str


class StatsResponse(BaseModel):
    """The stat row, derived entirely from the active run."""

    model_config = _RESPONSE

    tiles: list[StatTile]
    run_id: str
    split: str


class DistributionSlice(BaseModel):
    """One wedge of the domain donut."""

    model_config = _RESPONSE

    label: str
    count: int
    share: float


class DistributionResponse(BaseModel):
    """Corpus composition by class."""

    model_config = _RESPONSE

    total: int
    unit: str
    slices: list[DistributionSlice]
    #: Which labels were counted — ground truth from the source taxonomy, not
    #: model predictions. The two differ, and the distinction matters.
    basis: str
    note: str | None = None


class TrendSeries(BaseModel):
    """One class's counts, aligned to the shared year axis."""

    model_config = _RESPONSE

    label: str
    values: list[int]


class TrendsResponse(BaseModel):
    """Publication counts per class per year."""

    model_config = _RESPONSE

    years: list[int]
    series: list[TrendSeries]
    #: Series omitted to keep the chart legible, reported rather than dropped
    #: silently.
    dropped_series: int = 0
    basis: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    """One row of the papers table."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    authors_short: str | None = None
    year: int | None = None
    split: str
    #: Ground-truth label from the source taxonomy.
    true_label: str | None = None
    true_labels: list[str] = Field(default_factory=list)
    #: The run's prediction. ``None`` for training-split papers, which the model
    #: was fitted on and therefore has no honest score for.
    predicted_label: str | None = None
    predicted_labels: list[str] = Field(default_factory=list)
    correct: bool | None = None
    confidence: float | None = None
    confidence_kind: str = "unavailable"
    #: Server-derived (master spec §15). The client must not recompute it.
    needs_review: bool | None = None


class PaperDetail(PaperSummary):
    """One paper in full, for the preview panel."""

    model_config = _RESPONSE

    text: str
    labels: list[str] = Field(default_factory=list)
    venue: str | None = None
    n_authors: int | None = None
    n_references: int | None = None
    predicted_scores: list[LabelScore] = Field(default_factory=list)


class PaperListResponse(BaseModel):
    """A page of papers."""

    model_config = _RESPONSE

    items: list[PaperSummary]
    total: int
    limit: int
    offset: int
    splits: list[str]
    query: str | None = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    """Text to classify with the active run's model.

    Title and abstract are separate fields because the vectorizer was fitted on
    the configured field order; the server joins them the same way the dataset
    build did, so an ad-hoc classification goes through the same composition as
    every training example.
    """

    model_config = _REQUEST

    title: Annotated[str, Field(min_length=1, max_length=1000)]
    abstract: Annotated[str, Field(default="", max_length=200_000)] = ""

    @field_validator("abstract")
    @classmethod
    def _within_configured_limit(cls, value: str) -> str:
        """Reject text above ``security.max_text_chars``.

        Enforced against configuration rather than a literal so the ceiling has
        one definition (master spec §32). The static ``max_length`` above is only
        a backstop that bounds memory before this runs.
        """
        limit = _max_text_chars()
        if len(value) > limit:
            raise ValueError(f"abstract is {len(value)} characters; the limit is {limit}")
        return value


class ClassifyResult(BaseModel):
    """One classification outcome."""

    model_config = _RESPONSE

    predicted_label: str
    predicted_labels: list[str] = Field(default_factory=list)
    confidence: float | None = None
    confidence_kind: str = "unavailable"
    scores: list[LabelScore] = Field(default_factory=list)
    needs_review: bool | None = None


class ClassifyResponse(BaseModel):
    """Result of a real forward pass through the run's fitted pipeline."""

    model_config = _RESPONSE

    result: ClassifyResult
    run_id: str
    model_display_name: str
    #: What the score is and is not. Always populated.
    caveat: str


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
class SimilarItem(BaseModel):
    """One neighbour of a query paper."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    score: float
    label: str | None = None
    split: str | None = None


class SimilarResponse(BaseModel):
    """Nearest neighbours by cosine distance in the run's TF-IDF space."""

    model_config = _RESPONSE

    query_paper_id: str
    items: list[SimilarItem]
    #: ``"tfidf_cosine"``. Named in the payload so the UI cannot describe this as
    #: semantic or embedding-based similarity, which it is not.
    method: str
    #: Master spec §17: shared vocabulary is not methodological equivalence.
    caveat: str


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
class TermWeight(BaseModel):
    """One term's contribution to a linear model's decision."""

    model_config = _RESPONSE

    term: str
    contribution: float
    tfidf: float
    #: Contribution as a share of the largest in this list, for bar widths. The
    #: server computes it so every client scales the bars identically.
    weight: float


class SectionAttention(UnavailableFeature):
    """Placeholder for per-section attention weights.

    The dashboard was designed around a section-attention panel. Producing one
    requires the hierarchical attention network from Milestone 3; a linear
    bag-of-words model has no notion of a section at all. Approximating it — by
    splitting the abstract and summing term weights per chunk, say — would
    produce a chart indistinguishable from real attention and meaning something
    completely different, so this stays explicitly unavailable.
    """

    model_config = _RESPONSE

    requires: str = "Milestone 3 — hierarchical attention network over paper sections"


class ExplanationResponse(BaseModel):
    """Why the model assigned the label it did."""

    model_config = _RESPONSE

    paper_id: str
    predicted_label: str | None = None
    #: ``"linear_term_contributions"``.
    method: str
    terms: list[TermWeight] = Field(default_factory=list)
    #: Decision value for the predicted class, so the term list is checkable
    #: against the model's own arithmetic rather than taken on trust.
    decision_value: float | None = None
    section_attention: SectionAttention
    #: Master spec §14: a weight is not a causal claim.
    caveat: str


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """A question about one paper."""

    model_config = _REQUEST

    question: Annotated[str, Field(min_length=1, max_length=2000)]


class ErrorResponse(BaseModel):
    """Uniform error body.

    Every failure the API produces on purpose has this shape, so the client has
    one error path instead of guessing whether a body holds ``detail``,
    ``message``, or an HTML page from a proxy.
    """

    model_config = _RESPONSE

    error: str
    detail: str
    #: Populated when there is a concrete next step, e.g. the command to run.
    hint: str | None = None
