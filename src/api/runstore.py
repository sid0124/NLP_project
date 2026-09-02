"""Reading a finished training run as a queryable object.

The API trains nothing. It reads a ``results/<run_id>/`` directory produced by
``scripts/train_baseline.py`` and answers the dashboard's questions from those
artifacts plus the processed dataset the run was trained on. That split of
responsibility is deliberate: training is a batch job with a seed and a
manifest, and an HTTP request is the wrong place to start one.

What a loaded run can answer, and what it genuinely cannot:

==================== ======================================================
Answerable           Source
==================== ======================================================
Corpus composition   ``dataset_manifest.json`` + the split files
Per-paper prediction ``predictions_<split>.jsonl`` — written at train time
Headline metrics     ``metrics.json``
Ad-hoc classification ``model.joblib`` — a real forward pass on new text
Nearest neighbours   cosine distance in the run's own fitted TF-IDF space
Term contributions   ``tfidf_value * coefficient`` for the predicted class
==================== ======================================================

Not answerable, and reported as unavailable rather than approximated:
**section-level attention** (needs the Milestone 3 hierarchical model),
**grounded question answering** (needs a retrieval index), and **semantic
embedding similarity** (TF-IDF overlap is lexical, not learned — master
spec §17).

Predictions are read from disk rather than recomputed. The file was written by
the run being described, so a number shown in the dashboard is the same number
that appears in ``report.md``; recomputing invites the two to disagree.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from src.config.settings import Settings
from src.data_pipeline.split import DATASET_MANIFEST_NAME
from src.evaluation.report import (
    METRICS_NAME,
    MODEL_NAME,
    RUN_MANIFEST_NAME,
    predictions_file_name,
)
from src.models.baselines import CLASSIFIER_STEP, VECTORIZER_STEP, prediction_scores
from src.schemas.paper import DatasetRecord
from src.training.dataset import ProcessedDataset, load_processed_dataset
from src.utils.io import read_json, read_jsonl, resolve_path
from src.utils.logging import get_logger

__all__ = [
    "HELD_OUT_SPLITS",
    "RUN_ID_PATTERN",
    "LoadedRun",
    "PaperEntry",
    "RunStore",
    "RunSummary",
    "RunUnavailableError",
    "TermContribution",
    "is_valid_run_id",
]

logger = get_logger(__name__)

#: A run id names a directory, so it is validated as a token rather than a path.
#: Anything with a separator, a drive letter, or a ``..`` fails this pattern and
#: never reaches a filesystem join (master spec §40).
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Splits the model never saw during fitting. The papers list defaults to these:
#: a prediction on the training split is not evidence of anything, and showing
#: one beside held-out predictions would quietly report training accuracy.
HELD_OUT_SPLITS: tuple[str, ...] = ("val", "test")

#: Bytes in a gigabyte, for the storage meter. Decimal, matching how disk
#: capacity is advertised.
_BYTES_PER_GB = 1_000_000_000


class RunUnavailableError(RuntimeError):
    """Raised when no usable run exists, or a named run cannot be loaded."""


def is_valid_run_id(run_id: str) -> bool:
    """Return whether ``run_id`` is a safe directory token."""
    return bool(RUN_ID_PATTERN.match(run_id))


@dataclass(frozen=True)
class TermContribution:
    """One term's signed push toward the predicted class.

    Attributes:
        term: The n-gram as the vectorizer tokenised it.
        contribution: ``tfidf_value * coefficient`` — a summand of the model's
            own decision function, not a post-hoc attribution.
        tfidf: The term's weight in this document.
    """

    term: str
    contribution: float
    tfidf: float


@dataclass
class PaperEntry:
    """One paper: its dataset record, and the run's prediction for it if any.

    ``prediction`` is ``None`` for training-split papers, which were fitted on
    and therefore have no honest score to show.
    """

    record: DatasetRecord
    split: str
    prediction: dict[str, Any] | None = None

    @property
    def paper_id(self) -> str:
        """Stable identifier, as ingested."""
        return self.record.paper_id

    @property
    def year(self) -> int | None:
        """Publication year from record metadata, when present."""
        value = self.record.meta.get("year")
        return value if isinstance(value, int) else None

    @property
    def n_references(self) -> int:
        """Outgoing reference count recorded at build time."""
        value = self.record.meta.get("n_references")
        return value if isinstance(value, int) else 0

    @property
    def authors_short(self) -> str | None:
        """Display byline, e.g. ``"Zhang, Y. et al."``.

        Composed from the first author plus the author count, both recorded at
        build time. ``None`` when the source supplied no authors, so the UI can
        omit the line instead of printing a placeholder.
        """
        first = self.record.meta.get("first_author")
        if not isinstance(first, str) or not first.strip():
            return None
        count = self.record.meta.get("n_authors")
        if isinstance(count, int) and count > 1:
            return f"{first.strip()} et al."
        return first.strip()


@dataclass(frozen=True)
class RunSummary:
    """Identity and headline result of one run, for the run picker."""

    run_id: str
    created_at: str | None
    finished_at: str | None
    model_name: str | None
    model_display_name: str | None
    primary_metric_name: str | None
    primary_metric_value: float | None
    n_classes: int | None
    split_sizes: dict[str, int]
    is_complete: bool


def _sort_key(summary: RunSummary) -> tuple[str, str]:
    """Order runs newest-first by finish time, falling back to the id."""
    stamp = summary.finished_at or summary.created_at or ""
    return (stamp, summary.run_id)


class LoadedRun:
    """A run's artifacts, loaded once and reused across requests.

    The pipeline, the corpus TF-IDF matrix, and the paper index are built on
    first use rather than at construction, so ``/api/health`` stays answerable
    even when the model file itself is unreadable.
    """

    def __init__(self, run_dir: Path, settings: Settings) -> None:
        """Load a run's manifest and metrics.

        Args:
            run_dir: The ``results/<run_id>/`` directory.
            settings: Resolved configuration, used for the dataset fallback path
                and every threshold this class applies.

        Raises:
            RunUnavailableError: If the manifest or metrics file is unreadable.
        """
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self.settings = settings
        self._lock = threading.Lock()

        try:
            self.manifest: dict[str, Any] = read_json(run_dir / RUN_MANIFEST_NAME)
            self.metrics: dict[str, dict[str, Any]] = read_json(run_dir / METRICS_NAME)
        except (OSError, ValueError) as exc:
            raise RunUnavailableError(f"Run '{self.run_id}' is not readable: {exc}") from exc

        self.warnings: list[str] = []

    # -- identity ----------------------------------------------------------
    @property
    def model_name(self) -> str:
        """Baseline key, e.g. ``tfidf_logreg``."""
        return str(self.manifest.get("model", {}).get("name", "unknown"))

    @property
    def model_display_name(self) -> str:
        """Human-readable model name from ``configs/model.yaml``."""
        return str(self.manifest.get("model", {}).get("display_name", self.model_name))

    @property
    def classes(self) -> list[str]:
        """Class order the run used, as recorded in its manifest."""
        return [str(name) for name in self.manifest.get("labels", {}).get("classes", [])]

    @property
    def primary_split(self) -> str:
        """Split the headline metric describes."""
        return str(self.manifest.get("evaluation", {}).get("primary_split", "val"))

    @property
    def scored_splits(self) -> list[str]:
        """Splits this run wrote predictions for."""
        recorded = self.manifest.get("evaluation", {}).get("splits_scored")
        if isinstance(recorded, list) and recorded:
            return [str(name) for name in recorded]
        return [name for name in HELD_OUT_SPLITS if name in self.metrics]

    @property
    def confidence_kind(self) -> str:
        """``"probability"``, ``"decision"``, or ``"unavailable"``.

        Drives every confidence reading in the UI. ``"decision"`` means the model
        is a LinearSVC and the numbers are uncalibrated margins, not
        probabilities — the distinction is carried through rather than smoothed
        over.
        """
        return str(self.manifest.get("evaluation", {}).get("confidence_kind", "unavailable"))

    @property
    def is_synthetic_corpus(self) -> bool:
        """Whether the run trained on the generated test fixture.

        A dashboard rendering fixture numbers must say so; the corpus is
        separable by construction and its metrics are a wiring check, not a
        result.
        """
        source = str(self.manifest.get("dataset", {}).get("source", ""))
        return "synthetic" in source.lower() or "tests/fixtures" in source.replace("\\", "/")

    def summary(self) -> RunSummary:
        """Return this run's identity and headline metric."""
        evaluation = self.manifest.get("evaluation", {})
        primary = evaluation.get("primary_metric") or {}
        value = primary.get("value")
        return RunSummary(
            run_id=self.run_id,
            created_at=self.manifest.get("created_at"),
            finished_at=self.manifest.get("finished_at"),
            model_name=self.model_name,
            model_display_name=self.model_display_name,
            primary_metric_name=primary.get("name"),
            primary_metric_value=float(value) if isinstance(value, int | float) else None,
            n_classes=self.manifest.get("labels", {}).get("n_classes"),
            split_sizes=dict(self.manifest.get("dataset", {}).get("split_sizes") or {}),
            is_complete=(self.run_dir / MODEL_NAME).is_file(),
        )

    # -- dataset -----------------------------------------------------------
    @cached_property
    def dataset(self) -> ProcessedDataset:
        """The processed dataset this run was trained on.

        The manifest records an absolute directory. It is used only when it still
        looks like a dataset build; otherwise the configured ``processed_dir``
        applies, which is the right answer when a project has been moved or
        cloned elsewhere.

        Raises:
            RunUnavailableError: If neither location holds a readable dataset.
        """
        candidates: list[Path] = []
        recorded = self.manifest.get("dataset", {}).get("directory")
        if isinstance(recorded, str) and recorded:
            candidates.append(Path(recorded))
        candidates.append(self.settings.paths.resolved("processed_dir"))

        for candidate in candidates:
            if (candidate / DATASET_MANIFEST_NAME).is_file():
                try:
                    return load_processed_dataset(candidate)
                except (OSError, ValueError) as exc:
                    logger.warning("api | dataset at %s unusable: %s", candidate, exc)
        raise RunUnavailableError(
            f"Run '{self.run_id}' references dataset '{recorded}', which is not a readable "
            f"dataset build, and no dataset exists at the configured processed_dir. "
            f"Rebuild it with scripts/build_dataset.py."
        )

    @cached_property
    def dataset_is_stale(self) -> bool:
        """Whether the dataset on disk differs from the one the run trained on.

        A mismatch does not make the run wrong, but it does mean the papers the
        API lists are not exactly the papers that produced these metrics, so the
        UI is told and can say so.
        """
        recorded = self.manifest.get("dataset", {}).get("file_hashes") or {}
        current = self.dataset.dataset_hashes
        return any(
            recorded.get(split) and current.get(split) and recorded[split] != current[split]
            for split in set(recorded) | set(current)
        )

    # -- predictions -------------------------------------------------------
    @cached_property
    def predictions(self) -> dict[str, dict[str, dict[str, Any]]]:
        """``{split: {paper_id: prediction_row}}`` as written at train time."""
        by_split: dict[str, dict[str, dict[str, Any]]] = {}
        for split in self.scored_splits:
            path = self.run_dir / predictions_file_name(split)
            if not path.is_file():
                self.warnings.append(f"{path.name} is missing, so '{split}' shows no predictions.")
                continue
            rows: dict[str, dict[str, Any]] = {}
            for row in read_jsonl(path):
                paper_id = row.get("paper_id")
                if isinstance(paper_id, str):
                    rows[paper_id] = row
            by_split[split] = rows
        return by_split

    @cached_property
    def papers(self) -> dict[str, PaperEntry]:
        """Every paper in the corpus, keyed by id, with its prediction if scored.

        Insertion order is held-out splits first, so the default listing shows
        papers that actually have a prediction without needing a sort.
        """
        entries: dict[str, PaperEntry] = {}
        ordered = [*HELD_OUT_SPLITS, "train"]
        for split in [*ordered, *(s for s in self.dataset.splits if s not in ordered)]:
            data = self.dataset.splits.get(split)
            if data is None:
                continue
            rows = self.predictions.get(split, {})
            for record in data.records:
                entries[record.paper_id] = PaperEntry(
                    record=record, split=split, prediction=rows.get(record.paper_id)
                )
        return entries

    def paper(self, paper_id: str) -> PaperEntry | None:
        """Look up one paper by id.

        The lookup is a dictionary hit, never a filesystem join, so a hostile
        identifier can reach nothing outside the loaded corpus.
        """
        return self.papers.get(paper_id)

    def entries(self, splits: Sequence[str] | None = None) -> list[PaperEntry]:
        """Return papers, optionally restricted to the named splits."""
        if splits is None:
            return list(self.papers.values())
        wanted = set(splits)
        return [entry for entry in self.papers.values() if entry.split in wanted]

    # -- review state ------------------------------------------------------
    def needs_review(self, confidence: float | None, kind: str | None = None) -> bool | None:
        """Apply the configured human-review threshold (master spec §15).

        Derived here and sent to the client as a boolean. The client must not
        re-derive it: the threshold would then exist in two places and drift.

        Args:
            confidence: Score for the top label, or ``None`` if unavailable.
            kind: ``"probability"`` or ``"decision"``. Defaults to the run's own
                kind. A margin is compared against the margin threshold, never
                against the probability threshold.

        Returns:
            ``True`` when the prediction warrants review, ``False`` when it does
            not, and ``None`` when the model exposes no score to judge.
        """
        if confidence is None:
            return None
        decision = self.settings.api.decision
        if (kind or self.confidence_kind) == "decision":
            return confidence <= decision.review_margin_threshold
        return confidence <= decision.review_threshold

    # -- model -------------------------------------------------------------
    @cached_property
    def pipeline(self) -> Any:
        """The fitted scikit-learn pipeline.

        Raises:
            RunUnavailableError: If ``model.joblib`` is absent or unreadable.
        """
        path = self.run_dir / MODEL_NAME
        if not path.is_file():
            raise RunUnavailableError(
                f"Run '{self.run_id}' has no {MODEL_NAME}, so it cannot classify new text. "
                f"Re-run training with model.training.save_model enabled."
            )
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - joblib raises many unrelated types
            raise RunUnavailableError(
                f"Could not load {path.name} for run '{self.run_id}': {exc}. "
                f"The file may have been written by a different scikit-learn version "
                f"(this run recorded {self.manifest.get('versions', {}).get('scikit-learn')})."
            ) from exc

    @property
    def vectorizer(self) -> Any:
        """The fitted TF-IDF step of the pipeline."""
        return self.pipeline.named_steps[VECTORIZER_STEP]

    @property
    def classifier(self) -> Any:
        """The fitted classifier step of the pipeline."""
        return self.pipeline.named_steps[CLASSIFIER_STEP]

    def classify(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Run a real forward pass over new text.

        This is the one endpoint that is not a lookup: arbitrary text goes
        through the same fitted vectorizer and classifier the run was evaluated
        with, so the score is produced the same way every reported score was.

        Args:
            texts: Composed inputs, already joined from the configured fields.

        Returns:
            One result per input: predicted label, ranked per-class scores, the
            score kind, and the review flag.
        """
        pipeline = self.pipeline
        predicted = [str(label) for label in pipeline.predict(list(texts))]
        scores, kind = prediction_scores(pipeline, list(texts))
        classes = [str(label) for label in getattr(pipeline, "classes_", self.classes)]

        results: list[dict[str, Any]] = []
        for index, label in enumerate(predicted):
            ranked: list[dict[str, Any]] = []
            confidence: float | None = None
            if scores is not None:
                row = np.asarray(scores[index], dtype=float)
                order = np.argsort(-row)
                ranked = [
                    {"label": classes[position], "score": float(row[position])}
                    for position in order
                    if position < len(classes)
                ]
                best = float(row[order[0]])
                # A probability stands alone; a margin only means something
                # relative to the runner-up, which is what makes it comparable
                # across documents at all.
                confidence = (
                    best
                    if kind == "probability" or row.size < 2
                    else best - float(row[order[1]])
                )
            results.append(
                {
                    "predicted_label": label,
                    "confidence": confidence,
                    "confidence_kind": kind,
                    "scores": ranked,
                    "needs_review": self.needs_review(confidence, kind),
                }
            )
        return results

    # -- similarity --------------------------------------------------------
    @property
    def _corpus_matrix(self) -> tuple[np.ndarray, list[str]]:
        """L2-normalised TF-IDF matrix for the whole corpus, plus its paper ids.

        Normalisation is applied explicitly rather than assumed: ``norm`` is a
        configurable vectorizer parameter, and cosine similarity is only a dot
        product once the rows are unit length.
        """
        with self._lock:
            entries = list(self.papers.values())
            matrix = self.vectorizer.transform([entry.record.text for entry in entries])
            return normalize(matrix, norm="l2", copy=False), [e.paper_id for e in entries]

    def similar(self, paper_id: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Rank the corpus by lexical similarity to one paper.

        Cosine similarity in the run's own fitted TF-IDF space. This measures
        **shared vocabulary**, not learned semantics and not methodological
        equivalence (master spec §17); the caller is responsible for labelling it
        that way, and :mod:`src.api.routers.papers` does.

        Args:
            paper_id: Query paper, which must be in the corpus.
            top_k: Neighbours to return; defaults to the configured value.

        Returns:
            Neighbours above the configured floor, highest score first, never
            including the query itself.
        """
        matrix, ids = self._corpus_matrix
        try:
            position = ids.index(paper_id)
        except ValueError:
            return []

        config = self.settings.api.similarity
        limit = top_k or config.top_k
        scores = (matrix @ matrix[position].T).toarray().ravel()
        scores[position] = -1.0  # a paper is trivially its own best match

        order = np.argsort(-scores)[: max(limit, 0)]
        neighbours: list[dict[str, Any]] = []
        for index in order:
            score = float(scores[index])
            if score < config.min_score:
                break
            entry = self.papers[ids[index]]
            neighbours.append(
                {
                    "paper_id": entry.paper_id,
                    "title": entry.record.title,
                    "score": score,
                    "label": entry.record.label,
                    "split": entry.split,
                }
            )
        return neighbours

    # -- explanation -------------------------------------------------------
    def term_contributions(
        self, text: str, label: str, *, top_k: int | None = None
    ) -> list[TermContribution]:
        """Decompose a linear model's decision for one label into per-term terms.

        For a linear classifier the decision value *is* the sum of
        ``tfidf_value * coefficient`` over the document's terms plus an
        intercept, so these summands are the model's own arithmetic rather than a
        surrogate fitted after the fact. That makes them faithful — and still not
        a causal claim about the paper (master spec §14).

        Args:
            text: The composed document text.
            label: Class whose decision is being decomposed.
            top_k: Terms to return; defaults to the configured value.

        Returns:
            Terms pushing hardest toward ``label``, strongest first. Empty when
            the classifier exposes no coefficients or the label is unknown.
        """
        classifier = self.classifier
        coefficients = getattr(classifier, "coef_", None)
        if coefficients is None:
            return []

        classes = [str(name) for name in getattr(classifier, "classes_", self.classes)]
        if label not in classes:
            return []
        class_index = classes.index(label)

        coefficients = np.asarray(coefficients)
        if coefficients.shape[0] == 1:
            # Binary problems carry one coefficient row, oriented toward
            # classes_[1]; the other class is its negation.
            weights = coefficients[0] if class_index == 1 else -coefficients[0]
        else:
            weights = coefficients[class_index]

        row = self.vectorizer.transform([text])
        names = self.vectorizer.get_feature_names_out()
        indices = row.indices
        values = row.data
        contributions = values * weights[indices]

        limit = top_k or self.settings.api.explanation.top_k_terms
        order = np.argsort(-contributions)[: max(limit, 0)]
        return [
            TermContribution(
                term=str(names[indices[position]]),
                contribution=float(contributions[position]),
                tfidf=float(values[position]),
            )
            for position in order
            if contributions[position] > 0
        ]

    # -- aggregates --------------------------------------------------------
    @cached_property
    def class_counts(self) -> dict[str, int]:
        """Corpus-wide label frequencies, most common first."""
        counts: dict[str, int] = {}
        for entry in self.papers.values():
            if entry.record.label:
                counts[entry.record.label] = counts.get(entry.record.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def counts_by_year(self) -> dict[str, dict[int, int]]:
        """``{label: {year: count}}`` over papers that recorded a real year.

        The dashboard only has one honest time axis here: the publication year
        recorded on each paper. If a record does not carry one, it is skipped
        instead of being assigned to a synthetic bucket, because a fabricated
        year would make the trend chart look complete while being untrue.
        """
        table: dict[str, dict[int, int]] = {}
        for entry in self.papers.values():
            label = entry.record.label
            if not label:
                continue
            year = entry.year
            if year is None:
                continue
            table.setdefault(label, {})
            table[label][year] = table[label].get(year, 0) + 1
        return table

    @cached_property
    def total_references(self) -> int:
        """Sum of outgoing references across the corpus."""
        return sum(entry.n_references for entry in self.papers.values())


class RunStore:
    """Discovers run directories and caches the loaded ones.

    Discovery re-scans on every call so a training run finished after startup is
    picked up without a restart. Loading is cached per run id, because parsing
    predictions and transforming the corpus is far too expensive to repeat per
    request.
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the store to a configuration.

        Args:
            settings: Resolved settings; supplies the results directory and the
                default run id.
        """
        self.settings = settings
        self.results_dir = settings.results_dir
        self._cache: dict[str, LoadedRun] = {}
        self._lock = threading.Lock()

    def _candidate_dirs(self) -> Iterator[Path]:
        """Yield directories under the results root that look like runs."""
        if not self.results_dir.is_dir():
            return
        for child in sorted(self.results_dir.iterdir()):
            if (
                child.is_dir()
                and is_valid_run_id(child.name)
                and (child / RUN_MANIFEST_NAME).is_file()
            ):
                yield child

    def summaries(self) -> list[RunSummary]:
        """Return every discoverable run, newest first."""
        found: list[RunSummary] = []
        for directory in self._candidate_dirs():
            try:
                found.append(self.load(directory.name).summary())
            except RunUnavailableError as exc:
                logger.warning("api | skipping run %s: %s", directory.name, exc)
        return sorted(found, key=_sort_key, reverse=True)

    def load(self, run_id: str) -> LoadedRun:
        """Load one run by id, from cache when already loaded.

        Args:
            run_id: Directory name under the results root.

        Returns:
            The loaded run.

        Raises:
            RunUnavailableError: If the id is malformed, the directory is absent,
                or its manifest cannot be read.
        """
        if not is_valid_run_id(run_id):
            raise RunUnavailableError(
                f"'{run_id}' is not a valid run id. Run ids are the directory names under "
                f"{self.results_dir.name}/ and contain only letters, digits, '.', '_', or '-'."
            )
        with self._lock:
            cached = self._cache.get(run_id)
            if cached is not None:
                return cached

            # Joined only after the id has been validated as a bare token, so a
            # traversal attempt cannot escape the results root (master spec §40).
            directory = self.results_dir / run_id
            if not (directory / RUN_MANIFEST_NAME).is_file():
                raise RunUnavailableError(
                    f"No run '{run_id}' in {self.results_dir}. "
                    f"Available: {[d.name for d in self._candidate_dirs()] or 'none'}"
                )
            run = LoadedRun(directory, self.settings)
            self._cache[run_id] = run
            return run

    def active(self) -> LoadedRun:
        """Return the run the dashboard should show.

        The configured ``default_run_id`` wins when set; otherwise the most
        recently finished run, so a fresh training run appears without a config
        edit.

        Raises:
            RunUnavailableError: If no usable run exists.
        """
        pinned = self.settings.api.runs.default_run_id
        if pinned:
            return self.load(pinned)

        summaries = self.summaries()
        if not summaries:
            raise RunUnavailableError(
                f"No completed training run found in {self.results_dir}. "
                f"Train one first:\n"
                f"    python scripts/build_dataset.py --source data/sample\n"
                f"    python scripts/train_baseline.py --model tfidf_logreg"
            )
        return self.load(summaries[0].run_id)

    def invalidate(self) -> None:
        """Drop every cached run, so the next request re-reads from disk."""
        with self._lock:
            self._cache.clear()

    # -- storage meter -----------------------------------------------------
    def storage_usage(self) -> dict[str, Any]:
        """Measure disk used by the configured data directories.

        Real bytes against an operator-set quota. Unreadable entries are skipped
        rather than aborting the measurement, since a partial figure is more
        useful here than none.
        """
        quota_gb = self.settings.api.storage.quota_gb
        used = 0
        for relative in self.settings.api.storage.measured_dirs:
            directory = resolve_path(relative)
            if not directory.is_dir():
                continue
            used += _directory_size(directory)

        used_gb = used / _BYTES_PER_GB
        return {
            "used_bytes": used,
            "used_gb": round(used_gb, 3),
            "quota_gb": quota_gb,
            "percent": round(min(used_gb / quota_gb, 1.0) * 100, 1),
            "measured": [str(path) for path in self.settings.api.storage.measured_dirs],
        }


def _directory_size(directory: Path) -> int:
    """Return the total size in bytes of every file under ``directory``."""
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue  # vanished mid-walk, or permission denied
    return total
