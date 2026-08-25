"""Loading the processed splits that :mod:`scripts.build_dataset` produced.

Training consumes ``data/processed`` as a *contract*, not as a directory of loose
files. Four properties are checked on load, because each has a failure mode that
would otherwise show up as a plausible-looking but wrong metric:

1. **Presence.** Missing splits produce an actionable error naming the build
   command, rather than an opaque ``FileNotFoundError`` deep in a fit call.
2. **Integrity.** Each split file is re-hashed and compared against
   ``dataset_manifest.json``. A mismatch means the file changed after the build,
   so the manifest's provenance no longer describes what is about to be trained
   on. That is recorded loudly and carried into the run manifest.
3. **Text provenance.** ``record.text`` was composed at build time from
   ``text.fields``. If configuration has changed since, the model trains on the
   *old* field set while the run config advertises the new one. Recorded, not
   silently accepted.
4. **Label coverage.** Labels seen in the splits are checked against the saved
   vocabulary, whose index order is the class order every confusion matrix and
   every saved model uses.

Only presence is fatal. The rest are recorded as findings so an operator sees
them in the log and in ``run_manifest.json`` — a stale-but-usable dataset should
produce a loud run, not a refusal.

**No feature extraction happens here.** Text arrives as text; the vectorizer is
fitted inside the pipeline on the training split alone (master spec §9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from src.data_pipeline.split import (
    DATASET_MANIFEST_NAME,
    LABEL_VOCABULARY_NAME,
    SPLIT_NAMES,
    split_file_name,
)
from src.schemas.paper import DatasetRecord
from src.utils.io import read_json, read_jsonl, resolve_path, sha256_text
from src.utils.logging import get_logger

__all__ = [
    "DatasetIntegrityFinding",
    "DatasetNotFoundError",
    "ProcessedDataset",
    "SplitData",
    "load_processed_dataset",
]

logger = get_logger(__name__)

#: Splits a training run cannot proceed without. ``test`` is optional so a run
#: can proceed on a corpus too small to have produced one.
REQUIRED_SPLITS: tuple[str, ...] = ("train", "val")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the processed dataset is absent or incomplete."""


@dataclass(frozen=True)
class DatasetIntegrityFinding:
    """One non-fatal discrepancy between the dataset on disk and expectations.

    Attributes:
        check: Stable identifier, e.g. ``"file_hash"``.
        severity: ``"warning"`` or ``"info"``.
        detail: Human-readable explanation, written into the run manifest.
    """

    check: str
    severity: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable form for the run manifest."""
        return {"check": self.check, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class SplitData:
    """One split's records, with the parallel arrays a fit call needs.

    ``texts`` and ``labels`` are positionally aligned with ``records``, so a
    prediction at index *i* can always be traced back to its paper.
    """

    name: str
    path: Path
    records: list[DatasetRecord]
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        """Number of records in the split."""
        return len(self.records)

    @property
    def paper_ids(self) -> list[str]:
        """Paper identifiers, aligned with :attr:`texts`."""
        return [record.paper_id for record in self.records]

    @property
    def class_counts(self) -> dict[str, int]:
        """Label frequencies, most common first."""
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass
class ProcessedDataset:
    """Every split plus the provenance needed to describe a training run."""

    directory: Path
    splits: dict[str, SplitData]
    classes: list[str]
    manifest: dict[str, Any]
    findings: list[DatasetIntegrityFinding] = field(default_factory=list)

    def __getitem__(self, name: str) -> SplitData:
        """Return one split by name.

        Raises:
            KeyError: If the split was not loaded.
        """
        try:
            return self.splits[name]
        except KeyError:
            raise KeyError(
                f"Split '{name}' was not loaded. Available: {sorted(self.splits)}"
            ) from None

    def get(self, name: str) -> SplitData | None:
        """Return one split by name, or ``None`` if absent or empty."""
        split = self.splits.get(name)
        return split if split and len(split) else None

    @property
    def split_sizes(self) -> dict[str, int]:
        """Record count per split, in canonical split order."""
        return {name: len(self.splits[name]) for name in SPLIT_NAMES if name in self.splits}

    @property
    def dataset_hashes(self) -> dict[str, str]:
        """Per-split content hashes recorded by the build, for the run manifest."""
        files = self.manifest.get("files", {})
        return {
            name: info.get("sha256", "")
            for name, info in files.items()
            if isinstance(info, dict)
        }

    def findings_as_dicts(self) -> list[dict[str, str]]:
        """Return findings in JSON-serialisable form."""
        return [finding.as_dict() for finding in self.findings]


def _load_split(directory: Path, name: str) -> SplitData:
    """Parse one split file into validated records.

    Raises:
        DatasetNotFoundError: If the file is missing.
        ValueError: If a line fails schema validation — a corrupt or
            stale-schema dataset should not reach a fit call.
    """
    path = directory / split_file_name(name)
    if not path.is_file():
        raise DatasetNotFoundError(f"Missing split file: {path}")

    records: list[DatasetRecord] = []
    for line_number, payload in enumerate(read_jsonl(path), start=1):
        try:
            records.append(DatasetRecord(**payload))
        except PydanticValidationError as exc:
            raise ValueError(
                f"Record {line_number} of {path.name} does not match DatasetRecord. "
                f"The dataset was probably built by an older schema version; rebuild "
                f"it with scripts/build_dataset.py.\n{exc}"
            ) from exc

    # A record without a primary label cannot supply a multi-class target. It is
    # dropped rather than mapped to a placeholder class, which would invent a
    # category the taxonomy does not contain.
    usable = [record for record in records if record.label]
    if len(usable) != len(records):
        logger.warning(
            "data | %s: dropped %d record(s) with no primary label",
            path.name,
            len(records) - len(usable),
        )

    return SplitData(
        name=name,
        path=path,
        records=usable,
        texts=[record.text for record in usable],
        labels=[record.label or "" for record in usable],
    )


def _check_file_hashes(
    directory: Path, manifest: dict[str, Any], loaded: Sequence[str]
) -> list[DatasetIntegrityFinding]:
    """Re-hash each split file and compare against the build manifest.

    The hash is computed the same way the build computed it — over the decoded
    UTF-8 text — so the two are directly comparable on any platform regardless of
    how line endings are stored.
    """
    findings: list[DatasetIntegrityFinding] = []
    files = manifest.get("files") or {}
    for name in loaded:
        recorded = (files.get(name) or {}).get("sha256")
        path = directory / split_file_name(name)
        if not recorded:
            findings.append(
                DatasetIntegrityFinding(
                    "file_hash",
                    "warning",
                    f"{path.name}: manifest records no hash, so integrity is unverified",
                )
            )
            continue
        actual = sha256_text(path.read_text(encoding="utf-8"))
        if actual != recorded:
            findings.append(
                DatasetIntegrityFinding(
                    "file_hash",
                    "warning",
                    f"{path.name}: content changed since the build "
                    f"(manifest {recorded[:12]}…, on disk {actual[:12]}…). "
                    f"Provenance in dataset_manifest.json no longer describes this file.",
                )
            )
    return findings


def _check_text_fields(
    manifest: dict[str, Any], expected_fields: Sequence[str] | None
) -> list[DatasetIntegrityFinding]:
    """Compare the configured input fields against those used at build time."""
    if expected_fields is None:
        return []
    built_with = ((manifest.get("config") or {}).get("text") or {}).get("fields")
    if built_with is None:
        return [
            DatasetIntegrityFinding(
                "text_fields",
                "warning",
                "Manifest does not record which text fields were used; the model will "
                "train on whatever the build composed.",
            )
        ]
    if list(built_with) != list(expected_fields):
        return [
            DatasetIntegrityFinding(
                "text_fields",
                "warning",
                f"Dataset was built from text.fields={list(built_with)} but "
                f"configuration now says {list(expected_fields)}. Training uses the "
                f"text on disk, so the built field set wins. Rebuild the dataset to "
                f"apply the new configuration.",
            )
        ]
    return []


def _check_label_coverage(
    splits: dict[str, SplitData], classes: Sequence[str]
) -> list[DatasetIntegrityFinding]:
    """Verify the saved vocabulary covers every label present in the splits."""
    findings: list[DatasetIntegrityFinding] = []
    observed = {label for split in splits.values() for label in split.labels}
    unknown = sorted(observed - set(classes))
    if unknown:
        findings.append(
            DatasetIntegrityFinding(
                "label_vocabulary",
                "warning",
                f"{len(unknown)} label(s) in the splits are absent from "
                f"{LABEL_VOCABULARY_NAME}: {unknown[:5]}",
            )
        )

    train = splits.get("train")
    if train:
        missing_in_train = sorted(observed - set(train.labels))
        if missing_in_train:
            findings.append(
                DatasetIntegrityFinding(
                    "train_coverage",
                    "warning",
                    f"{len(missing_in_train)} class(es) never appear in train and so "
                    f"can never be predicted: {missing_in_train[:5]}",
                )
            )
    return findings


def load_processed_dataset(
    data_dir: Path | str,
    *,
    expected_text_fields: Sequence[str] | None = None,
    required_splits: Sequence[str] = REQUIRED_SPLITS,
) -> ProcessedDataset:
    """Load processed splits with their manifest, vocabulary, and integrity checks.

    Args:
        data_dir: Directory written by ``scripts/build_dataset.py``.
        expected_text_fields: ``text.fields`` from the active configuration. When
            given, a mismatch against the build is recorded as a finding.
        required_splits: Splits whose absence is fatal.

    Returns:
        A :class:`ProcessedDataset` holding every split that exists on disk.

    Raises:
        DatasetNotFoundError: If the directory, the manifest, the vocabulary, or
            any required split file is missing, or if a required split is empty.
        ValueError: If a record fails schema validation.
    """
    directory = resolve_path(data_dir)
    build_hint = (
        "Build it first:\n"
        "    python scripts/build_dataset.py --source data/sample --out data/processed"
    )

    if not directory.is_dir():
        raise DatasetNotFoundError(f"Processed data directory not found: {directory}\n{build_hint}")

    manifest_path = directory / DATASET_MANIFEST_NAME
    vocabulary_path = directory / LABEL_VOCABULARY_NAME
    for required in (manifest_path, vocabulary_path):
        if not required.is_file():
            raise DatasetNotFoundError(
                f"{directory} is not a complete dataset build: {required.name} is missing.\n"
                f"{build_hint}"
            )

    manifest: dict[str, Any] = read_json(manifest_path)
    vocabulary: dict[str, Any] = read_json(vocabulary_path)
    classes: list[str] = list(vocabulary.get("classes") or [])

    missing = [
        name for name in required_splits if not (directory / split_file_name(name)).is_file()
    ]
    if missing:
        raise DatasetNotFoundError(
            f"{directory} is missing required split file(s): "
            f"{', '.join(split_file_name(name) for name in missing)}.\n{build_hint}"
        )

    splits: dict[str, SplitData] = {}
    for name in SPLIT_NAMES:
        if (directory / split_file_name(name)).is_file():
            splits[name] = _load_split(directory, name)

    for name in required_splits:
        if not len(splits[name]):
            raise DatasetNotFoundError(
                f"Split '{name}' in {directory} contains no usable labelled records. "
                f"A model cannot be trained or selected on an empty split.\n{build_hint}"
            )

    findings = [
        *_check_file_hashes(directory, manifest, list(splits)),
        *_check_text_fields(manifest, expected_text_fields),
        *_check_label_coverage(splits, classes),
    ]

    mode = ((manifest.get("config") or {}).get("labels") or {}).get("mode", "unknown")
    logger.info(
        "data | loaded %s from %s (%d classes, mode=%s)",
        {name: len(split) for name, split in splits.items()},
        directory,
        len(classes),
        mode,
    )
    for finding in findings:
        logger.warning("data | %s: %s", finding.check, finding.detail)

    return ProcessedDataset(
        directory=directory,
        splits=splits,
        classes=classes,
        manifest=manifest,
        findings=findings,
    )
