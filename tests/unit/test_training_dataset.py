"""Loading the processed splits a dataset build produced.

The loader is the boundary between "files on disk" and "arrays a model fits on",
so these tests pin both halves of its contract: what it must refuse outright, and
what it must merely record. A stale dataset should produce a *loud* run, not a
refusal, and the difference between those two behaviours is easy to regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.data_pipeline.split import DATASET_MANIFEST_NAME, LABEL_VOCABULARY_NAME, split_file_name
from src.training.dataset import (
    DatasetNotFoundError,
    ProcessedDataset,
    load_processed_dataset,
)
from src.utils.io import read_json, write_json


def _rewrite_first_record(path: Path, **changes: object) -> None:
    """Edit the first record of a JSONL split in place."""
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record.update(changes)
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
def test_loads_every_split_written_by_the_build(processed_dataset: ProcessedDataset) -> None:
    assert set(processed_dataset.splits) == {"train", "val", "test"}
    assert all(len(split) for split in processed_dataset.splits.values())


def test_parallel_arrays_stay_aligned(processed_dataset: ProcessedDataset) -> None:
    for split in processed_dataset.splits.values():
        assert len(split.texts) == len(split.labels) == len(split.records) == len(split)
        assert len(split.label_sets) == len(split.records)
        for index, record in enumerate(split.records):
            assert split.texts[index] == record.text
            assert split.labels[index] == record.label
            assert split.label_sets[index] == record.labels


def test_classes_come_from_the_vocabulary_file(
    processed_dataset: ProcessedDataset, synthetic_dataset_dir: Path
) -> None:
    """Class order is the on-disk contract, not whatever a split happens to hold.

    Confusion-matrix indices and a saved model's ``classes_`` both depend on it, so
    it must come from the vocabulary file rather than being re-derived per split.
    """
    saved = read_json(synthetic_dataset_dir / LABEL_VOCABULARY_NAME)["classes"]
    assert processed_dataset.classes == saved
    assert processed_dataset.classes == sorted(processed_dataset.classes)


def test_every_class_reaches_every_split(processed_dataset: ProcessedDataset) -> None:
    """Stratification held, so per-class metrics rest on non-zero support."""
    for split in processed_dataset.splits.values():
        assert set(split.labels) == set(processed_dataset.classes)


def test_no_paper_appears_in_two_splits(processed_dataset: ProcessedDataset) -> None:
    seen: set[str] = set()
    for split in processed_dataset.splits.values():
        ids = set(split.paper_ids)
        assert not (ids & seen), "a paper leaked across splits"
        seen |= ids


def test_split_sizes_use_canonical_order(processed_dataset: ProcessedDataset) -> None:
    assert list(processed_dataset.split_sizes) == ["train", "val", "test"]
    assert sum(processed_dataset.split_sizes.values()) == 80


def test_class_counts_are_descending(processed_dataset: ProcessedDataset) -> None:
    counts = list(processed_dataset["train"].class_counts.values())
    assert counts == sorted(counts, reverse=True)


def test_dataset_hashes_exposed_for_the_run_manifest(
    processed_dataset: ProcessedDataset,
) -> None:
    hashes = processed_dataset.dataset_hashes
    assert set(hashes) == {"train", "val", "test"}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_clean_dataset_produces_no_findings(processed_dataset: ProcessedDataset) -> None:
    assert processed_dataset.findings == []


def test_indexing_and_get(processed_dataset: ProcessedDataset) -> None:
    assert processed_dataset["train"].name == "train"
    assert processed_dataset.get("train") is not None
    assert processed_dataset.get("nonexistent") is None
    with pytest.raises(KeyError, match="Available"):
        processed_dataset["nonexistent"]


# ---------------------------------------------------------------------------
# Fatal conditions — a run must not start
# ---------------------------------------------------------------------------
def test_missing_directory_names_the_build_command(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="build_dataset.py") as exc:
        load_processed_dataset(tmp_path / "absent")
    assert "--source" in str(exc.value)


@pytest.mark.parametrize("removed", [DATASET_MANIFEST_NAME, LABEL_VOCABULARY_NAME])
def test_incomplete_build_is_fatal(mutable_dataset_dir: Path, removed: str) -> None:
    (mutable_dataset_dir / removed).unlink()
    with pytest.raises(DatasetNotFoundError, match=removed):
        load_processed_dataset(mutable_dataset_dir)


@pytest.mark.parametrize("split", ["train", "val"])
def test_missing_required_split_is_fatal(mutable_dataset_dir: Path, split: str) -> None:
    (mutable_dataset_dir / split_file_name(split)).unlink()
    with pytest.raises(DatasetNotFoundError, match=split_file_name(split)):
        load_processed_dataset(mutable_dataset_dir)


def test_absent_test_split_is_tolerated(mutable_dataset_dir: Path) -> None:
    """``test`` is optional: a corpus too small to yield one can still be trained."""
    (mutable_dataset_dir / split_file_name("test")).unlink()
    dataset = load_processed_dataset(mutable_dataset_dir)
    assert set(dataset.splits) == {"train", "val"}
    assert dataset.get("test") is None


def test_empty_required_split_is_fatal(mutable_dataset_dir: Path) -> None:
    (mutable_dataset_dir / split_file_name("val")).write_text("", encoding="utf-8")
    with pytest.raises(DatasetNotFoundError, match="no usable labelled records"):
        load_processed_dataset(mutable_dataset_dir)


def test_schema_drift_is_fatal_and_names_the_fix(mutable_dataset_dir: Path) -> None:
    """A record the schema rejects must never reach a fit call."""
    _rewrite_first_record(mutable_dataset_dir / split_file_name("val"), unexpected_field=1)
    with pytest.raises(ValueError, match="build_dataset.py"):
        load_processed_dataset(mutable_dataset_dir)


# ---------------------------------------------------------------------------
# Non-fatal findings — the run proceeds, loudly
# ---------------------------------------------------------------------------
def _finding_checks(dataset: ProcessedDataset) -> set[str]:
    return {finding.check for finding in dataset.findings}


def test_edited_split_file_records_a_hash_finding(mutable_dataset_dir: Path) -> None:
    """Content that changed after the build makes the manifest's provenance stale."""
    _rewrite_first_record(mutable_dataset_dir / split_file_name("val"), title="edited by hand")
    dataset = load_processed_dataset(mutable_dataset_dir)
    assert "file_hash" in _finding_checks(dataset)
    detail = next(f.detail for f in dataset.findings if f.check == "file_hash")
    assert "content changed since the build" in detail
    # Non-fatal: the data is still loaded and usable.
    assert len(dataset["val"]) > 0


def test_manifest_without_hashes_records_a_finding(mutable_dataset_dir: Path) -> None:
    manifest = read_json(mutable_dataset_dir / DATASET_MANIFEST_NAME)
    for info in manifest["files"].values():
        info.pop("sha256")
    write_json(mutable_dataset_dir / DATASET_MANIFEST_NAME, manifest)
    dataset = load_processed_dataset(mutable_dataset_dir)
    assert "file_hash" in _finding_checks(dataset)
    assert "integrity is unverified" in " ".join(f.detail for f in dataset.findings)


def test_text_field_drift_is_recorded_not_fatal(
    synthetic_dataset_dir: Path, settings: Settings
) -> None:
    """Config changed since the build: the text on disk wins, and the run says so."""
    drifted = [*settings.app.text.fields, "keywords"]
    dataset = load_processed_dataset(synthetic_dataset_dir, expected_text_fields=drifted)
    assert "text_fields" in _finding_checks(dataset)
    detail = next(f.detail for f in dataset.findings if f.check == "text_fields")
    assert "the built field set wins" in detail


def test_matching_text_fields_produce_no_finding(
    synthetic_dataset_dir: Path, settings: Settings
) -> None:
    dataset = load_processed_dataset(
        synthetic_dataset_dir, expected_text_fields=settings.app.text.fields
    )
    assert "text_fields" not in _finding_checks(dataset)


def test_label_outside_the_vocabulary_is_recorded(mutable_dataset_dir: Path) -> None:
    vocabulary = read_json(mutable_dataset_dir / LABEL_VOCABULARY_NAME)
    vocabulary["classes"] = vocabulary["classes"][:-1]
    write_json(mutable_dataset_dir / LABEL_VOCABULARY_NAME, vocabulary)
    dataset = load_processed_dataset(mutable_dataset_dir)
    assert "label_vocabulary" in _finding_checks(dataset)


def test_class_missing_from_train_is_recorded(mutable_dataset_dir: Path) -> None:
    """A class absent from train can never be predicted — say so before training."""
    path = mutable_dataset_dir / split_file_name("train")
    dropped = "Software"
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["label"] != dropped
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    dataset = load_processed_dataset(mutable_dataset_dir)
    assert "train_coverage" in _finding_checks(dataset)
    assert dropped in " ".join(f.detail for f in dataset.findings)


def test_records_without_a_primary_label_are_dropped(mutable_dataset_dir: Path) -> None:
    """No placeholder class is invented for an unlabelled record."""
    path = mutable_dataset_dir / split_file_name("val")
    before = len(path.read_text(encoding="utf-8").splitlines())
    _rewrite_first_record(path, label=None)

    dataset = load_processed_dataset(mutable_dataset_dir)
    assert len(dataset["val"]) == before - 1
    assert all(label for label in dataset["val"].labels)
