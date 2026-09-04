"""Unit tests for the dataset validator and preprocessing report (spec §3)."""

import json

import pytest

from src.data_pipeline.dataset_validator import (
    validate_dataset,
    write_preprocessing_report,
)
from src.schemas.paper import DatasetRecord


def _record(paper_id, text, label="NLP", **extra):
    return DatasetRecord(paper_id=paper_id, text=text, title=paper_id, label=label, **extra)


def test_valid_records_pass():
    records = [_record("p1", "A" * 200), _record("p2", "B" * 200)]
    valid, report = validate_dataset(records)
    assert len(valid) == 2
    assert report.invalid_records == 0


def test_missing_label_rejected():
    records = [_record("p1", "A" * 200, label=None)]
    valid, report = validate_dataset(records, label_vocabulary=["NLP"])
    assert len(valid) == 0
    assert report.reason_counts["missing_label"] == 1


def test_invalid_category_rejected():
    records = [_record("p1", "A" * 200, label="NotARegisteredDomain")]
    valid, report = validate_dataset(records, label_vocabulary=["NLP"])
    assert len(valid) == 0
    assert report.reason_counts["invalid_category"] == 1
    assert "NotARegisteredDomain" in report.unknown_labels


def test_duplicate_paper_id_rejected():
    records = [_record("dup", "A" * 200), _record("dup", "B" * 200)]
    valid, report = validate_dataset(records)
    assert len(valid) == 1
    assert report.reason_counts["duplicate_paper_id"] == 1


def test_short_document_rejected():
    records = [_record("p1", "too short")]
    valid, report = validate_dataset(records, min_text_chars=100)
    assert len(valid) == 0
    assert report.reason_counts["too_short"] == 1


def test_malformed_placeholder_rejected():
    records = [_record("p1", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5)]
    valid, report = validate_dataset(records)
    assert len(valid) == 0
    assert report.reason_counts["malformed_text"] == 1


def test_empty_document_rejected():
    records = [_record("p1", "")]
    valid, report = validate_dataset(records)
    assert len(valid) == 0
    assert report.reason_counts["empty_document"] == 1


def test_imbalance_warning():
    records = [
        _record(f"a{i}", "A" * 200, label="Common")
        for i in range(20)
    ] + [
        _record(f"b{i}", "B" * 200, label="Rare")
        for i in range(1)
    ]
    valid, report = validate_dataset(records, imbalance_warn_ratio=5.0)
    assert report.imbalance_ratio == 20.0
    assert any("imbalance" in w.lower() for w in report.warnings)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        validate_dataset([])


def test_write_preprocessing_report(tmp_path):
    records = [_record("p1", "A" * 200), _record("p2", "short")]
    _, report = validate_dataset(records, min_text_chars=100)
    md_path = write_preprocessing_report(report, tmp_path)
    assert md_path.exists()
    assert "Rejected records" in md_path.read_text(encoding="utf-8")
    json_path = tmp_path / "preprocessing_report.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["valid_records"] == 1
    assert data["invalid_records"] == 1