"""Unit tests for the universal dataset loader (spec §3)."""

import json

import pytest

from src.data_pipeline.dataset_loader import (
    DatasetLoadError,
    load_csv,
    load_dataset,
    load_json,
    load_jsonl,
)


def test_load_csv(tmp_path):
    path = tmp_path / "papers.csv"
    path.write_text(
        "paper_id,title,abstract,label,year\n"
        "p1,Attention for NLP,We study attention models.,Artificial Intelligence,2020\n"
        "p2,Graph Nets,We propose graph networks.,Software,2021\n",
        encoding="utf-8",
    )
    records = load_csv(path)
    assert len(records) == 2
    assert records[0].paper_id == "p1"
    assert records[0].label == "Artificial Intelligence"
    assert records[0].year == 2020
    assert "attention" in records[0].text.lower()


def test_load_json(tmp_path):
    path = tmp_path / "papers.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "paper_id": "p1",
                        "title": "A Paper",
                        "abstract": "An abstract with enough words for the model.",
                        "label": "NLP",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = load_json(path)
    assert len(records) == 1
    assert records[0].paper_id == "p1"
    assert records[0].label == "NLP"


def test_load_jsonl(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text(
        '{"paper_id": "p1", "title": "One", "abstract": "Body text with words.", "label": "A"}\n'
        '{"paper_id": "p2", "title": "Two", "abstract": "More body text.", "label": "B"}\n',
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[1].paper_id == "p2"


def test_load_dataset_infers_format(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("paper_id,title,abstract,label\np1,T,A.,A\n", encoding="utf-8")
    assert load_dataset(path)[0].paper_id == "p1"


def test_load_dataset_unknown_format_raises(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_text("not really parquet", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_dataset(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_load_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_csv(path)


def test_domain_column_falls_back_to_label(tmp_path):
    path = tmp_path / "papers.csv"
    path.write_text(
        "paper_id,title,abstract,domain\np1,T,Body text here.,Computer Vision\n",
        encoding="utf-8",
    )
    record = load_csv(path)[0]
    assert record.label == "Computer Vision"