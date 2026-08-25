"""End-to-end: a synthetic corpus through training to a populated run directory.

This is the test that would catch a break no unit test can see — a module whose
own contract holds but which no longer composes with its neighbours. It runs the
real :func:`train_baseline` over the real synthetic dataset for **both** configured
baselines, then inspects what landed on disk.

Everything is offline and writes only into ``tmp_path``: no network, and no run
ever appears in the repository's own ``results/``.

Training is genuinely slow relative to a unit test, so each model is trained once
per session and the resulting directory is inspected by many assertions rather
than retrained for each.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pytest

from src.config.settings import Settings
from src.evaluation.report import (
    CLASS_DISTRIBUTION_STEM,
    CLASSIFICATION_REPORT_NAME,
    CONFUSION_MATRIX_STEM,
    METRICS_NAME,
    MODEL_NAME,
    REPORT_NAME,
    RUN_CONFIG_STEM,
    RUN_MANIFEST_NAME,
    predictions_file_name,
)
from src.training.dataset import ProcessedDataset
from src.training.train_baseline import RunExistsError, TrainingResult, train_baseline
from src.utils.io import PROJECT_ROOT, read_json

MODEL_NAMES = ["tfidf_logreg", "tfidf_svm"]

#: The CLI entry point, invoked as a real subprocess so its exit code is observed
#: the way a shell or a CI job would observe it.
CLI = PROJECT_ROOT / "scripts" / "train_baseline.py"


@pytest.fixture(scope="session", params=MODEL_NAMES)
def trained(
    request: pytest.FixtureRequest,
    settings: Settings,
    synthetic_dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainingResult:
    """One completed training run per configured baseline, built once per session."""
    model_name = request.param
    results_root = tmp_path_factory.mktemp(f"results_{model_name}")
    return train_baseline(
        settings,
        model_name,
        data_dir=synthetic_dataset_dir,
        results_dir=results_root,
        run_id=f"{model_name}-itest",
    )


def _figure_names(settings: Settings) -> tuple[str, str]:
    """Return the confusion-matrix and class-distribution file names."""
    suffix = settings.model.evaluation.plots.figure_format.lstrip(".")
    return f"{CONFUSION_MATRIX_STEM}.{suffix}", f"{CLASS_DISTRIBUTION_STEM}.{suffix}"


# ---------------------------------------------------------------------------
# The artifact contract
# ---------------------------------------------------------------------------
def test_artifact_names_match_the_documented_contract(settings: Settings) -> None:
    """Pins the required file names once, so other tests can use the constants.

    Every other assertion in this module refers to the constants, which would
    happily follow a rename. This test is what stops a rename from quietly
    changing the deliverable.
    """
    assert MODEL_NAME == "model.joblib"
    assert METRICS_NAME == "metrics.json"
    assert CLASSIFICATION_REPORT_NAME == "classification_report.json"
    assert RUN_MANIFEST_NAME == "run_manifest.json"
    assert REPORT_NAME == "report.md"
    assert RUN_CONFIG_STEM == "run_config"
    assert _figure_names(settings) == ("confusion_matrix.png", "class_distribution.png")


def test_every_expected_file_is_written(trained: TrainingResult, settings: Settings) -> None:
    confusion_matrix, class_distribution = _figure_names(settings)
    expected = {
        MODEL_NAME,
        METRICS_NAME,
        CLASSIFICATION_REPORT_NAME,
        RUN_MANIFEST_NAME,
        REPORT_NAME,
        f"{RUN_CONFIG_STEM}.yaml",
        f"{RUN_CONFIG_STEM}.json",
        confusion_matrix,
        class_distribution,
        predictions_file_name(trained.primary_split),
    }
    present = {path.name for path in trained.run_dir.iterdir()}
    assert expected <= present, f"missing: {sorted(expected - present)}"


def test_no_artifact_is_empty(trained: TrainingResult) -> None:
    """A zero-byte file would satisfy ``exists()`` while carrying nothing."""
    for path in trained.run_dir.iterdir():
        if path.is_file():
            assert path.stat().st_size > 0, f"{path.name} is empty"


def test_reported_artifacts_exist_on_disk(trained: TrainingResult) -> None:
    """The manifest's own artifact list must not name a file it did not write."""
    for name in trained.artifacts:
        assert (trained.run_dir / name).is_file(), f"{name} was reported but is absent"


def test_no_temporary_files_are_left_behind(trained: TrainingResult) -> None:
    """Atomic writes use temp files; none should survive a successful run."""
    leftovers = [path.name for path in trained.run_dir.iterdir() if path.suffix == ".tmp"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Metrics on disk
# ---------------------------------------------------------------------------
def test_metrics_file_covers_every_scored_split(trained: TrainingResult) -> None:
    metrics = read_json(trained.run_dir / METRICS_NAME)
    assert set(metrics) == set(trained.metrics)
    assert trained.primary_split in metrics


def test_metrics_on_disk_match_the_returned_result(trained: TrainingResult) -> None:
    """The in-memory result and the persisted file must not diverge."""
    metrics = read_json(trained.run_dir / METRICS_NAME)
    primary = metrics[trained.primary_split]["primary_metric"]
    assert primary["name"] == trained.primary_metric["name"]
    assert primary["value"] == pytest.approx(trained.primary_metric["value"])


def test_macro_f1_is_the_selection_metric(trained: TrainingResult, settings: Settings) -> None:
    """Master spec §13: macro F1 drives model selection, not accuracy."""
    assert trained.primary_metric["name"] == settings.model.evaluation.primary_metric == "macro_f1"


def test_the_model_learns_the_separable_signal(trained: TrainingResult) -> None:
    """A sanity floor, not a benchmark.

    The synthetic corpus plants disjoint marker vocabularies per class, so a
    working TF-IDF pipeline should score far above chance. A low number here means
    the wiring is broken — text misaligned with labels, or the vectorizer fitted on
    the wrong array — rather than that the model is weak.
    """
    assert trained.primary_metric["value"] > 0.5


def test_classification_report_is_written_for_the_primary_split(
    trained: TrainingResult,
) -> None:
    report = read_json(trained.run_dir / CLASSIFICATION_REPORT_NAME)
    assert "accuracy" in report
    assert "macro avg" in report


def test_predictions_are_traceable_to_papers(trained: TrainingResult) -> None:
    """Aggregate metrics cannot support error analysis; this file can."""
    path = trained.run_dir / predictions_file_name(trained.primary_split)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == trained.metrics[trained.primary_split]["n_samples"]
    for row in rows:
        assert row["paper_id"]
        assert row["correct"] == (row["true_label"] == row["predicted_label"])


def test_confidence_provenance_travels_with_the_predictions(
    trained: TrainingResult,
) -> None:
    """LogReg records probabilities; the SVM records a margin, labelled as one."""
    path = trained.run_dir / predictions_file_name(trained.primary_split)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    expected = "probability" if trained.model_name == "tfidf_logreg" else "decision"
    assert first["confidence_kind"] == expected
    assert trained.metrics[trained.primary_split]["confidence"]["kind"] == expected


def test_svm_run_completes_without_probabilities(trained: TrainingResult) -> None:
    """The SVM path must produce a full run despite having no ``predict_proba``."""
    if trained.model_name != "tfidf_svm":
        pytest.skip("specific to the classifier without predict_proba")
    confidence = trained.metrics[trained.primary_split]["confidence"]
    assert confidence["available"] is True
    assert confidence["statistic"] == "top1_minus_top2_margin"
    assert "NOT a probability" in confidence["caveat"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_manifest_records_dataset_identity(
    trained: TrainingResult, synthetic_manifest: dict[str, Any]
) -> None:
    """A metric is only reproducible if the exact input can be identified."""
    dataset = read_json(trained.run_dir / RUN_MANIFEST_NAME)["dataset"]
    assert dataset["split_sizes"]["train"] > 0
    recorded = {name: info["sha256"] for name, info in synthetic_manifest["files"].items()}
    assert dataset["file_hashes"] == recorded


def test_manifest_records_the_run_environment(trained: TrainingResult) -> None:
    manifest = read_json(trained.run_dir / RUN_MANIFEST_NAME)
    assert manifest["run_id"] == trained.run_id
    assert manifest["seed"] == 42
    assert manifest["versions"]["scikit-learn"]
    assert manifest["versions"]["python"].startswith("3.11")
    assert manifest["model"]["n_features"] > 0


def test_manifest_records_a_clean_dataset_as_clean(trained: TrainingResult) -> None:
    """The fixture dataset is freshly built, so there is nothing to report."""
    manifest = read_json(trained.run_dir / RUN_MANIFEST_NAME)
    assert manifest["dataset"]["integrity_findings"] == []


def test_run_config_omits_environment_secrets(trained: TrainingResult) -> None:
    """Master spec §32: a results directory gets zipped and shared.

    ``settings.env`` holds the OpenAlex contact address today and is where an API
    key would live tomorrow, so it must not be serialised into an artifact.
    """
    payload = read_json(trained.run_dir / f"{RUN_CONFIG_STEM}.json")
    assert "env" not in payload
    assert {"app", "dataset", "model"} <= set(payload)

    for suffix in ("yaml", "json"):
        text = (trained.run_dir / f"{RUN_CONFIG_STEM}.{suffix}").read_text(encoding="utf-8").lower()
        assert "mailto" not in text
        assert "api_key" not in text


def test_report_states_the_leakage_guarantee(trained: TrainingResult) -> None:
    """The written summary must say how the numbers were produced."""
    text = (trained.run_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "training split only" in text
    assert trained.run_id in text
    assert "uncalibrated" in text.lower()
    # A baseline must not be presented as the finished system.
    assert "classical baselines" in text


# ---------------------------------------------------------------------------
# The saved model is usable
# ---------------------------------------------------------------------------
def test_saved_model_reloads_and_predicts(
    trained: TrainingResult, processed_dataset: ProcessedDataset
) -> None:
    """A model artifact that cannot be loaded and reused is not a deliverable."""
    pipeline = joblib.load(trained.run_dir / MODEL_NAME)
    texts = processed_dataset[trained.primary_split].texts
    predictions = list(pipeline.predict(texts))
    assert len(predictions) == len(texts)
    assert set(predictions) <= set(trained.manifest["labels"]["classes"])


def test_reloaded_model_reproduces_the_recorded_predictions(
    trained: TrainingResult, processed_dataset: ProcessedDataset
) -> None:
    """Determinism check: the artifact and the run agree, prediction for prediction."""
    pipeline = joblib.load(trained.run_dir / MODEL_NAME)
    split = processed_dataset[trained.primary_split]
    path = trained.run_dir / predictions_file_name(trained.primary_split)
    recorded = [
        json.loads(line)["predicted_label"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert list(pipeline.predict(split.texts)) == recorded


# ---------------------------------------------------------------------------
# Runs never overwrite each other
# ---------------------------------------------------------------------------
def test_a_populated_run_directory_is_never_reused(
    settings: Settings, synthetic_dataset_dir: Path, results_dir: Path
) -> None:
    """Master spec §51: an earlier run's results must survive a repeated id."""
    common = {"data_dir": synthetic_dataset_dir, "results_dir": results_dir, "run_id": "fixed"}
    first = train_baseline(settings, "tfidf_logreg", **common)
    before = {path.name: path.stat().st_mtime_ns for path in first.run_dir.iterdir()}

    with pytest.raises(RunExistsError, match="never overwritten"):
        train_baseline(settings, "tfidf_logreg", **common)

    after = {path.name: path.stat().st_mtime_ns for path in first.run_dir.iterdir()}
    assert after == before, "the failed second run modified the first run's artifacts"


def test_two_runs_coexist_under_distinct_ids(
    settings: Settings, synthetic_dataset_dir: Path, results_dir: Path
) -> None:
    runs = [
        train_baseline(
            settings,
            "tfidf_logreg",
            data_dir=synthetic_dataset_dir,
            results_dir=results_dir,
            run_id=f"run-{index}",
        )
        for index in range(2)
    ]
    assert {path.name for path in results_dir.iterdir()} == {"run-0", "run-1"}
    assert all((run.run_dir / METRICS_NAME).is_file() for run in runs)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_multilabel_mode_refuses_rather_than_reporting_multiclass_numbers(
    settings: Settings, synthetic_dataset_dir: Path, results_dir: Path
) -> None:
    """Multi-label training is not implemented, and must not appear to be."""
    labels = settings.labels.model_copy(update={"mode": "multilabel"})
    multilabel = settings.model_copy(
        update={"app": settings.app.model_copy(update={"labels": labels})}
    )
    with pytest.raises(NotImplementedError, match="multiclass"):
        train_baseline(
            multilabel,
            "tfidf_logreg",
            data_dir=synthetic_dataset_dir,
            results_dir=results_dir,
        )


def test_unknown_model_fails_before_touching_the_dataset(
    settings: Settings, tmp_path: Path, results_dir: Path
) -> None:
    """The model name is validated first, so a typo does not cost a dataset load."""
    with pytest.raises(KeyError, match="tfidf_logreg"):
        train_baseline(
            settings, "no_such_model", data_dir=tmp_path / "absent", results_dir=results_dir
        )


# ---------------------------------------------------------------------------
# The CLI, as a real process
# ---------------------------------------------------------------------------
def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``scripts/train_baseline.py`` in a subprocess."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CLI), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_cli_exits_zero_and_writes_a_run(
    model_name: str, synthetic_dataset_dir: Path, tmp_path: Path, settings: Settings
) -> None:
    """The documented invocation must succeed end to end from a shell."""
    results = tmp_path / "cli_results"
    process = _run_cli(
        "--model", model_name,
        "--data-dir", str(synthetic_dataset_dir),
        "--results-dir", str(results),
        "--seed", "42",
    )
    assert process.returncode == 0, process.stderr
    assert "Run complete" in process.stdout

    run_dirs = list(results.iterdir())
    assert len(run_dirs) == 1
    assert run_dirs[0].name.startswith(model_name)

    confusion_matrix, class_distribution = _figure_names(settings)
    present = {path.name for path in run_dirs[0].iterdir()}
    assert {METRICS_NAME, MODEL_NAME, REPORT_NAME, confusion_matrix, class_distribution} <= present


@pytest.mark.slow
def test_cli_seed_override_reaches_the_manifest(
    synthetic_dataset_dir: Path, tmp_path: Path
) -> None:
    """``--seed`` must override configuration without editing the YAML file."""
    results = tmp_path / "seeded"
    process = _run_cli(
        "--model", "tfidf_logreg",
        "--data-dir", str(synthetic_dataset_dir),
        "--results-dir", str(results),
        "--run-id", "seed-check",
        "--seed", "1234",
    )
    assert process.returncode == 0, process.stderr
    assert read_json(results / "seed-check" / RUN_MANIFEST_NAME)["seed"] == 1234
    # The override is passed to the loader, never written back to configs/.
    committed = (PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    assert "seed: 42" in committed


def test_cli_rejects_an_unknown_model_with_a_helpful_list() -> None:
    process = _run_cli("--model", "not_a_model")
    assert process.returncode != 0
    assert "tfidf_logreg" in process.stderr + process.stdout


def test_cli_requires_the_model_flag() -> None:
    process = _run_cli()
    assert process.returncode != 0
    assert "--model" in process.stderr


def test_cli_reports_a_missing_dataset_without_a_traceback(tmp_path: Path) -> None:
    """An unbuilt dataset is an expected state, so it gets an actionable message."""
    process = _run_cli(
        "--model", "tfidf_logreg",
        "--data-dir", str(tmp_path / "never_built"),
        "--results-dir", str(tmp_path / "results"),
    )
    assert process.returncode != 0
    combined = process.stdout + process.stderr
    assert "build_dataset.py" in combined
    assert "Traceback" not in combined
