"""Shared pytest fixtures.

Every fixture here is **offline**. Nothing in the suite touches the network, so
tests are runnable on a plane, in CI without credentials, and identically on
every machine (master spec §51). The two data sources are:

* ``tests/fixtures/openalex_sample.jsonl`` — a committed corpus of messy payloads
  for parsing, validation, and dedup tests.
* ``tests/fixtures/synthetic_corpus.py`` — a generated, separable corpus for
  split and training mechanics.

The generated dataset is built once per session and shared read-only, because
writing it runs the real labelling and splitting stages and there is no reason to
pay that per test. Anything that needs to *mutate* a dataset copies it into its
own ``tmp_path`` first.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings, load_settings  # noqa: E402
from src.data_pipeline.split import DATASET_MANIFEST_NAME  # noqa: E402
from src.training.dataset import ProcessedDataset, load_processed_dataset  # noqa: E402
from src.utils.io import read_json  # noqa: E402
from tests.fixtures.synthetic_corpus import (  # noqa: E402
    CLASS_MARKERS,
    write_synthetic_dataset,
)

#: Records generated per class. Small enough that the whole suite stays fast,
#: large enough that a 70/15/15 stratified split puts every class in every split.
PER_CLASS = 20

#: Relaxed class-size floor for the fixture. Production uses 40 (see
#: ``configs/config.yaml``); a fixture that large would make every test slow for
#: no added coverage.
FIXTURE_MIN_CLASS_COUNT = 5


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Project settings loaded from ``configs/``.

    Session-scoped and treated as read-only. A test that needs a different
    configuration derives one with ``model_copy(update=...)`` rather than
    mutating this instance, so ordering between tests never matters.
    """
    return load_settings()


@pytest.fixture(scope="session")
def synthetic_dataset_dir(tmp_path_factory: pytest.TempPathFactory, settings: Settings) -> Path:
    """A processed dataset on disk, in the exact layout a real build produces.

    Built through the production labelling and splitting stages, so a test that
    passes against this fixture is exercising the same code path a real dataset
    takes — not a hand-written approximation of it.
    """
    directory = tmp_path_factory.mktemp("processed")
    write_synthetic_dataset(
        directory,
        settings,
        per_class=PER_CLASS,
        min_class_count=FIXTURE_MIN_CLASS_COUNT,
    )
    return directory


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_dataset_dir: Path) -> dict[str, Any]:
    """The dataset manifest written alongside the synthetic splits."""
    return read_json(synthetic_dataset_dir / DATASET_MANIFEST_NAME)


@pytest.fixture(scope="session")
def processed_dataset(synthetic_dataset_dir: Path, settings: Settings) -> ProcessedDataset:
    """The synthetic dataset loaded through the training loader."""
    return load_processed_dataset(
        synthetic_dataset_dir, expected_text_fields=settings.app.text.fields
    )


@pytest.fixture
def mutable_dataset_dir(synthetic_dataset_dir: Path, tmp_path: Path) -> Path:
    """A per-test copy of the synthetic dataset, safe to modify.

    Corruption tests need to edit a split file or a manifest. Doing that to the
    session fixture would leak into every later test, so they get a copy.
    """
    destination = tmp_path / "processed"
    shutil.copytree(synthetic_dataset_dir, destination)
    return destination


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """An empty results root, so a test run never writes into ``results/``."""
    directory = tmp_path / "results"
    directory.mkdir()
    return directory


@pytest.fixture(scope="session")
def class_markers() -> dict[str, tuple[str, ...]]:
    """Per-class marker vocabulary used by the synthetic corpus.

    Exposed so a test can assert on features the generator guaranteed, rather
    than re-deriving them from the generated text.
    """
    return CLASS_MARKERS


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close any matplotlib figures a test leaves open.

    Plot tests that fail mid-render would otherwise accumulate open figures and
    trip matplotlib's warning about too many open figures, turning one real
    failure into a wall of unrelated noise.

    Checked through ``sys.modules`` rather than by importing: a test that never
    plots should not pay matplotlib's import cost, nor surface its third-party
    deprecation warnings in an unrelated test's report.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")
