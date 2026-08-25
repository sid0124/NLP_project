"""Fixtures for the API contract tests.

Building a client means training a real run. The API reads run artifacts, so a
test against a hand-written manifest would prove nothing about what the dashboard
actually receives — it would only prove that a fixture matches itself.

Two runs are trained once per session, one per confidence kind. That is not
duplication for its own sake: LinearSVC's unbounded decision margin and logistic
regression's probability take different paths through the review threshold, the
``confidence_kind`` field, and the classification caveat, and the frontend renders
them with different geometry. A suite that only ever saw one of them would let the
other break silently.

Everything is offline and writes only under ``tmp_path_factory``, so no test run
ever leaves a directory in the repository's own ``results/``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_settings, get_store
from src.api.runstore import RunStore
from src.config.settings import Settings
from src.evaluation.report import MODEL_NAME
from src.training.train_baseline import train_baseline

#: Run ids for the two fixture runs. Named rather than timestamped so a test can
#: pin one and assertions about run identity stay readable.
SVM_RUN_ID = "api-svm"
LOGREG_RUN_ID = "api-logreg"

#: Which model each fixture run was trained with, and the confidence kind it
#: therefore reports. Pinned here because the whole point of two runs is that
#: these differ.
RUN_MODELS = {SVM_RUN_ID: "tfidf_svm", LOGREG_RUN_ID: "tfidf_logreg"}
RUN_CONFIDENCE_KINDS = {SVM_RUN_ID: "decision", LOGREG_RUN_ID: "probability"}

#: A client builder: ``build(run_id, results_dir=..., security=..., env=...)``.
ClientFactory = Callable[..., TestClient]


def _without_plots(base: Settings) -> Settings:
    """Return settings with figure rendering disabled.

    The API never reads a PNG, and matplotlib rendering is the slowest part of a
    training run. Turning it off here keeps the fixture cheap without changing a
    single number the API serves.
    """
    plots = base.model.evaluation.plots.model_copy(
        update={"confusion_matrix": False, "class_distribution": False}
    )
    evaluation = base.model.evaluation.model_copy(update={"plots": plots})
    return base.model_copy(update={"model": base.model.model_copy(update={"evaluation": evaluation})})


@pytest.fixture(scope="session")
def api_results_dir(
    tmp_path_factory: pytest.TempPathFactory,
    settings: Settings,
    synthetic_dataset_dir: Path,
) -> Path:
    """A results root holding one completed run per confidence kind."""
    root = tmp_path_factory.mktemp("api_results")
    fast = _without_plots(settings)
    for run_id, model_name in RUN_MODELS.items():
        train_baseline(
            fast,
            model_name,
            data_dir=synthetic_dataset_dir,
            results_dir=root,
            run_id=run_id,
        )
    return root


@pytest.fixture(scope="session")
def api_client_factory(settings: Settings, api_results_dir: Path) -> Iterator[ClientFactory]:
    """Return a builder for API clients wired to a temporary results root.

    Dependency overrides rather than :func:`src.api.deps.reset_caches`: clearing
    the cache would only make the app reload ``configs/``, which points at the
    repository's real results directory. Overriding is also what keeps the
    process-wide cache untouched, so an API test cannot perturb an unrelated one.

    The store is built once per configuration and shared by every request from
    that client, exactly as it is in the running server — a fresh store per
    request would re-parse predictions and re-transform the corpus each time.
    """
    clients: list[TestClient] = []

    def build(
        run_id: str | None = None,
        *,
        results_dir: Path | None = None,
        security: dict[str, object] | None = None,
        env: dict[str, object] | None = None,
    ) -> TestClient:
        runs = settings.api.runs.model_copy(
            update={"results_dir": results_dir or api_results_dir, "default_run_id": run_id}
        )
        api_update: dict[str, object] = {"runs": runs}
        if security:
            api_update["security"] = settings.api.security.model_copy(update=security)

        config = settings.model_copy(update={"api": settings.api.model_copy(update=api_update)})
        if env:
            config = config.model_copy(update={"env": settings.env.model_copy(update=env)})

        app = create_app(config)
        store = RunStore(config)
        app.dependency_overrides[get_settings] = lambda: config
        app.dependency_overrides[get_store] = lambda: store

        client = TestClient(app)
        clients.append(client)
        return client

    yield build

    for client in clients:
        client.close()


@pytest.fixture(scope="session")
def svm_client(api_client_factory: ClientFactory) -> TestClient:
    """A client pinned to the LinearSVC run, whose scores are decision margins."""
    return api_client_factory(SVM_RUN_ID)


@pytest.fixture(scope="session")
def logreg_client(api_client_factory: ClientFactory) -> TestClient:
    """A client pinned to the logistic-regression run, whose scores are probabilities."""
    return api_client_factory(LOGREG_RUN_ID)


@pytest.fixture(scope="session")
def unpinned_client(api_client_factory: ClientFactory) -> TestClient:
    """A client with no ``default_run_id``, so the store picks the active run itself."""
    return api_client_factory(None)


@pytest.fixture(scope="session")
def no_run_client(
    api_client_factory: ClientFactory, tmp_path_factory: pytest.TempPathFactory
) -> TestClient:
    """A client whose results root is empty — the state of a fresh clone."""
    return api_client_factory(None, results_dir=tmp_path_factory.mktemp("api_no_runs"))


@pytest.fixture(scope="session")
def modelless_client(
    api_client_factory: ClientFactory,
    api_results_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> TestClient:
    """A client whose run has metrics and predictions but no saved model.

    Reachable in practice: ``model.training.save_model`` can be off, and a run
    directory can be copied without its largest file. The dashboard must still
    render everything that does not need a forward pass.
    """
    root = tmp_path_factory.mktemp("api_modelless")
    shutil.copytree(api_results_dir / SVM_RUN_ID, root / SVM_RUN_ID)
    (root / SVM_RUN_ID / MODEL_NAME).unlink()
    return api_client_factory(SVM_RUN_ID, results_dir=root)
