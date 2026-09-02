"""The API contract the dashboard is built against.

Every assertion here corresponds to something the frontend does with the payload.
That is the selection rule: if breaking a field would leave the dashboard showing
a wrong number rather than an obvious blank, it is pinned here.

Three of those obligations are worth naming, because they are the ones a
plausible-looking refactor breaks without failing anything else:

* **The review threshold is derived on the server** (master spec §15). The client
  receives ``needs_review`` as a boolean and never re-derives it, so the threshold
  has one home. :func:`test_needs_review_matches_the_configured_threshold` checks
  the flag against the configured number rather than against itself.
* **Training-split rows carry no prediction.** The model was fitted on them, so a
  score there is not evidence. The API returns nulls, and the dashboard renders
  "not scored" — an easy thing to "fix" into a leak.
* **A score is labelled by kind.** A decision margin is not a probability, and
  ``confidence_kind`` is what stops the UI printing "64%" for an unbounded number.

Read alongside ``test_api_errors.py``, which covers what happens when a request is
wrong, unauthorised, oversized, or arrives with nothing trained.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi.testclient import TestClient

from src.api.capabilities import (
    ATTENTION_UNAVAILABLE_REASON,
    EXPLANATION_CAVEAT,
    RAG_UNAVAILABLE_REASON,
    SIMILARITY_CAVEAT,
    SIMILARITY_METHOD,
)
from src.config.settings import Settings
from src.training.dataset import ProcessedDataset
from tests.integration.conftest import (
    LOGREG_RUN_ID,
    RUN_CONFIDENCE_KINDS,
    RUN_MODELS,
    SVM_RUN_ID,
)

#: Capabilities that are false in this milestone by design, each for a reason the
#: UI renders verbatim. Pinned as a set so landing one of them has to be a
#: deliberate edit here rather than a silent change in tone.
UNBUILT_CAPABILITIES = {
    "authentication",
}


def get(client: TestClient, path: str, **params: Any) -> dict[str, Any]:
    """GET a JSON endpoint, asserting it succeeded."""
    response = client.get(path, params=params or None)
    assert response.status_code == 200, f"{path} -> {response.status_code} {response.text}"
    return response.json()


def first_paper_id(client: TestClient, split: str = "held_out") -> str:
    """Return the id of the first paper in a split, for detail-endpoint tests."""
    return get(client, "/api/papers", split=split, limit=1)["items"][0]["paper_id"]


# ---------------------------------------------------------------------------
# System: health, meta, capabilities, runs
# ---------------------------------------------------------------------------
def test_health_needs_no_api_key_and_reports_what_loaded(svm_client: TestClient) -> None:
    """Liveness is answerable and names the run, so a probe can see readiness.

    Deliberately unauthenticated: a monitoring probe that needs the shared secret
    stops working the moment the secret rotates.
    """
    body = get(svm_client, "/api/health")

    assert body["status"] == "ok"
    assert body["run_id"] == SVM_RUN_ID
    assert body["dataset_ready"] is True
    assert body["model_ready"] is True
    assert body["warnings"] == []
    assert body["environment"]  # named, never empty


def test_meta_carries_everything_the_shell_needs_to_render_once(
    svm_client: TestClient, settings: Settings
) -> None:
    """``/api/meta`` is the only blocking request the dashboard makes.

    It therefore has to carry identity, storage, the active run, the capability
    table, and the caveats together. Splitting any of them out would put a second
    request in front of the first paint.
    """
    body = get(svm_client, "/api/meta")

    assert body["app_name"] == settings.app.project.name
    assert body["version"] == settings.app.project.version
    assert body["run"]["run_id"] == SVM_RUN_ID
    assert body["capabilities"], "the UI renders unavailable panels from this table"
    assert body["caveats"], "master spec §14/§15/§17 require standing caveats"

    # No accounts exist. Saying so in the payload is what stops the UI implying a
    # signed-in user (master spec §40).
    assert body["user"]["is_authenticated"] is False

    storage = body["storage"]
    assert storage["used_bytes"] >= 0
    assert storage["quota_gb"] == settings.api.storage.quota_gb
    assert storage["measured"], "the meter names the directories it measured"


def test_every_unavailable_capability_explains_itself(svm_client: TestClient) -> None:
    """An unavailable feature carries a reason; an available one does not need one.

    This is the mechanism the dashboard uses instead of a placeholder chart: it
    prints the server's own sentence. An entry with ``available: false`` and no
    reason would render an empty panel with no explanation.
    """
    capabilities = {entry["key"]: entry for entry in get(svm_client, "/api/meta")["capabilities"]}

    for key, entry in capabilities.items():
        assert entry["label"], f"{key} has no display label"
        if not entry["available"]:
            assert entry["reason"], f"{key} is unavailable with no reason given"
            assert len(entry["reason"]) > 40, f"{key}'s reason is too terse to be useful"

    assert UNBUILT_CAPABILITIES <= set(capabilities)
    for key in UNBUILT_CAPABILITIES:
        assert capabilities[key]["available"] is False, f"{key} claims to be built"

    for key in (
        "corpus",
        "classification",
        "confidence",
        "similarity_lexical",
        "similarity_semantic",
        "trends",
        "section_attention",
        "rag_ask",
        "pdf_upload",
        "comparison",
        "research_gaps",
    ):
        assert capabilities[key]["available"] is True, f"{key} should be available"


def test_meta_declares_the_corpus_synthetic(svm_client: TestClient) -> None:
    """A run trained on the generated corpus says so, first and unmissably.

    These fixtures are separable by construction, so a macro-F1 of 1.0 measures
    the wiring. The caveat is what keeps that number from reading as a result.
    """
    body = get(svm_client, "/api/meta")

    assert body["run"]["dataset"]["is_synthetic"] is True
    assert "generated test corpus" in body["caveats"][0]
    assert body["run"]["dataset"]["is_stale"] is False


def test_run_detail_names_the_confidence_kind_per_model(
    svm_client: TestClient, logreg_client: TestClient
) -> None:
    """The two baselines expose different quantities, and the payload says which.

    The dashboard's confidence column is headed "Margin" or "Confidence" from
    this field, and its bar geometry differs: a probability is a proportion of a
    bounded scale, a margin is signed and unbounded.
    """
    for client, run_id in ((svm_client, SVM_RUN_ID), (logreg_client, LOGREG_RUN_ID)):
        run = get(client, "/api/runs/active")

        assert run["run_id"] == run_id
        assert run["model_name"] == RUN_MODELS[run_id]
        assert run["confidence_kind"] == RUN_CONFIDENCE_KINDS[run_id]
        assert run["model_ready"] is True
        assert run["classes"], "the client colours domains by name from this list"
        assert run["seed"] is not None, "a run that cannot be reproduced is not a result"
        assert run["primary_split"] in run["metrics"]


def test_run_detail_reports_the_headline_metric_by_name(svm_client: TestClient) -> None:
    """Metrics are named, not assumed to be accuracy.

    ``primary_metric`` travels with the number so the stat tile can label itself;
    a UI that hard-coded "Accuracy" would mislabel a macro-F1 run.
    """
    run = get(svm_client, "/api/runs/active")
    primary = run["metrics"][run["primary_split"]]["primary_metric"]

    assert primary["name"]
    assert 0.0 <= primary["value"] <= 1.0

    confidence = run["metrics"][run["primary_split"]]["confidence"]
    assert confidence["kind"] == run["confidence_kind"]
    assert confidence["caveat"], "the confidence statistic explains what it is"


def test_runs_list_marks_exactly_one_active(svm_client: TestClient) -> None:
    """Both trained runs are discoverable and one is flagged as displayed."""
    body = get(svm_client, "/api/runs")
    runs = {entry["run_id"]: entry for entry in body["runs"]}

    assert set(RUN_MODELS) <= set(runs)
    assert body["active_run_id"] == SVM_RUN_ID
    assert [entry["is_active"] for entry in body["runs"]].count(True) == 1
    assert runs[SVM_RUN_ID]["is_active"] is True
    assert all(entry["is_complete"] for entry in body["runs"])


def test_unpinned_configuration_selects_a_real_run(unpinned_client: TestClient) -> None:
    """With no ``default_run_id``, the newest finished run is served.

    This is the path a fresh ``train_baseline.py`` takes: the new run appears
    without a configuration edit.
    """
    body = get(unpinned_client, "/api/meta")

    assert body["run"]["run_id"] in RUN_MODELS
    assert get(unpinned_client, "/api/runs")["active_run_id"] == body["run"]["run_id"]


def test_a_run_can_be_fetched_by_id(svm_client: TestClient) -> None:
    """``/api/runs/{id}`` returns the same detail as ``/api/runs/active`` for it."""
    assert get(svm_client, f"/api/runs/{SVM_RUN_ID}") == get(svm_client, "/api/runs/active")


# ---------------------------------------------------------------------------
# Dashboard aggregates
# ---------------------------------------------------------------------------
def test_stats_returns_the_four_tiles_in_display_order(
    svm_client: TestClient, processed_dataset: ProcessedDataset
) -> None:
    """The stat row is fully server-rendered: value, note, icon, and hue.

    The client positions the tiles and nothing else. In particular the note lines
    are sentences from the server, so the counts they quote cannot disagree with
    the numbers above them.
    """
    body = get(svm_client, "/api/stats")
    tiles = {tile["id"]: tile for tile in body["tiles"]}

    assert [tile["id"] for tile in body["tiles"]] == ["papers", "domains", "score", "review"]
    for tile in body["tiles"]:
        assert tile["label"] and tile["value"] and tile["note"]
        assert tile["icon"] and tile["hue"].startswith("--")

    total = sum(processed_dataset.split_sizes.values())
    held_out = sum(
        count for name, count in processed_dataset.split_sizes.items() if name in ("val", "test")
    )
    assert tiles["papers"]["value"] == f"{total:,}"
    assert f"{held_out:,}" in tiles["papers"]["note"]
    assert tiles["domains"]["value"] == str(len(processed_dataset.classes))


def test_review_tile_counts_only_scored_held_out_predictions(
    svm_client: TestClient, settings: Settings
) -> None:
    """The flagged count is bounded by what was actually scored (master spec §15).

    Held-out only: the model's confidence on the split it was fitted to is
    inflated, and pooling it in would understate how much needs a human look.
    """
    tiles = {tile["id"]: tile for tile in get(svm_client, "/api/stats")["tiles"]}
    rows = get(svm_client, "/api/papers", split="held_out", limit=settings.api.pagination.max_page_size)

    scored = [row for row in rows["items"] if row["needs_review"] is not None]
    flagged = [row for row in scored if row["needs_review"]]

    assert tiles["review"]["value"] == f"{len(flagged):,}"
    assert f"of {len(scored)} scored predictions" in tiles["review"]["note"]


def test_domain_distribution_counts_ground_truth_not_predictions(
    svm_client: TestClient, processed_dataset: ProcessedDataset
) -> None:
    """The donut describes the corpus, and says so in ``basis``.

    Counting predictions instead would draw the model's view of itself, which is
    the number a reader is least able to check. Cross-checked here against the
    dataset's own labels rather than against another endpoint.
    """
    body = get(svm_client, "/api/stats/domains")

    truth = Counter(
        label for split in processed_dataset.splits.values() for label in split.labels
    )
    assert body["total"] == sum(truth.values())
    assert {entry["label"]: entry["count"] for entry in body["slices"]} == dict(truth)
    assert "ground-truth" in body["basis"]
    assert body["unit"] == "papers"
    assert sum(entry["count"] for entry in body["slices"]) == body["total"], "slices must sum"
    assert abs(sum(entry["share"] for entry in body["slices"]) - 1.0) < 0.01


def test_trends_are_corpus_composition_over_time(
    svm_client: TestClient, settings: Settings
) -> None:
    """Years ascend, every series is aligned to them, and the basis is stated.

    A ragged series would silently shift a line's x-position, so the equal-length
    guarantee is what lets the client draw with an index rather than a lookup.
    """
    body = get(svm_client, "/api/research/trends")

    assert body["years"] == sorted(body["years"])
    assert body["series"], "the fixture corpus records publication years"
    for series in body["series"]:
        assert series["label"]
        assert len(series["values"]) == len(body["years"])
        assert all(value >= 0 for value in series["values"])

    assert len(body["series"]) <= settings.api.trends.max_series
    assert body["dropped_series"] == 0
    assert "publication year" in body["basis"]


# ---------------------------------------------------------------------------
# Papers: the table the dashboard is mostly made of
# ---------------------------------------------------------------------------
def test_default_listing_is_held_out_only(svm_client: TestClient) -> None:
    """The default table shows evaluation data, not data the model was fitted to.

    A table mixing the two reads as a results table while quoting training
    accuracy for most of its rows.
    """
    body = get(svm_client, "/api/papers")

    assert body["splits"] == ["val", "test"]
    assert {row["split"] for row in body["items"]} <= {"val", "test"}
    assert body["total"] > 0
    for row in body["items"]:
        assert row["predicted_label"], "a held-out row carries the run's prediction"
        assert row["needs_review"] in (True, False), "and a server-derived review flag"
        assert row["confidence"] is not None
        assert row["correct"] in (True, False)


def test_training_split_rows_carry_no_prediction(svm_client: TestClient) -> None:
    """A paper the model was fitted on is returned unscored, on purpose.

    The run stores predictions for held-out splits only, and the API does not
    quietly re-run the model to fill the column. The dashboard renders these as
    "not scored"; inventing a number here is the leak this test exists to catch.
    """
    body = get(svm_client, "/api/papers", split="train", limit=20)

    assert body["total"] > 0
    for row in body["items"]:
        assert row["split"] == "train"
        assert row["true_label"], "ground truth is known for training rows"
        assert row["predicted_label"] is None
        assert row["confidence"] is None
        assert row["needs_review"] is None
        assert row["correct"] is None


def test_split_filters_partition_the_corpus(
    svm_client: TestClient, processed_dataset: ProcessedDataset
) -> None:
    """``all`` is the whole corpus and the named splits add up to it."""
    totals = {
        name: get(svm_client, "/api/papers", split=name, limit=1)["total"]
        for name in ("all", "train", "val", "test", "held_out")
    }

    assert totals["all"] == sum(processed_dataset.split_sizes.values())
    assert totals["train"] + totals["val"] + totals["test"] == totals["all"]
    assert totals["held_out"] == totals["val"] + totals["test"]


def test_needs_review_matches_the_configured_threshold(
    svm_client: TestClient, logreg_client: TestClient, settings: Settings
) -> None:
    """The flag is the configured threshold applied to the score (master spec §15).

    Checked against the configuration rather than against the payload's own
    consistency, because the failure this guards against is the threshold being
    re-derived somewhere else — in the client, or against the wrong bar. A margin
    compared to the probability threshold would flag almost nothing.
    """
    decision = settings.api.decision
    limit = settings.api.pagination.max_page_size

    margins = get(svm_client, "/api/papers", split="held_out", limit=limit)["items"]
    assert margins
    for row in margins:
        assert row["confidence_kind"] == "decision"
        assert row["needs_review"] is (row["confidence"] <= decision.review_margin_threshold)

    probabilities = get(logreg_client, "/api/papers", split="held_out", limit=limit)["items"]
    assert probabilities
    for row in probabilities:
        assert row["confidence_kind"] == "probability"
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["needs_review"] is (row["confidence"] <= decision.review_threshold)


def test_needs_review_filter_selects_by_the_server_flag(logreg_client: TestClient) -> None:
    """Filtering on the flag partitions the scored rows and nothing else.

    Exercised on the probability run because the margin run flags nothing at all
    on a separable corpus, and a filter that returns an empty set proves little.
    """
    limit = 200
    flagged = get(logreg_client, "/api/papers", needs_review="true", limit=limit)
    clear = get(logreg_client, "/api/papers", needs_review="false", limit=limit)
    everything = get(logreg_client, "/api/papers", limit=limit)

    assert flagged["total"] > 0, "the fixture must contain at least one flagged row"
    assert all(row["needs_review"] is True for row in flagged["items"])
    assert all(row["needs_review"] is False for row in clear["items"])
    assert flagged["total"] + clear["total"] == everything["total"]


def test_search_matches_title_id_and_label_case_insensitively(svm_client: TestClient) -> None:
    """The one search box covers the three fields shown in the table.

    ``query`` is echoed so the empty state can name what was searched for rather
    than saying "no results".
    """
    sample = get(svm_client, "/api/papers", split="all", limit=1)["items"][0]

    by_id = get(svm_client, "/api/papers", split="all", q=sample["paper_id"])
    assert by_id["total"] == 1
    assert by_id["items"][0]["paper_id"] == sample["paper_id"]
    assert by_id["query"] == sample["paper_id"]

    word = sample["title"].split()[0]
    lower = get(svm_client, "/api/papers", split="all", q=word.lower())
    upper = get(svm_client, "/api/papers", split="all", q=word.upper())
    assert lower["total"] == upper["total"] > 0

    by_label = get(svm_client, "/api/papers", split="all", q=sample["true_label"])
    assert by_label["total"] >= 1

    assert get(svm_client, "/api/papers", split="all", q="zzz-no-such-paper")["total"] == 0


def test_pagination_walks_the_corpus_without_gaps_or_repeats(svm_client: TestClient) -> None:
    """Consecutive pages are disjoint and cover everything, with a stable total.

    The dashboard's pager reports "showing 11–20 of 56" from ``offset``,
    ``limit``, and ``total``; if the total moved between pages that label would
    be wrong on every page but the first.
    """
    size = 4
    first = get(svm_client, "/api/papers", split="all", limit=size, offset=0)
    second = get(svm_client, "/api/papers", split="all", limit=size, offset=size)

    assert first["limit"] == size and first["offset"] == 0
    assert second["offset"] == size
    assert first["total"] == second["total"]
    assert len(first["items"]) == len(second["items"]) == size

    ids_first = [row["paper_id"] for row in first["items"]]
    ids_second = [row["paper_id"] for row in second["items"]]
    assert not set(ids_first) & set(ids_second)

    walked: list[str] = []
    offset = 0
    while offset < first["total"]:
        page = get(svm_client, "/api/papers", split="all", limit=size, offset=offset)
        walked.extend(row["paper_id"] for row in page["items"])
        offset += size
    assert len(walked) == first["total"] == len(set(walked))


def test_page_beyond_the_end_is_empty_rather_than_an_error(svm_client: TestClient) -> None:
    """An over-large offset returns no rows and the real total.

    The pager can then recover by clamping, instead of showing an error for what
    is only a stale offset after a filter change.
    """
    body = get(svm_client, "/api/papers", offset=10_000)

    assert body["items"] == []
    assert body["total"] > 0
    assert body["offset"] == 10_000


def test_oversized_limit_is_clamped_not_rejected(svm_client: TestClient, settings: Settings) -> None:
    """A too-large page size is capped, and the response reports the cap.

    Pagination is a hint. Refusing the request would fail a page the client can
    still be served; echoing the granted ``limit`` is what keeps the two in step.
    """
    body = get(svm_client, "/api/papers", split="all", limit=10_000)

    assert body["limit"] == settings.api.pagination.max_page_size
    assert len(body["items"]) <= body["limit"]


def test_default_page_size_comes_from_configuration(svm_client: TestClient, settings: Settings) -> None:
    """An unspecified ``limit`` uses the configured default, not a literal."""
    assert get(svm_client, "/api/papers")["limit"] == settings.api.pagination.default_page_size


# ---------------------------------------------------------------------------
# One paper: detail, similarity, explanation
# ---------------------------------------------------------------------------
def test_paper_detail_extends_the_row_it_came_from(svm_client: TestClient) -> None:
    """Detail is the table row plus the text and the per-class scores.

    Same field names, so the client can render a preview from either payload
    without a second mapping layer.
    """
    row = get(svm_client, "/api/papers", limit=1)["items"][0]
    detail = get(svm_client, f"/api/papers/{row['paper_id']}")

    for key, value in row.items():
        assert detail[key] == value, f"detail disagrees with the row on {key}"

    assert detail["text"], "the preview needs the abstract"
    assert detail["labels"], "secondary labels travel with the record"
    assert detail["predicted_scores"], "a held-out paper has ranked per-class scores"

    scores = [entry["score"] for entry in detail["predicted_scores"]]
    assert scores == sorted(scores, reverse=True), "scores arrive ranked"
    assert detail["predicted_scores"][0]["label"] == detail["predicted_label"]


def test_similar_papers_are_labelled_lexical(svm_client: TestClient, settings: Settings) -> None:
    """Neighbours come back ranked, self-excluded, and described as TF-IDF cosine.

    Master spec §17: shared vocabulary is not methodological equivalence. The
    method name and caveat travel in the payload so the UI cannot present this
    as semantic similarity — which is a separate, unbuilt capability.
    """
    paper_id = first_paper_id(svm_client)
    body = get(svm_client, f"/api/papers/{paper_id}/similar")

    assert body["query_paper_id"] == paper_id
    assert body["method"] == SIMILARITY_METHOD
    assert body["caveat"] == SIMILARITY_CAVEAT
    assert body["items"], "a separable corpus has near neighbours"
    assert len(body["items"]) <= settings.api.similarity.top_k

    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["paper_id"] != paper_id for item in body["items"]), "self is not a neighbour"
    for item in body["items"]:
        assert 0.0 <= item["score"] <= 1.0
        assert item["score"] >= settings.api.similarity.min_score
        assert item["title"] and item["split"]


def test_similar_papers_respects_an_explicit_limit(svm_client: TestClient) -> None:
    """``limit`` narrows the neighbour list."""
    paper_id = first_paper_id(svm_client)
    assert len(get(svm_client, f"/api/papers/{paper_id}/similar", limit=2)["items"]) <= 2


def test_explanation_decomposes_the_model_s_own_decision(
    svm_client: TestClient, settings: Settings
) -> None:
    """Term contributions are the model's decision sum, normalised server-side.

    ``weight`` exists so every client draws identical bars from the same numbers
    instead of inventing a normalisation; ``contribution`` keeps the signed value
    a reader needs. Master spec §14: this describes the classifier, not causality,
    and the caveat says so.
    """
    paper_id = first_paper_id(svm_client)
    body = get(svm_client, f"/api/papers/{paper_id}/explanation")

    assert body["method"] == "linear_term_contributions"
    assert body["predicted_label"]
    assert body["caveat"] == EXPLANATION_CAVEAT
    assert 0 < len(body["terms"]) <= settings.api.explanation.top_k_terms

    # Ranked by signed contribution: the list answers "what pushed toward this
    # label", so a term arguing against it belongs at the bottom, not near the top
    # by virtue of a large magnitude.
    contributions = [term["contribution"] for term in body["terms"]]
    assert contributions == sorted(contributions, reverse=True)

    weights = [term["weight"] for term in body["terms"]]
    assert all(0.0 <= weight <= 1.0 for weight in weights)
    assert max(weights) == 1.0, "the strongest term anchors the bar scale at 1.0"
    for term in body["terms"]:
        assert term["term"]
        assert term["tfidf"] > 0, "a term absent from the text cannot contribute"


def test_explanation_reports_the_raw_decision_value(svm_client: TestClient) -> None:
    """The decision value is returned so the contributions can be checked.

    For a linear model it is the same number the ranked score list reports for
    that label, which is what makes the two panels reconcilable rather than two
    independent claims.
    """
    paper_id = first_paper_id(svm_client)
    explanation = get(svm_client, f"/api/papers/{paper_id}/explanation")
    detail = get(svm_client, f"/api/papers/{paper_id}")

    scores = {entry["label"]: entry["score"] for entry in detail["predicted_scores"]}
    assert explanation["decision_value"] is not None
    assert abs(explanation["decision_value"] - scores[explanation["predicted_label"]]) < 1e-6


def test_section_attention_returns_section_weights(svm_client: TestClient) -> None:
    """The explanation endpoint returns active section attention weights for canonical sections."""
    paper_id = first_paper_id(svm_client)
    attention = get(svm_client, f"/api/papers/{paper_id}/explanation")["section_attention"]

    assert attention["available"] is True
    assert len(attention["sections"]) > 0
    for sec in attention["sections"]:
        assert sec["name"]
        assert sec["canonical_name"]
        assert 0.0 <= sec["weight"] <= 1.0


def test_ask_endpoint_returns_passage_evidence_answer(svm_client: TestClient) -> None:
    """POST /papers/{id}/ask returns passage-grounded Q&A answers."""
    paper_id = first_paper_id(svm_client)
    response = svm_client.post(
        f"/api/papers/{paper_id}/ask",
        json={"question": "What is the main topic or dataset of this paper?"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["paper_id"] == paper_id
    assert body["question"] == "What is the main topic or dataset of this paper?"
    assert body["answer"]
    assert body["confidence"] >= 0.0


def test_explanation_works_for_a_training_paper_and_says_it_is_a_fit(
    svm_client: TestClient,
) -> None:
    """A training-split paper is explainable, and the run stores no prediction for it.

    Both facts hold at once, and together they are why the dashboard labels this
    panel as decomposing a fit: the endpoint runs the model live, so it produces a
    label the listing deliberately leaves null.
    """
    paper_id = first_paper_id(svm_client, split="train")
    row = get(svm_client, "/api/papers", split="train", q=paper_id)["items"][0]
    body = get(svm_client, f"/api/papers/{paper_id}/explanation")

    assert row["predicted_label"] is None, "the listing reports no stored prediction"
    assert body["predicted_label"], "the live decomposition still has a label to explain"
    assert body["terms"]


# ---------------------------------------------------------------------------
# Classification: the one endpoint that runs the model on new text
# ---------------------------------------------------------------------------
def test_classify_runs_a_real_forward_pass(
    svm_client: TestClient, class_markers: dict[str, tuple[str, ...]]
) -> None:
    """Submitted text is scored by the run's own fitted pipeline.

    Asserted through a marker vocabulary the generator guarantees is unique to one
    class: a stub returning the majority label would pass a shape check but not
    this one.
    """
    label, markers = next(iter(class_markers.items()))
    response = svm_client.post(
        "/api/papers/classify",
        json={"title": " ".join(markers[:3]), "abstract": " ".join(markers)},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["result"]["predicted_label"] == label
    assert body["run_id"] == SVM_RUN_ID
    assert body["model_display_name"]

    scores = body["result"]["scores"]
    assert [entry["score"] for entry in scores] == sorted(
        (entry["score"] for entry in scores), reverse=True
    )
    assert scores[0]["label"] == body["result"]["predicted_label"]


def test_classify_labels_its_score_by_kind(
    svm_client: TestClient, logreg_client: TestClient
) -> None:
    """The caveat matches the quantity, so no client prints a margin as a percent.

    Master spec §32's concern in miniature: the distinction between a probability
    and an unbounded margin lives in the payload, not in the client's assumptions
    about which model is loaded.
    """
    payload = {"title": "Neural network architecture", "abstract": "convolution transformer"}

    margin = svm_client.post("/api/papers/classify", json=payload).json()
    assert margin["result"]["confidence_kind"] == "decision"
    assert "margin" in margin["caveat"]
    assert "meaningless as a percentage" in margin["caveat"]

    probability = logreg_client.post("/api/papers/classify", json=payload).json()
    result = probability["result"]
    assert result["confidence_kind"] == "probability"
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(entry["score"] for entry in result["scores"]) - 1.0) < 1e-6
    assert "uncalibrated" in probability["caveat"]


def test_classify_derives_the_review_flag_server_side(
    logreg_client: TestClient, settings: Settings
) -> None:
    """An ad-hoc classification is flagged by the same threshold as a stored one."""
    body = logreg_client.post(
        "/api/papers/classify", json={"title": "ambiguous", "abstract": "generic method paper"}
    ).json()["result"]

    expected = body["confidence"] <= settings.api.decision.review_threshold
    assert body["needs_review"] is expected


# ---------------------------------------------------------------------------
# Same-origin hosting
# ---------------------------------------------------------------------------
def test_the_dashboard_is_served_from_the_api_origin(svm_client: TestClient) -> None:
    """``GET /`` returns the dashboard shell and its assets are reachable.

    Same-origin hosting is what makes the default configuration CORS-free: a
    same-origin fetch is not a cross-origin request, so the "allow ``*`` while
    debugging" mistake has no opportunity to happen (master spec §40).
    """
    index = svm_client.get("/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert "<div id=\"app\"" in index.text or "<body" in index.text

    for asset in ("/js/app.js", "/css/app.css"):
        response = svm_client.get(asset)
        assert response.status_code == 200, f"{asset} is referenced by index.html"


def test_the_static_mount_never_shadows_an_api_route(svm_client: TestClient) -> None:
    """Mounting the frontend at ``/`` leaves ``/api`` intact.

    The mount is added last for exactly this reason; added first it would answer
    every request and the API would appear to have no routes at all.
    """
    assert get(svm_client, "/api/health")["status"] == "ok"
    assert svm_client.get("/api/openapi.json").status_code == 200
    assert svm_client.get("/api/docs").status_code == 200


def test_openapi_documents_the_json_surface(svm_client: TestClient) -> None:
    """Every endpoint the dashboard calls is in the schema, under the prefix."""
    paths = get(svm_client, "/api/openapi.json")["paths"]

    for path in (
        "/api/health",
        "/api/meta",
        "/api/runs",
        "/api/stats",
        "/api/stats/domains",
        "/api/research/trends",
        "/api/papers",
        "/api/papers/{paper_id}",
        "/api/papers/{paper_id}/similar",
        "/api/papers/{paper_id}/explanation",
        "/api/papers/{paper_id}/ask",
        "/api/papers/classify",
    ):
        assert path in paths, f"{path} is undocumented"
