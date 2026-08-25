"""What the API does when a request is wrong, unauthorised, or unanswerable.

The contract tests cover the shapes a working request produces. These cover the
other half, which is the half a client actually has to program against: a
dashboard needs to distinguish "no run trained yet" from "that paper does not
exist" from "your key is wrong", and it can only do that if each failure is
distinct, uniform, and stable.

Four groups, each protecting a decision that would be easy to erode:

* **One error envelope.** Every deliberate failure carries ``error`` and
  ``detail``, so the frontend has one error path rather than three.
* **Refusal beats fabrication.** ``/ask`` returns 501 with master spec §20's exact
  sentence. A plausible generated answer with no retrieval behind it is the single
  outcome that specification forbids, and the test pins the wording verbatim.
* **Degraded, not broken.** With no run, or a run missing its model, the API stays
  up and reports what is missing. Panels that need nothing but the corpus keep
  working.
* **Security responses are still API responses.** A 401 or a 413 carries the same
  envelope and the same headers as a 200.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.capabilities import RAG_UNAVAILABLE_REASON
from src.api.security import SECURITY_HEADERS
from src.config.settings import Settings
from tests.integration.conftest import SVM_RUN_ID, ClientFactory

#: Master spec §20, verbatim. The exact sentence is part of the contract, not a
#: paraphrasable message: a retrieval system that cannot ground an answer must say
#: this rather than produce one.
RAG_REFUSAL = "Information not found in the provided paper."

#: Paths that must not resolve to anything outside the run store, whatever the
#: encoding. Traversal is handled by the lookup being a dictionary hit rather than
#: a path join, so these are regression cover for that staying true.
TRAVERSAL_PATHS = (
    "/api/papers/..%2F..%2Fetc%2Fpasswd",
    "/api/papers/%2E%2E%2F%2E%2E%2Fconfigs%2Fapi.yaml",
    "/api/papers/%2E%2E",
    "/api/runs/..%2F..%2Fsecret",
    "/api/runs/%2E%2E%2F%2E%2E%2F.env",
)


def error_body(response: object) -> dict[str, object]:
    """Assert a response carries the uniform error envelope and return it."""
    body = response.json()  # type: ignore[attr-defined]
    assert isinstance(body, dict), f"error body is not an object: {body!r}"
    assert isinstance(body.get("error"), str) and body["error"], "missing machine-readable key"
    assert isinstance(body.get("detail"), str) and body["detail"], "missing human-readable detail"
    assert set(body) <= {"error", "detail", "hint"}, f"unexpected error fields: {sorted(body)}"
    return body


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "/api/papers/no-such-paper",
        "/api/papers/no-such-paper/similar",
        "/api/papers/no-such-paper/explanation",
        "/api/runs/no-such-run",
    ],
)
def test_unknown_resource_is_a_404_with_the_shared_envelope(
    svm_client: TestClient, path: str
) -> None:
    """A missing paper or run is a 404 the client can branch on.

    Named ``not_found`` rather than left to fall through to a generic key: the
    dashboard distinguishes a stale link from a server that has no run, and both
    would otherwise arrive as ``"error"``.
    """
    response = svm_client.get(path)

    assert response.status_code == 404
    assert error_body(response)["error"] == "not_found"


def test_an_unknown_api_path_is_json_not_html(svm_client: TestClient) -> None:
    """A typo under ``/api`` still answers in JSON.

    Worth pinning because the static mount at ``/`` is the catch-all: it is the
    thing that answers an unmatched path, and if its 404 escaped the app's
    exception handler the client would get an HTML page where it expects an error
    object.
    """
    response = svm_client.get("/api/no-such-endpoint")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert error_body(response)["error"] == "not_found"


def test_a_wrong_method_still_returns_the_shared_envelope(svm_client: TestClient) -> None:
    """A method mistake answers in JSON, with a key the client can branch on.

    Worth pinning because of *how* it resolves. Mounting the dashboard at ``/``
    means the static app matches every path, so it — not the router — answers a
    request the router only partially matched. It allows ``GET`` and ``HEAD``, so a
    ``POST`` to a read-only endpoint surfaces as 405 while a ``GET`` to a
    write-only one surfaces as 404. Both are honest and neither is an HTML page,
    which is what the frontend depends on.
    """
    wrong_method = svm_client.post("/api/papers", json={})
    assert wrong_method.status_code == 405
    assert error_body(wrong_method)["error"] == "method_not_allowed"

    paper_id = svm_client.get("/api/papers", params={"limit": 1}).json()["items"][0]["paper_id"]
    read_of_a_post_route = svm_client.get(f"/api/papers/{paper_id}/ask")
    assert read_of_a_post_route.status_code == 404
    assert error_body(read_of_a_post_route)["error"] == "not_found"


@pytest.mark.parametrize("path", TRAVERSAL_PATHS)
def test_traversal_in_a_path_parameter_finds_nothing(svm_client: TestClient, path: str) -> None:
    """An encoded traversal attempt is an ordinary miss.

    Ids are looked up in a dictionary built from the run's own manifest, never
    joined onto a filesystem path, so ``../`` is just a string that matches no
    paper. Pinned so a future "load the paper from disk on demand" optimisation
    cannot quietly introduce a path join.
    """
    response = svm_client.get(path)

    assert response.status_code in (404, 422), f"{path} returned {response.status_code}"
    body = error_body(response)
    for leaked in ("root:", "PRIVATE KEY", "ARIS_API_KEY", "api_key"):
        assert leaked not in body["detail"], f"{path} leaked '{leaked}' into the error"


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------
def test_an_unknown_split_names_the_valid_ones(svm_client: TestClient) -> None:
    """A bad ``split`` is rejected with the accepted values listed.

    Rejecting rather than falling back to a default: silently serving a different
    split than the one requested would put wrong rows under a filter label the
    user chose.
    """
    response = svm_client.get("/api/papers", params={"split": "bogus"})

    assert response.status_code == 422
    detail = error_body(response)["detail"]
    assert "split must be one of" in detail
    for valid in ("all", "train", "val", "test", "held_out"):
        assert valid in detail


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": -5}, "limit"),
        ({"offset": -1}, "offset"),
        ({"needs_review": "maybe"}, "needs_review"),
        ({"q": "x" * 500}, "q"),
    ],
)
def test_malformed_query_parameters_are_rejected_with_the_field_named(
    svm_client: TestClient, params: dict[str, object], field: str
) -> None:
    """A 422 says which parameter was wrong, without echoing what was sent.

    The field path is enough to fix a client. The submitted value is deliberately
    omitted from the body: echoing input into an error is how a payload ends up in
    a log aggregator that was never meant to hold it.
    """
    response = svm_client.get("/api/papers", params=params)

    assert response.status_code == 422
    body = error_body(response)
    assert body["error"] == "validation_error"
    assert field in str(body.get("hint") or "")


@pytest.mark.parametrize(
    "payload",
    [
        {"abstract": "a title is required"},
        {"title": ""},
        {"title": "   "},
        {"title": "ok", "abstract": "ok", "unexpected": "field"},
        {"title": "ok", "extra_model": 1},
    ],
)
def test_classify_rejects_a_malformed_body(svm_client: TestClient, payload: dict) -> None:
    """Missing, empty, and unknown fields are all 422s from the schema.

    Unknown keys are rejected rather than ignored (``extra="forbid"``), because a
    typo in a client payload silently doing nothing is worse than a loud failure —
    the request appears to succeed while the field never arrives.
    """
    response = svm_client.post("/api/papers/classify", json=payload)

    assert response.status_code == 422
    assert error_body(response)["error"] == "validation_error"


def test_classify_enforces_the_configured_text_ceiling(
    svm_client: TestClient, settings: Settings
) -> None:
    """An abstract above ``security.max_text_chars`` is refused by the schema.

    The limit lives in configuration and is read at validation time, so this is
    checked against the configured number rather than a literal in the test
    (master spec §32). Under the ceiling the same request succeeds, which is what
    proves the rejection is the limit and not the size of the body.
    """
    limit = settings.api.security.max_text_chars

    too_long = svm_client.post(
        "/api/papers/classify", json={"title": "ok", "abstract": "word " * (limit // 2)}
    )
    assert too_long.status_code == 422
    body = error_body(too_long)
    assert str(limit) in str(body.get("hint") or ""), "the error names the configured limit"

    within = svm_client.post(
        "/api/papers/classify", json={"title": "ok", "abstract": "y" * (limit - 1)}
    )
    assert within.status_code == 200


def test_ask_validates_the_question_before_refusing(svm_client: TestClient) -> None:
    """An empty question is a 422; the refusal is not a way to skip validation."""
    paper_id = svm_client.get("/api/papers", params={"limit": 1}).json()["items"][0]["paper_id"]
    response = svm_client.post(f"/api/papers/{paper_id}/ask", json={"question": ""})

    assert response.status_code == 422
    assert error_body(response)["error"] == "validation_error"


# ---------------------------------------------------------------------------
# Refusal rather than fabrication (master spec §20)
# ---------------------------------------------------------------------------
def test_ask_refuses_with_the_exact_required_sentence(svm_client: TestClient) -> None:
    """Question answering returns 501 and §20's wording, verbatim.

    The endpoint exists so the dashboard's composer has a definite answer to
    submitting a question. It refuses categorically rather than per question
    because there is no retriever and no full text, so *every* answer would be
    ungrounded — the one outcome §20 forbids.
    """
    paper_id = svm_client.get("/api/papers", params={"limit": 1}).json()["items"][0]["paper_id"]
    response = svm_client.post(
        f"/api/papers/{paper_id}/ask", json={"question": "What method does this paper use?"}
    )

    assert response.status_code == 501
    body = error_body(response)
    assert body["error"] == "not_implemented"
    assert body["detail"] == RAG_REFUSAL

    # A slug for code, a sentence for people. The user-facing explanation lives in
    # the ``rag_ask`` capability so the wording has one home.
    assert response.headers["X-Unavailable-Reason"] == "retrieval-index-missing"
    capabilities = {
        entry["key"]: entry for entry in svm_client.get("/api/meta").json()["capabilities"]
    }
    assert capabilities["rag_ask"]["reason"] == RAG_UNAVAILABLE_REASON


def test_ask_about_an_unknown_paper_is_404_not_501(svm_client: TestClient) -> None:
    """Existence is checked before the refusal.

    A 501 for a paper that does not exist would tell the client the feature is
    missing when the real problem is a stale id, and the two need different
    handling in the UI.
    """
    response = svm_client.post("/api/papers/no-such-paper/ask", json={"question": "x"})

    assert response.status_code == 404
    assert error_body(response)["error"] == "not_found"


# ---------------------------------------------------------------------------
# Request size
# ---------------------------------------------------------------------------
def test_an_oversized_body_is_refused_before_it_is_read(
    svm_client: TestClient, settings: Settings
) -> None:
    """413 from the declared ``Content-Length``, with the fix in the hint.

    Checked from the header rather than by reading the body, so an oversized
    request costs a header parse instead of memory. The hint names the setting
    because raising a limit is a configuration change, not a code change.
    """
    limit = settings.api.security.max_request_bytes
    response = svm_client.post(
        "/api/papers/classify", json={"title": "t", "abstract": "x" * (limit + 100)}
    )

    assert response.status_code == 413
    body = error_body(response)
    assert body["error"] == "payload_too_large"
    assert str(limit) in body["detail"]
    assert "max_request_bytes" in str(body["hint"])


def test_a_non_integer_content_length_is_a_400(svm_client: TestClient) -> None:
    """An unparseable ``Content-Length`` fails closed.

    Treating it as absent would let a crafted header skip the size check
    altogether, which is the whole purpose of the middleware.
    """
    response = svm_client.post(
        "/api/papers/classify",
        content=b'{"title":"t"}',
        headers={"content-length": "not-a-number", "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert error_body(response)["error"] == "bad_request"


# ---------------------------------------------------------------------------
# API key (master spec §40: authentication-ready)
# ---------------------------------------------------------------------------
def test_a_required_key_protects_data_endpoints(api_client_factory: ClientFactory) -> None:
    """With enforcement on, data needs the key and health does not.

    Health is exempt on purpose: a monitoring probe that needs the shared secret
    stops working the moment the secret rotates, which is when monitoring matters
    most.
    """
    client = api_client_factory(
        SVM_RUN_ID, security={"require_api_key": True}, env={"aris_api_key": "test-secret"}
    )

    missing = client.get("/api/meta")
    assert missing.status_code == 401
    assert error_body(missing)["error"] == "unauthorized"
    assert missing.headers["WWW-Authenticate"] == "X-API-Key"

    assert client.get("/api/meta", headers={"X-API-Key": "test-secret"}).status_code == 200
    assert client.get("/api/papers", headers={"X-API-Key": "test-secret"}).status_code == 200
    assert client.get("/api/health").status_code == 200, "liveness stays open"


@pytest.mark.parametrize("supplied", ["", "wrong", "test-secre", "test-secrets", "TEST-SECRET"])
def test_a_wrong_key_is_rejected_without_echoing_it(
    api_client_factory: ClientFactory, supplied: str
) -> None:
    """Near-misses fail, and the error never quotes the submitted value.

    Prefixes and suffixes of the real key are included because the comparison is
    ``compare_digest``: a short-circuiting ``==`` would leak the length of the
    matching prefix through response timing.
    """
    client = api_client_factory(
        SVM_RUN_ID, security={"require_api_key": True}, env={"aris_api_key": "test-secret"}
    )

    response = client.get("/api/meta", headers={"X-API-Key": supplied})

    assert response.status_code == 401
    detail = error_body(response)["detail"]
    assert "X-API-Key" in detail, "the error names the header a client should use"
    assert supplied not in detail or not supplied, "the submitted value is not echoed"
    assert "test-secret" not in detail, "the expected value is never disclosed"


def test_enforcement_without_a_configured_key_fails_as_a_misconfiguration(
    api_client_factory: ClientFactory,
) -> None:
    """Required-but-unset is a 500 naming the cause, not a permanent 401.

    A 401 would be indistinguishable from a wrong key and no value could satisfy
    it, sending the operator to look for a client bug. The 500 says which variable
    is missing and where to set it.
    """
    client = api_client_factory(
        SVM_RUN_ID, security={"require_api_key": True}, env={"aris_api_key": None}
    )

    response = client.get("/api/meta", headers={"X-API-Key": "anything"})

    assert response.status_code == 500
    body = error_body(response)
    assert body["error"] == "internal_error"
    assert "ARIS_API_KEY" in body["detail"]


def test_the_default_configuration_is_open_and_says_so(settings: Settings) -> None:
    """Shipping open is the honest default for a loopback dev server.

    Pinned as a test so it stays a decision rather than an accident: the server
    binds to localhost, logs a warning at startup, and turning enforcement on is
    one environment variable plus one flag — no code change.
    """
    assert settings.api.security.require_api_key is False
    assert settings.api.server.host in ("127.0.0.1", "localhost")
    assert settings.api.security.api_key_header == "X-API-Key"


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/api/health", "/api/meta", "/api/papers", "/"])
def test_security_headers_are_on_every_successful_response(
    svm_client: TestClient, path: str
) -> None:
    """The header set is attached to JSON and to the static dashboard alike.

    The dashboard is the response that most needs them, and it is served by a
    different Starlette app than the routers — so covering both is the point.
    """
    headers = svm_client.get(path).headers

    for name, value in SECURITY_HEADERS.items():
        assert headers.get(name) == value, f"{path} is missing {name}"


def test_security_headers_survive_an_error_response(svm_client: TestClient) -> None:
    """A 404 is still hardened.

    Errors are the responses most likely to reflect input, so losing the header
    set here would matter more than losing it on a 200.
    """
    headers = svm_client.get("/api/papers/no-such-paper").headers

    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_cors_never_allows_any_origin_and_is_silent_same_origin(
    svm_client: TestClient, settings: Settings
) -> None:
    """CORS is configured with explicit origins only, and adds nothing by default.

    ``*`` would let any page on the internet read this API's responses. The two
    configured entries exist for the separate-dev-server case; because the
    dashboard is normally served from the API's own origin its fetches carry no
    ``Origin`` header at all, so no allowance is emitted on the path that matters.
    """
    assert settings.api.cors.allows_any_origin is False
    assert "*" not in settings.api.cors.allow_origins
    assert settings.api.cors.allow_credentials is False, "no cookie auth to echo"

    assert "access-control-allow-origin" not in svm_client.get("/api/health").headers

    allowed = settings.api.cors.allow_origins[0]
    echoed = svm_client.get("/api/health", headers={"Origin": allowed})
    assert echoed.headers.get("access-control-allow-origin") == allowed

    denied = svm_client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


# ---------------------------------------------------------------------------
# Degraded states
# ---------------------------------------------------------------------------
def test_a_fresh_clone_with_no_run_stays_up_and_says_what_is_missing(
    no_run_client: TestClient,
) -> None:
    """With nothing trained, the API reports degraded rather than failing to start.

    This is the state of a clone before ``train_baseline.py`` has run, so it is
    the first thing a new contributor sees. The shell must render, with an
    explanation instead of empty panels.
    """
    health = no_run_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["run_id"] is None
    assert health.json()["warnings"], "degraded health explains itself"

    meta = no_run_client.get("/api/meta")
    assert meta.status_code == 200, "the shell still renders"
    body = meta.json()
    assert body["run"] is None
    assert len(body["caveats"]) == 1
    assert "no error" in body["caveats"][0].lower() or "necessity" in body["caveats"][0]
    assert all(entry["available"] is False for entry in body["capabilities"])


@pytest.mark.parametrize(
    "path",
    ["/api/stats", "/api/stats/domains", "/api/research/trends", "/api/papers", "/api/runs/active"],
)
def test_data_endpoints_are_503_with_no_run_and_name_the_command(
    no_run_client: TestClient, path: str
) -> None:
    """503, not 500: the request was correct and the server has nothing to serve.

    The hint carries the command that fixes it, because "service unavailable" on a
    local dev box is almost always one missing training run.
    """
    response = no_run_client.get(path)

    assert response.status_code == 503
    body = error_body(response)
    assert body["error"] == "run_unavailable"
    assert "train_baseline.py" in str(body["hint"])


def test_a_run_without_its_model_still_serves_the_corpus(modelless_client: TestClient) -> None:
    """A run missing ``model.joblib`` degrades feature by feature, not wholesale.

    Reachable in practice — ``save_model`` can be off, and a run directory can be
    copied without its largest file. Everything read from the run's own artifacts
    keeps working; only the endpoints that need a forward pass fail.
    """
    meta = modelless_client.get("/api/meta").json()
    assert meta["run"] is not None, "metrics and predictions are still readable"
    assert meta["run"]["model_ready"] is False
    assert meta["run"]["warnings"], "the missing model is reported, not hidden"

    health = modelless_client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert health["dataset_ready"] is True
    assert health["model_ready"] is False

    capabilities = {entry["key"]: entry for entry in meta["capabilities"]}
    assert capabilities["corpus"]["available"] is True
    assert capabilities["classification"]["available"] is False
    assert "no saved model" in capabilities["classification"]["reason"]

    # Stored numbers come from the run's artifacts, so they survive.
    for path in ("/api/papers", "/api/stats", "/api/stats/domains", "/api/research/trends"):
        assert modelless_client.get(path).status_code == 200, f"{path} needs no model"


@pytest.mark.parametrize("suffix", ["similar", "explanation"])
def test_model_dependent_endpoints_are_503_without_a_model(
    modelless_client: TestClient, suffix: str
) -> None:
    """A forward pass with no model is a 503 that names the missing artifact.

    The paper id is read from the listing rather than hard-coded, so a 404 from a
    stale id cannot masquerade as the 503 this test is looking for.
    """
    paper_id = modelless_client.get("/api/papers", params={"limit": 1}).json()["items"][0][
        "paper_id"
    ]
    response = modelless_client.get(f"/api/papers/{paper_id}/{suffix}")

    assert response.status_code == 503
    assert "model" in error_body(response)["detail"].lower()


def test_classify_is_503_without_a_model(modelless_client: TestClient) -> None:
    """Classification cannot degrade to a guess.

    Returning the majority class, or the ground-truth label of a similar paper,
    would produce a response indistinguishable from a real prediction.
    """
    response = modelless_client.post(
        "/api/papers/classify", json={"title": "anything", "abstract": "any text"}
    )

    assert response.status_code == 503
    assert "model" in error_body(response)["detail"].lower()
