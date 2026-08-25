/* ==========================================================================
   API client
   ==========================================================================
   One fetch wrapper, one error type, one function per endpoint.

   The base URL is same-origin `/api` because the server mounts this directory
   at `/` (see src/api/app.py). A same-origin fetch is not a cross-origin
   request, so nothing here needs CORS to be relaxed. The override below exists
   for the separate-dev-server case and is read from a `data-api-base`
   attribute on <html>, NOT from the query string: a URL parameter is
   attacker-supplied, and a link that repoints the dashboard at another host
   would make this page render someone else's data as if it were yours.

   Every deliberate failure from the API has the same body -- {error, detail,
   hint} -- so this module can normalise all of them into one ApiError and the
   UI needs one error path rather than three. A 501 from /ask is an *expected*
   outcome, not a bug: the caller catches it and renders the refusal.
   ========================================================================== */

/** Give up on a request after this long. A local read of a cached run is fast;
    anything past this is a server that is not coming back. */
const TIMEOUT_MS = 15000;

/** Header the server expects the shared key in (configs/api.yaml). */
const API_KEY_HEADER = "X-API-Key";

/** Where the operator's key is kept when one is required. sessionStorage, not
    localStorage: it should not outlive the tab. */
const API_KEY_STORE = "ri-api-key";

/**
 * An API call that did not return data.
 *
 * `status` is 0 when the request never reached a server, which is the case the
 * UI must distinguish -- "backend is down" needs different words from "the
 * backend said no".
 */
export class ApiError extends Error {
  constructor(message, { status = 0, code = "error", detail = "", hint = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.hint = hint;
  }

  /** True when no HTTP response arrived at all. */
  get isUnreachable() {
    return this.status === 0;
  }

  /** True when the server is up but has no usable training run. */
  get isRunMissing() {
    return this.status === 503;
  }
}

/** Resolve the API root. */
function base() {
  const override = document.documentElement.dataset.apiBase;
  return (override || "/api").replace(/\/$/, "");
}

function keyHeader() {
  let key = null;
  try {
    key = sessionStorage.getItem(API_KEY_STORE);
  } catch {
    // Private-browsing modes can throw on storage access. An absent key is a
    // 401 with a clear message, which is a better outcome than a dead page.
  }
  return key ? { [API_KEY_HEADER]: key } : {};
}

/**
 * Store the shared API key for this tab.
 *
 * @param {string} key Value of ARIS_API_KEY on the server.
 */
export function setApiKey(key) {
  sessionStorage.setItem(API_KEY_STORE, key);
}

/** Append a query string, dropping empty parameters. */
function withQuery(path, params = {}) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

/**
 * Perform one API request.
 *
 * @param {string} path Path below the API root, e.g. `/meta`.
 * @param {{method?: string, body?: unknown, timeout?: number}} [options]
 * @returns {Promise<any>} Parsed JSON body.
 * @throws {ApiError} On a network failure, a timeout, or any non-2xx status.
 */
async function request(path, { method = "GET", body, timeout = TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  let response;
  try {
    response = await fetch(base() + path, {
      method,
      headers: {
        Accept: "application/json",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        ...keyHeader(),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      credentials: "same-origin",
    });
  } catch (cause) {
    const timedOut = cause instanceof DOMException && cause.name === "AbortError";
    throw new ApiError(timedOut ? "Request timed out" : "Cannot reach the API", {
      status: 0,
      code: timedOut ? "timeout" : "unreachable",
      detail: timedOut
        ? `The server did not answer ${path} within ${Math.round(timeout / 1000)} seconds.`
        : `No response from ${base()}${path}.`,
      hint: "Start it with: python scripts/serve_api.py",
    });
  } finally {
    // Cleared once headers are in: the body of these responses is small, and a
    // stray timer would abort a stream that is already arriving.
    clearTimeout(timer);
  }

  // 204 has no body; every other response here is JSON. A proxy error page is
  // not, which is why the parse is allowed to fail rather than throwing a
  // SyntaxError from inside a render path.
  const payload = response.status === 204 ? null : await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(`${method} ${path} failed with ${response.status}`, {
      status: response.status,
      code: payload?.error ?? "error",
      detail: payload?.detail ?? response.statusText ?? "The request failed.",
      hint: payload?.hint ?? null,
    });
  }
  return payload;
}

/* ==========================================================================
   Endpoints
   ==========================================================================
   Thin by design: each one names a route and returns its body unchanged. No
   reshaping happens here, so a field in the UI can be traced to a field in
   src/api/schemas.py without a translation layer in between.
   ========================================================================== */

/** GET /api/health — liveness plus what the server managed to load. */
export const getHealth = () => request("/health");

/** GET /api/meta — identity, storage, active run, capabilities, caveats. */
export const getMeta = () => request("/meta");

/** GET /api/stats — the four headline tiles. */
export const getStats = () => request("/stats");

/** GET /api/stats/domains — corpus composition by ground-truth label. */
export const getDomains = () => request("/stats/domains");

/** GET /api/research/trends — papers per domain per publication year. */
export const getTrends = () => request("/research/trends");

/**
 * GET /api/papers — a page of the corpus.
 *
 * @param {{split?: string, q?: string, needs_review?: boolean, limit?: number,
 *          offset?: number}} [params]
 */
export const listPapers = (params = {}) => request(withQuery("/papers", params));

/** GET /api/papers/{id} — one paper in full. */
export const getPaper = (paperId) => request(`/papers/${encodeURIComponent(paperId)}`);

/** GET /api/papers/{id}/similar — lexically nearest neighbours. */
export const getSimilar = (paperId, limit) =>
  request(withQuery(`/papers/${encodeURIComponent(paperId)}/similar`, { limit }));

/** GET /api/papers/{id}/explanation — term contributions for the prediction. */
export const getExplanation = (paperId) =>
  request(`/papers/${encodeURIComponent(paperId)}/explanation`);

/** POST /api/papers/classify — run new text through the run's own model. */
export const classify = (title, abstract = "") =>
  request("/papers/classify", { method: "POST", body: { title, abstract } });

/**
 * POST /api/papers/{id}/ask — always refuses in this build.
 *
 * Kept as a real request rather than a client-side shortcut. The refusal and
 * its wording are the server's (master spec §20); if a retrieval index lands
 * later, this call starts succeeding with no change here.
 *
 * @throws {ApiError} 501 with the refusal as `detail`.
 */
export const ask = (paperId, question) =>
  request(`/papers/${encodeURIComponent(paperId)}/ask`, {
    method: "POST",
    body: { question },
  });
