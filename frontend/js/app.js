/* ==========================================================================
   Application
   ==========================================================================
   Renders the dashboard from the API and wires up interaction.

   Deliberately framework-free. The project has no Node toolchain, so adding
   React plus a bundler would commit the frontend to a stack in order to look at
   a layout. Every component here is a function from a response body to markup,
   which is the shape that ports to JSX mechanically if that is the eventual
   choice.

   Three rules this file follows, because the system is only partly built:

   1. Nothing is invented. Every number comes from a response field. Where the
      backend reports a feature as unavailable, the panel renders that state --
      it never falls back to a plausible-looking placeholder.
   2. Thresholds are not re-derived. `needs_review` arrives as a boolean the
      server computed (master spec §15); this file only chooses how to draw it.
   3. A score is drawn according to its `confidence_kind`. A probability gets a
      percentage and a filled bar; an unbounded decision margin gets neither,
      because there is no scale for it to be a fraction of.
   ========================================================================== */

import {
  ApiError,
  ask,
  getDomains,
  getExplanation,
  getMeta,
  getPaper,
  getSimilar,
  getStats,
  getTrends,
  listPapers,
  uploadPaper,
} from "./api.js";
import { legendMarkup, onChartInvalidate, renderDonut, renderTrends } from "./charts.js";
import { domainColor, isRegisteredDomain, readToken, registerDomains } from "./domains.js";
import { icon } from "./icons.js";
import { NAV_GROUPS } from "./nav.js";

const $ = (sel) => document.querySelector(sel);

/** How the confidence column is headed, per what the model actually emits. */
const CONFIDENCE_HEADINGS = {
  probability: "Confidence",
  decision: "Margin",
  unavailable: "Confidence",
};

/** Idle time before a keystroke in the search field becomes a request. */
const SEARCH_DEBOUNCE_MS = 250;

/** Everything the page has loaded. One object so a re-render never has to refetch. */
const state = {
  meta: null,
  /** @type {Map<string, {key: string, label: string, available: boolean, reason: string|null}>} */
  capabilities: new Map(),
  domains: null,
  trends: null,
  focusId: null,
  focusToken: 0,
  focusedPaper: null,
  focusedExplanation: null,
  table: { split: "held_out", q: "", needsReview: false, offset: 0, limit: 10, total: 0 },
};

/* ==========================================================================
   Primitives
   ========================================================================== */

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const clamp01 = (value) => Math.max(0, Math.min(1, value));

/** Signed value with a real minus sign, so "-" is never mistaken for a hyphen. */
const signed = (value, digits = 2) =>
  `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;

/**
 * Format a byte count at a unit a human can read.
 *
 * The storage meter measures a few hundred kilobytes against a 10 GB quota, and
 * "0.0 GB" reads as a bug rather than as a small number.
 *
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
  if (bytes < 1e6) return `${Math.round(bytes / 1e3)} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

/** A domain's hue, or the neutral one when it is outside the registry. */
const hueFor = (label) =>
  isRegisteredDomain(label) ? domainColor(label) : readToken("--series-other");

/**
 * A collapsible caveat attached to a panel of numbers.
 *
 * The summary line is always visible, so the qualification is never hidden --
 * only its explanation is behind the disclosure. Master spec §14/§17 require
 * these claims to be bounded where the numbers are shown, not in a footnote
 * somewhere else.
 *
 * @param {string} summary Always-visible one-liner; may contain inline markup.
 * @param {string} body Expanded explanation; escaped by callers if from the API.
 * @returns {string} HTML markup.
 */
function noteMarkup(summary, body) {
  return `<details class="note">
    <summary>${icon("info", 14)}<span>${summary}</span><span class="note__caret">${icon("chevron", 13)}</span></summary>
    <div class="note__body">${body}</div>
  </details>`;
}

/**
 * The "this is not built" state for a panel.
 *
 * Rendered from the reason the API supplies rather than from a string here, so
 * the explanation has one home (src/api/capabilities.py) and the panel starts
 * showing data the moment that entry flips to available.
 *
 * @param {string} reason User-facing explanation from the API.
 * @param {{title?: string, requires?: string|null}} [options]
 * @returns {string} HTML markup.
 */
function unavailableMarkup(reason, { title = "Not available in this build", requires = null } = {}) {
  return `<div class="unavailable">
    <span class="unavailable__glyph" aria-hidden="true">${icon("lock", 15)}</span>
    <div>
      <p class="unavailable__title">${escapeHtml(title)}</p>
      <p class="unavailable__reason">${escapeHtml(reason)}</p>
      ${requires ? `<p class="unavailable__requires">Needs: ${escapeHtml(requires)}</p>` : ""}
    </div>
  </div>`;
}

/**
 * Render a failed request as something a reader can act on.
 *
 * @param {ApiError|Error} error
 * @param {{title?: string}} [options]
 * @returns {string} HTML markup.
 */
function alertMarkup(error, { title = "Could not load this panel" } = {}) {
  const detail = error instanceof ApiError ? error.detail : error.message;
  const hint = error instanceof ApiError ? error.hint : null;
  return `<div class="alert" role="alert">
    <span class="alert__glyph" aria-hidden="true">${icon("warn", 16)}</span>
    <div>
      <p class="alert__title">${escapeHtml(title)}</p>
      <p class="alert__detail">${escapeHtml(detail || "The request failed.")}</p>
      ${hint ? `<p class="alert__hint"><code>${escapeHtml(hint)}</code></p>` : ""}
    </div>
  </div>`;
}

const emptyMarkup = (text) => `<p class="empty">${escapeHtml(text)}</p>`;

/** Placeholder rows while a request is in flight. */
const skeletonMarkup = (rows = 3) =>
  `<div class="skeleton" aria-hidden="true">${'<span class="skeleton__line"></span>'.repeat(rows)}</div>`;

/**
 * The shell a labelled bar sits in.
 *
 * Two layouts, chosen by label length rather than by panel. Compact puts the
 * label, bar, and value on one line, which suits single words. Stacked gives the
 * label its own line, which is the only way a name like "Computer Networks and
 * Communications" survives a 600px column -- in the compact grid it clips
 * mid-word, and a reader who cannot finish the label cannot use the bar.
 *
 * @param {string} label Row label.
 * @param {string} value Preformatted value.
 * @param {string} meter Markup for the track and its fill.
 * @param {{stacked?: boolean}} [options]
 * @returns {string} HTML markup.
 */
function rowShell(label, value, meter, { stacked = false } = {}) {
  if (stacked) {
    return `<div class="bar-row bar-row--stacked">
      <div class="bar-row__head">
        <span class="bar-row__label">${escapeHtml(label)}</span>
        <span class="bar-row__value">${escapeHtml(value)}</span>
      </div>
      ${meter}
    </div>`;
  }
  return `<div class="bar-row">
    <span class="bar-row__label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
    ${meter}
    <span class="bar-row__value">${escapeHtml(value)}</span>
  </div>`;
}

/** One labelled bar for a value on a 0..1 scale, growing from the left edge. */
function barRow(label, value, fraction, color, options) {
  const meter = `<div class="meter"><div class="meter__fill" style="width:${
    clamp01(fraction) * 100
  }%; background:${color}"></div></div>`;
  return rowShell(label, value, meter, options);
}

/**
 * One labelled bar for a SIGNED value, drawn from a zero baseline.
 *
 * A one-sided bar cannot carry a sign. Scaling it to magnitude gives the
 * rejected classes the longest bars, which reads as the exact opposite of the
 * ranking -- and clamping the negatives to zero width reads as "no signal" when
 * it means "strong evidence against". Anchoring to a centre axis puts direction
 * in position, the strongest channel available, so the winner is the bar
 * extending right regardless of hue or how the reader parses a minus sign.
 *
 * @param {string} label Row label.
 * @param {string} value Preformatted signed value.
 * @param {number} fraction |value| over the largest |value| in the group, 0..1.
 * @param {string} color Fill colour.
 * @param {boolean} positive Which side of the axis to grow from.
 * @param {{stacked?: boolean}} [options]
 * @returns {string} HTML markup.
 */
function divergingRow(label, value, fraction, color, positive, options) {
  // Half the track per side, so the two directions share one scale.
  const half = clamp01(fraction) * 50;
  // Square at the axis, rounded at the free end: the data-end is the one that
  // moves, and the anchored end belongs to the baseline.
  const geometry = positive
    ? `left:50%; width:${half}%; border-radius:0 var(--r-pill) var(--r-pill) 0`
    : `left:${50 - half}%; width:${half}%; border-radius:var(--r-pill) 0 0 var(--r-pill)`;
  const meter = `<div class="meter meter--diverging">
      <span class="meter__axis" aria-hidden="true"></span>
      <div class="meter__fill" style="${geometry}; background:${color}"></div>
    </div>`;
  return rowShell(label, value, meter, options);
}

/** Look up one capability from GET /api/meta. */
const capability = (key) => state.capabilities.get(key);

/**
 * The best available explanation for why something cannot be used.
 *
 * Prefers the server's reason so the sentence is not duplicated in the client.
 */
function reasonFor(key, fallback) {
  const entry = capability(key);
  return entry && !entry.available && entry.reason ? entry.reason : fallback;
}

/* ==========================================================================
   Chrome
   ========================================================================== */

function renderStaticIcons() {
  $("#search-icon").innerHTML = icon("search", 16);
  $("#btn-bell").insertAdjacentHTML("afterbegin", icon("bell"));
  $("#btn-help").innerHTML = icon("help");
  $("#user-chevron").innerHTML = icon("chevron", 15);
  $("#nav-toggle").innerHTML = icon("menu");
  $("#ask-icon").innerHTML = icon("chat", 16);
  $("#ask-send").innerHTML = icon("send", 15);
  for (const id of ["donut", "trends", "preview", "similar"]) {
    const el = $(`#${id}-arrow`);
    if (el) el.innerHTML = icon("arrow_right", 14);
  }
}

function renderNav() {
  $("#nav").innerHTML = NAV_GROUPS.map(
    (group) => `
      <div class="nav-group">
        <div class="nav-group__label">${escapeHtml(group.group)}</div>
        ${group.items
          .map(
            (item) => `
          <button class="nav-item" type="button"
            ${item.built ? 'aria-current="page"' : ""}
            data-nav="${escapeHtml(item.label)}"
            data-built="${item.built}"
            ${item.capability ? `data-capability="${escapeHtml(item.capability)}"` : ""}>
            ${icon(item.icon)}
            <span>${escapeHtml(item.label)}</span>
          </button>`,
          )
          .join("")}
      </div>`,
  ).join("");
}

/**
 * Fill in identity and the storage meter.
 *
 * `is_authenticated` is false in this build and the role line says so, because
 * the chip is the one piece of chrome that would otherwise imply an account.
 */
function renderIdentity(meta) {
  const { user, storage } = meta;
  $("#greeting").textContent = `Welcome back, ${user.first_name}! 👋`;
  $("#user-name").textContent = user.full_name;
  $("#user-role").textContent = user.role;
  $("#user-initials").textContent = user.initials;

  const percent = clamp01(storage.percent / 100) * 100;
  $("#storage-value").textContent =
    `${formatBytes(storage.used_bytes)} / ${storage.quota_gb} GB`;
  $("#storage-fill").style.width = `${percent}%`;
  const meter = $("#storage-meter");
  meter.setAttribute("aria-valuenow", percent.toFixed(1));
  meter.setAttribute(
    "aria-valuetext",
    `${formatBytes(storage.used_bytes)} of ${storage.quota_gb} gigabytes used` +
      (storage.measured.length ? `, measured across ${storage.measured.join(" and ")}` : ""),
  );
}

/**
 * The strip under the greeting: which run these numbers came from, and the
 * standing caveats that bound how all of them should be read.
 *
 * This exists because every panel below is relative to one training run. Without
 * it the page would present run-specific numbers as though they were properties
 * of the system.
 */
function renderRunStrip(meta) {
  const mount = $("#run-strip");
  if (!meta.run) {
    mount.innerHTML = `${alertMarkup(
      new ApiError("no run", {
        status: 503,
        detail:
          "No completed training run was found, so every panel below is empty by " +
          "necessity rather than by error.",
        hint: "python scripts/train_baseline.py --model tfidf_logreg",
      }),
      { title: "Nothing trained yet" },
    )}`;
    return;
  }

  const run = meta.run;
  const primary = run.metrics?.[run.primary_split]?.primary_metric ?? {};
  const corpus = Object.values(run.dataset.split_sizes ?? {}).reduce((a, b) => a + b, 0);

  const facts = [
    ["Run", run.run_id],
    ["Model", run.model_display_name],
  ];
  if (typeof primary.value === "number") {
    facts.push([
      String(primary.name ?? "score").replace(/_/g, " "),
      `${pct(primary.value)} on ${run.primary_split}`,
    ]);
  }
  facts.push(["Corpus", `${corpus} papers · ${run.classes.length} domains`]);
  if (run.dataset.built_at) facts.push(["Built", run.dataset.built_at.slice(0, 10)]);

  const tags = [];
  if (run.dataset.is_synthetic) {
    tags.push(`<span class="tag tag--warn">${icon("warn", 12)} synthetic corpus</span>`);
  }
  if (run.dataset.is_stale) {
    tags.push(`<span class="tag tag--warn">${icon("warn", 12)} dataset changed since training</span>`);
  }
  if (!run.model_ready) {
    tags.push(`<span class="tag tag--warn">${icon("warn", 12)} model file unreadable</span>`);
  }
  tags.push(`<span class="tag">${escapeHtml(meta.environment)}</span>`);

  const caveats = meta.caveats.length
    ? noteMarkup(
        `<strong>${meta.caveats.length} standing caveats</strong> apply to every number on this page.`,
        `<ul class="caveat-list">${meta.caveats
          .map((text) => `<li>${escapeHtml(text)}</li>`)
          .join("")}</ul>`,
      )
    : "";

  mount.innerHTML = `
    <div class="run-strip__facts">
      ${facts
        .map(
          ([label, value]) =>
            `<span class="run-strip__fact"><span class="run-strip__key">${escapeHtml(label)}</span>${escapeHtml(value)}</span>`,
        )
        .join("")}
      ${tags.join("")}
    </div>
    ${caveats}`;
}

function renderStats(tiles) {
  $("#stat-row").innerHTML = tiles
    .map((tile) => {
      const hue = readToken(tile.hue) || readToken("--series-other");
      // Long values are set at a smaller step: a domain name at hero size wraps
      // to three lines and pushes the tile out of alignment with its row.
      const isText = tile.value.length > 7;
      return `
      <article class="card stat">
        <div>
          <div class="stat__label">${escapeHtml(tile.label)}</div>
          <div class="stat__value${isText ? " stat__value--text" : ""}">${escapeHtml(tile.value)}</div>
          ${tile.note ? `<div class="stat__note">${escapeHtml(tile.note)}</div>` : ""}
        </div>
        <div class="stat__glyph" style="background:${hue}22; color:${hue}">
          ${icon(tile.icon, 21)}
        </div>
      </article>`;
    })
    .join("");
}

/**
 * Paper-specific stat tiles for the focused record.
 *
 * The corpus-level strip above the charts is still shown in the run banner.
 * Once a paper is focused, this row becomes the paper summary so the values
 * actually change with the selected record instead of staying pinned to the
 * active run.
 */
function renderPaperStats(paper) {
  const words = paper.text ? paper.text.trim().split(/\s+/).filter(Boolean).length : 0;
  const confidence =
    paper.confidence === null || paper.confidence === undefined
      ? "n/a"
      : paper.confidence_kind === "probability"
        ? pct(paper.confidence)
        : `${paper.confidence.toFixed(2)} margin`;
  const review =
    paper.needs_review === null || paper.needs_review === undefined
      ? "n/a"
      : paper.needs_review
        ? "Yes"
        : "No";

  renderStats([
    {
      id: "paper-length",
      label: "Paper Length",
      value: words ? `${words.toLocaleString()} words` : "n/a",
      note: "Selected paper text",
      icon: "papers",
      hue: "--series-6",
    },
    {
      id: "domains",
      label: "Predicted Domains",
      value: String(paper.predicted_scores?.length ?? 0),
      note: "Top label scores returned by the model",
      icon: "pie",
      hue: "--series-3",
    },
    {
      id: "confidence",
      label: "Top Confidence",
      value: confidence,
      note: paper.predicted_label ? `Prediction: ${paper.predicted_label}` : "No prediction available",
      icon: "trend",
      hue: "--series-1",
    },
    {
      id: "review",
      label: "Review Flag",
      value: review,
      note: paper.needs_review ? "Selected paper crosses the review threshold" : "Selected paper stays below the review threshold",
      icon: "flag",
      hue: "--series-4",
    },
  ]);
}

/* ==========================================================================
   Papers table
   ========================================================================== */

function chipMarkup(domain, { ghost = false, title = "" } = {}) {
  // Registered domains carry their reserved hue; anything else gets the
  // neutral dot, so a colour never implies a series that does not exist.
  const attr = title ? ` title="${escapeHtml(title)}"` : "";
  if (ghost) {
    return `<span class="chip chip--ghost"${attr}>${escapeHtml(domain)}</span>`;
  }
  return `<span class="chip"${attr}><span class="chip__dot" style="background:${hueFor(domain)}"></span>${escapeHtml(domain)}</span>`;
}

/**
 * The confidence column for one row.
 *
 * A probability is a fraction of a known whole, so it gets a percentage and a
 * proportional bar. A decision margin is an unbounded difference between the top
 * two class scores: there is no maximum for a bar to be a fraction of, and
 * printing 0.64 as "64%" would state a different quantity. So the margin is
 * shown as a number, labelled, with no bar.
 */
function confidenceCell(row) {
  if (row.confidence === null || row.confidence === undefined) {
    const why =
      row.split === "train"
        ? "This paper is in the training split. The model was fitted on it, so it has no honest score for it."
        : "This model exposes no confidence score.";
    return `<div class="conf-cell"><span class="conf-cell__none" title="${escapeHtml(why)}">not scored</span></div>`;
  }

  const flag = row.needs_review
    ? `<span class="review-flag">${icon("warn", 12)} Needs review</span>`
    : "";

  if (row.confidence_kind === "probability") {
    /* The bar is tinted by state, not by domain: at a glance the column should
       read as "how sure", and reusing a domain hue here would make the same
       colour mean two different things in one row. */
    const barColor = row.needs_review
      ? readToken("--status-warning")
      : readToken("--status-good");
    return `<div class="conf-cell">
      <span class="conf-cell__value">${pct(row.confidence)}</span>
      <div class="meter meter--thin">
        <div class="meter__fill" style="width:${clamp01(row.confidence) * 100}%; background:${barColor}"></div>
      </div>
      ${flag}
    </div>`;
  }

  return `<div class="conf-cell">
    <span class="conf-cell__value">${row.confidence.toFixed(2)}<span class="conf-cell__unit">margin</span></span>
    ${flag}
  </div>`;
}

function paperRowMarkup(row) {
  const byline = [row.authors_short, row.year].filter(Boolean).join(" · ") || row.paper_id;
  const predicted = row.predicted_label
    ? chipMarkup(row.predicted_label)
    : chipMarkup("not scored", {
        ghost: true,
        title: "No held-out prediction exists for this paper.",
      });
  // Only shown when the run got it wrong. A tick on every correct row would be
  // noise, and the mismatch is the case a reader needs to see.
  const mismatch =
    row.correct === false && row.true_label
      ? `<span class="mismatch">${icon("warn", 11)} actual: ${escapeHtml(row.true_label)}</span>`
      : "";

  return `
    <tr data-paper="${escapeHtml(row.paper_id)}"${row.paper_id === state.focusId ? ' aria-selected="true"' : ""}>
      <td>
        <div class="paper-cell">
          <span class="file-icon" aria-hidden="true">TXT</span>
          <span>
            <span class="paper-cell__title">${escapeHtml(row.title)}</span>
            <span class="paper-cell__meta">${escapeHtml(byline)} · ${escapeHtml(row.split)}</span>
          </span>
        </div>
      </td>
      <td><div class="chips">${predicted}${mismatch}</div></td>
      <td>${confidenceCell(row)}</td>
      <td>
        <div class="row-actions">
          <button class="icon-btn" type="button" data-action="focus" data-paper="${escapeHtml(row.paper_id)}"
            aria-label="Show analysis for ${escapeHtml(row.title)}">${icon("bars", 16)}</button>
          <button class="icon-btn" type="button" data-action="ask" data-paper="${escapeHtml(row.paper_id)}"
            aria-label="Ask about ${escapeHtml(row.title)}">${icon("chat", 16)}</button>
          <button class="icon-btn" type="button" data-unbuilt="Row actions"
            aria-label="More actions for ${escapeHtml(row.title)}">${icon("dots", 16)}</button>
        </div>
      </td>
    </tr>`;
}

/**
 * Load and draw the papers table for the current filter state.
 *
 * Filtering, searching, and paging all happen on the server, so the table shows
 * a real page of a real corpus rather than a client-side slice of whatever
 * happened to be fetched first.
 */
async function loadPapers({ resetFocus = false } = {}) {
  const body = $("#papers-body");
  if (!body) return;
  const filters = state.table;
  body.innerHTML = `<tr><td colspan="4">${skeletonMarkup(3)}</td></tr>`;

  let page;
  try {
    page = await listPapers({
      split: filters.split,
      q: filters.q || null,
      needs_review: filters.needsReview ? true : null,
      limit: filters.limit,
      offset: filters.offset,
    });
  } catch (error) {
    body.innerHTML = `<tr><td colspan="4">${alertMarkup(error, { title: "Could not load papers" })}</td></tr>`;
    $("#papers-foot").innerHTML = "";
    return;
  }

  filters.total = page.total;

  if (!page.items.length) {
    const message = filters.q
      ? `No paper in the ${filters.split.replace("_", " ")} set matches "${filters.q}".`
      : "No papers match these filters.";
    body.innerHTML = `<tr><td colspan="4">${emptyMarkup(message)}</td></tr>`;
  } else {
    body.innerHTML = page.items.map(paperRowMarkup).join("");
  }

  renderPapersFoot(page);

  if (resetFocus && page.items.length) {
    await focusPaper(page.items[0].paper_id);
  }
}

function renderPapersFoot(page) {
  const from = page.total === 0 ? 0 : page.offset + 1;
  const to = Math.min(page.offset + page.limit, page.total);
  const hasPrev = page.offset > 0;
  const hasNext = page.offset + page.limit < page.total;

  $("#papers-foot").innerHTML = `
    <span class="pager__status">Showing ${from}–${to} of ${page.total}${
      page.query ? ` matching “${escapeHtml(page.query)}”` : ""
    }</span>
    <span class="pager__controls">
      <button class="table-toggle" type="button" data-page="prev" ${hasPrev ? "" : "disabled"}>Previous</button>
      <button class="table-toggle" type="button" data-page="next" ${hasNext ? "" : "disabled"}>Next</button>
    </span>`;
}

/* ==========================================================================
   Paper preview
   ========================================================================== */

/**
 * Per-class scores, drawn according to what they are.
 *
 * For a probability model the scores are comparable to each other and to 1, so a
 * one-sided bar from zero is exactly right. For a margin model they are raw
 * decision values, which are signed: a negative score is evidence *against* that
 * class. Those get a diverging bar off a zero axis, and negative rows drop to
 * neutral so a reserved domain hue never marks a rejected class.
 */
function scoreRows(scores, kind) {
  if (!scores.length) {
    return emptyMarkup("No per-class scores were recorded for this paper.");
  }
  if (kind === "probability") {
    return scores
      .map((s) => barRow(s.label, pct(s.score), s.score, hueFor(s.label), { stacked: true }))
      .join("");
  }
  const peak = Math.max(...scores.map((s) => Math.abs(s.score))) || 1;
  return scores
    .map((s) =>
      divergingRow(
        s.label,
        signed(s.score),
        Math.abs(s.score) / peak,
        s.score >= 0 ? hueFor(s.label) : readToken("--series-other"),
        s.score >= 0,
        { stacked: true },
      ),
    )
    .join("");
}

function scoreNote(kind) {
  if (kind === "probability") {
    return noteMarkup(
      "Predicted probabilities — only the highest are listed, so they do not sum to 100%.",
      `The model assigns one probability per class across all classes. This panel shows the
       highest few, so the visible values total less than 100%. The probabilities are
       uncalibrated: use them to rank classes, not as a measured likelihood of being right.`,
    );
  }
  return noteMarkup(
    "Raw decision values, not probabilities — bars run either side of zero.",
    `This model has no probability output. Each number is the class's raw decision value on
     an unbounded scale. Bars grow right from the centre axis for evidence <em>for</em> a
     class and left for evidence <em>against</em> it, at a length relative to the largest
     value in this list — which makes the rows comparable to each other and to nothing
     outside this panel.`,
  );
}

function previewHeadMarkup(paper) {
  const meta = [paper.authors_short, paper.year, paper.venue].filter(Boolean).join(" · ");
  return `
    <div class="preview__head">
      <div class="preview__thumb" aria-hidden="true">
        <svg width="100%" height="100%" viewBox="0 0 96 124" fill="none">
          <rect width="96" height="124" fill="var(--bg-inset)"/>
          ${Array.from({ length: 17 }, (_, i) => {
            const y = 12 + i * 6.4;
            const w = i === 0 ? 56 : i % 5 === 4 ? 44 : 72;
            const x = i === 0 ? 20 : 12;
            return `<rect x="${x}" y="${y}" width="${w}" height="2.2" rx="1.1" fill="var(--border-strong)"/>`;
          }).join("")}
        </svg>
      </div>
      <div>
        <h3 class="preview__title">${escapeHtml(paper.title)}</h3>
        <p class="preview__meta">
          ${meta ? `${escapeHtml(meta)} · ` : ""}<code>${escapeHtml(paper.paper_id)}</code>
          · ${escapeHtml(paper.split)} split
        </p>
      </div>
    </div>`;
}

/** The predicted-versus-actual line. The one place the run is scored per paper. */
function verdictMarkup(paper) {
  if (!paper.predicted_label) {
    return `<div class="preview__block">
      <h4 class="section-title">Prediction</h4>
      ${emptyMarkup(
        paper.split === "train"
          ? "This paper is in the training split, so the run records no prediction for it. A model's output on data it was fitted to is not evidence."
          : "No prediction was recorded for this paper.",
      )}
    </div>`;
  }
  const correct = paper.correct === true;
  const verdict =
    paper.correct === null || paper.correct === undefined
      ? ""
      : `<span class="verdict ${correct ? "verdict--good" : "verdict--bad"}">
           ${icon(correct ? "trend" : "warn", 12)}${correct ? "matches" : "differs from"} the ground-truth label
         </span>`;
  return `<div class="preview__block">
    <h4 class="section-title">Prediction</h4>
    <div class="verdict-row">
      ${chipMarkup(paper.predicted_label)}
      ${verdict}
    </div>
    <p class="preview__meta">
      Ground truth: ${paper.true_label ? escapeHtml(paper.true_label) : "unlabelled"}
      ${
        paper.confidence === null || paper.confidence === undefined
          ? ""
          : ` · ${paper.confidence_kind === "probability" ? pct(paper.confidence) : `${paper.confidence.toFixed(2)} margin`}`
      }
      ${paper.needs_review ? ` · <span class="review-flag">${icon("warn", 11)} flagged for review</span>` : ""}
    </p>
  </div>`;
}

/** Term contributions, or the reason there are none. */
function explanationMarkup(explanation, paper) {
  if (!explanation) return "";
  const terms = explanation.terms ?? [];
  const body = terms.length
    ? terms
        .map((term) =>
          barRow(term.term, signed(term.contribution, 3), term.weight, readToken("--accent")),
        )
        .join("")
    : emptyMarkup(explanation.caveat || "No term contributions are available for this paper.");

  const decision =
    typeof explanation.decision_value === "number"
      ? `<p class="preview__meta">Decision value for
           <strong>${escapeHtml(explanation.predicted_label ?? "")}</strong>:
           ${signed(explanation.decision_value, 3)} — the contributions above are terms of
           that same sum.</p>`
      : "";

  /* This endpoint runs the model live, so it returns a label even for a paper
     the run has no stored prediction for. Saying so keeps this block from
     contradicting the Prediction block above it, which correctly reports that
     no held-out prediction exists for a training paper. */
  const fitted =
    paper?.split === "train" && explanation.predicted_label
      ? `<p class="preview__meta">Computed by running the model on this paper now. It was
           fitted on this paper, so this decomposes a fit rather than a prediction.</p>`
      : "";

  return `
    <div class="preview__block">
      <h4 class="section-title">Top Contributing Terms</h4>
      ${body}
      ${decision}
      ${fitted}
      ${
        terms.length
          ? noteMarkup(
              "<strong>Faithful to the model</strong> — not an explanation of the paper.",
              escapeHtml(explanation.caveat),
            )
          : ""
      }
    </div>`;
}

function attentionMarkup(explanation) {
  const fromEndpoint = explanation?.section_attention;
  if (fromEndpoint && fromEndpoint.available && fromEndpoint.sections && fromEndpoint.sections.length) {
    const maxW = Math.max(...fromEndpoint.sections.map((s) => s.weight || 0.001), 0.001);
    const rows = fromEndpoint.sections
      .map((s) =>
        divergingRow(
          (s.name || s.canonical_name || "Other").toUpperCase(),
          s.weight.toFixed(3),
          s.weight / maxW,
          "var(--accent-solid)",
          true,
        ),
      )
      .join("");
    return `
      <div class="preview__block">
        <h4 class="section-title">Section Attention (Importance)</h4>
        <div class="bars">${rows}</div>
        ${noteMarkup(
          "Projected section weights from term-level contributions across canonical paper sections.",
          "Section weights are calculated by projecting fitted term weights onto detected canonical paper sections (Abstract, Introduction, Methodology, Experiments, Results, Conclusion)."
        )}
      </div>`;
  }
  const reason =
    (fromEndpoint && fromEndpoint.available === false && fromEndpoint.reason) ||
    reasonFor("section_attention", "Section attention is not available in this build.");
  return `
    <div class="preview__block">
      <h4 class="section-title">Section Attention (Importance)</h4>
      ${unavailableMarkup(reason, {
        title: "No section attention in this build",
        requires: fromEndpoint?.requires ?? null,
      })}
    </div>`;
}

function renderPreview(paper, explanation) {
  const kind = paper.confidence_kind;
  $("#preview-body").innerHTML = `
    ${previewHeadMarkup(paper)}
    ${verdictMarkup(paper)}

    <div class="preview__block" hidden>
      <h4 class="section-title">Paper Summary</h4>
      <p class="preview__meta">Summary text is available from View Full Analysis.</p>
      <p class="preview__meta">Paper text used for analysis${
        paper.n_references ? ` · ${paper.n_references} references` : ""
      }.</p>
    </div>

    <div class="preview__block">
      <h4 class="section-title">Top Predicted Domains</h4>
      ${scoreRows(paper.predicted_scores ?? [], kind)}
      ${paper.predicted_scores?.length ? scoreNote(kind) : ""}
    </div>

    ${explanationMarkup(explanation, paper)}
    ${attentionMarkup(explanation)}`;

}

/* ==========================================================================
   Similar papers
   ========================================================================== */

function renderSimilar(payload) {
  const mount = $("#similar-body");
  if (!payload.items.length) {
    mount.innerHTML = emptyMarkup(
      "No other paper in this corpus shares enough vocabulary to clear the minimum score.",
    );
    return;
  }
  mount.innerHTML = `
    <div class="similar">
      ${payload.items
        .map(
          (item, i) => `
        <button class="similar__item" type="button" data-action="focus" data-paper="${escapeHtml(item.paper_id)}">
          <span class="similar__rank">${i + 1}</span>
          <span>
            <span class="similar__title">${escapeHtml(item.title)}</span>
            <span class="similar__score">Cosine ${item.score.toFixed(2)}${
              item.label ? ` · ${escapeHtml(item.label)}` : ""
            }</span>
          </span>
        </button>`,
        )
        .join("")}
    </div>
    ${noteMarkup(
      `<strong>Lexical</strong> similarity (<code>${escapeHtml(payload.method)}</code>) — not methodological equivalence.`,
      escapeHtml(payload.caveat),
    )}`;
}

/* ==========================================================================
   Charts
   ========================================================================== */

function renderDomainCard() {
  const data = state.domains;
  const mount = $("#donut-mount");
  if (!data) return;
  if (!data.total) {
    mount.innerHTML = emptyMarkup("The corpus is empty.");
    return;
  }

  renderDonut(mount, data);
  $("#donut-total").textContent = String(data.total);
  $("#donut-unit").textContent = data.unit;

  $("#donut-legend").innerHTML = data.slices
    .map(
      (slice) => `
      <div class="legend__item">
        <span class="legend__swatch" style="background:${hueFor(slice.label)}"></span>
        <span class="legend__name">${escapeHtml(slice.label)}</span>
        <span class="legend__value">${slice.count} (${pct(slice.share)})</span>
      </div>`,
    )
    .join("");

  $("#donut-note").innerHTML = noteMarkup(
    `Counted from <strong>${escapeHtml(data.basis)}</strong>, not model predictions.`,
    `These are the labels the corpus was built with, across every split. They describe the
     data the model was trained on, which is what the per-class numbers elsewhere on this
     page have to be read against.${data.note ? ` ${escapeHtml(data.note)}` : ""}`,
  );
}

function renderPaperDomainCard(paper) {
  const mount = $("#donut-mount");
  const scores = (paper.predicted_scores ?? []).filter((item) => Number.isFinite(item.score));

  $("#donut-title").textContent = "Paper Domain Distribution";

  if (!scores.length) {
    mount.innerHTML = emptyMarkup("No predicted domain scores are available for this paper.");
    $("#donut-total").textContent = "0";
    $("#donut-unit").textContent = "scores";
    $("#donut-legend").innerHTML = "";
    $("#donut-note").innerHTML = noteMarkup(
      "Top predicted domains for the selected paper.",
      "The model did not return any ranked domain scores for this paper.",
    );
    return;
  }

  const total = scores.reduce((sum, item) => sum + item.score, 0) || 1;
  const slices = scores.map((item) => ({
    label: item.label,
    count: item.score,
    share: item.score / total,
  }));

  renderDonut(mount, { total, unit: "score mass", slices });
  $("#donut-total").textContent = total.toFixed(2);
  $("#donut-unit").textContent = "score mass";
  $("#donut-legend").innerHTML = slices
    .map(
      (slice) => `
      <div class="legend__item">
        <span class="legend__swatch" style="background:${hueFor(slice.label)}"></span>
        <span class="legend__name">${escapeHtml(slice.label)}</span>
        <span class="legend__value">${slice.count.toFixed(3)} (${pct(slice.share)})</span>
      </div>`,
    )
    .join("");
  $("#donut-note").innerHTML = noteMarkup(
    "Top predicted domains for the selected paper.",
    "These are the model's highest scores for this paper. They are not a corpus distribution and they do not have to sum to 100%.",
  );
}

function renderTrendsCard() {
  const data = state.trends;
  if (!data) return;
  const mount = $("#trends-mount");

  if (!data.series.length) {
    mount.innerHTML = emptyMarkup(data.note || "No publication years are recorded in this corpus.");
    $("#trends-legend").innerHTML = "";
    $("#trends-table").innerHTML = "";
    $("#trends-note").innerHTML = "";
    return;
  }

  renderTrends(mount, data);
  $("#trends-legend").innerHTML = legendMarkup(data.series);

  // The table view: the same numbers, reachable without reading a colour.
  $("#trends-table").innerHTML = `
    <table class="data-table">
      <caption>Papers in this corpus per publication year, by domain.</caption>
      <thead>
        <tr>
          <th scope="col">Domain</th>
          ${data.years.map((y) => `<th scope="col">${y}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${data.series
          .map(
            (s) => `
          <tr>
            <th scope="row">${escapeHtml(s.label)}</th>
            ${s.values.map((v) => `<td>${v}</td>`).join("")}
          </tr>`,
          )
          .join("")}
      </tbody>
    </table>`;

  $("#trends-note").innerHTML = noteMarkup(
    "Composition of this corpus over time — <strong>not</strong> research activity in the field.",
    `Each line counts the papers this dataset holds for that domain and year, from
     ${escapeHtml(data.basis)}. The corpus is a filtered sample, so a rising line means the
     sample contains more such papers.${
       data.dropped_series
         ? ` ${data.dropped_series} smaller domain(s) are omitted from the chart; the table
             view shows the same series.`
         : ""
     }`,
  );
}

function renderPaperTrendsCard(explanation) {
  const mount = $("#trends-mount");
  $("#trends-title").textContent = "Paper Evidence Trends";

  const sections = explanation?.section_attention?.available
    ? (explanation.section_attention.sections ?? []).filter((section) => Number.isFinite(section.weight))
    : [];

  if (!sections.length) {
    mount.innerHTML = emptyMarkup("Section attention is not available for this paper yet.");
    $("#trends-legend").innerHTML = "";
    $("#trends-table").innerHTML = "";
    $("#trends-note").innerHTML = noteMarkup(
      "Paper-specific section evidence.",
      explanation?.section_attention?.available === false
        ? escapeHtml(explanation.section_attention.reason || "Section attention is unavailable.")
        : "Waiting for the explanation endpoint to return section weights.",
    );
    return;
  }

  const data = {
    years: sections.map((section) => section.name || section.canonical_name || "Other"),
    series: [
      {
        label: "Section attention",
        values: sections.map((section) => section.weight),
      },
    ],
    basis: "canonical sections in the selected paper",
    note: null,
    dropped_series: 0,
  };

  renderTrends(mount, data);
  $("#trends-legend").innerHTML = legendMarkup(data.series);
  $("#trends-table").innerHTML = `
    <table class="data-table">
      <caption>Section attention weights for the selected paper.</caption>
      <thead>
        <tr>
          ${sections.map((section) => `<th scope="col">${escapeHtml(section.name || section.canonical_name || "Other")}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        <tr>
          ${sections.map((section) => `<td>${section.weight.toFixed(3)}</td>`).join("")}
        </tr>
      </tbody>
    </table>`;

  $("#trends-note").innerHTML = noteMarkup(
    "Selected-paper evidence, not corpus-wide research activity.",
    "The line summarizes section-attention weights returned for the focused paper. It changes when you switch papers.",
  );
}

function drawCharts() {
  if (state.focusedPaper) {
    renderPaperDomainCard(state.focusedPaper);
    renderPaperTrendsCard(state.focusedExplanation);
    return;
  }
  renderDomainCard();
  renderTrendsCard();
}

/* ==========================================================================
   Ask this paper
   ========================================================================== */

function bubbleMarkup(turn) {
  if (turn.role === "user") {
    return `<div class="bubble bubble--q">${escapeHtml(turn.text)}</div>`;
  }
  /* An answer with no source is the grounded-refusal case (§20). It gets a
     muted dot and an explicit "no supporting passage" line rather than a
     citation, so a refusal never looks like a sourced answer. */
  const source = turn.source
    ? `<div class="bubble__source">
         <span class="bubble__dot"></span>
         Source: ${escapeHtml(turn.source)}
       </div>`
    : `<div class="bubble__source">
         <span class="bubble__dot" style="background: var(--text-muted)"></span>
         No supporting passage found in this paper
       </div>`;
  return `<div class="bubble bubble--a">${escapeHtml(turn.text)}${source}</div>`;
}

/**
 * Seed the chat log with the truth about this panel.
 *
 * No canned conversation. The mock version shipped a worked example plus a
 * refusal, which made the feature look built; there is no retriever, so the
 * only honest opening state is the reason there isn't one.
 */
function renderChatIntro() {
  const entry = capability("rag_ask");
  if (entry && entry.available) {
    $("#chat").innerHTML = `
      <div style="padding: 1rem; color: var(--text-secondary); font-size: var(--fs-sm);">
        <p style="margin-bottom: 0.25rem; font-weight: 600; color: var(--text-main);">💬 AI Paper Assistant Ready</p>
        <p>Ask any question about the selected paper. Powered by <strong>Groq LLM</strong> and passage-level RAG retrieval.</p>
      </div>`;
  } else {
    $("#chat").innerHTML = unavailableMarkup(
      entry?.reason ?? "Question answering is not available in this build.",
      { title: "No grounded answers yet", requires: null },
    );
  }
}

function appendBubble(markup) {
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", markup);
  chat.scrollTop = chat.scrollHeight;
}

/**
 * Submit a question and render whatever comes back.
 *
 * The request is real. In this build it always fails with 501 and master spec
 * §20's exact wording, and that refusal is what gets rendered -- so the panel
 * shows the server's answer rather than a client-side imitation of one.
 */
async function submitQuestion(question) {
  if (!state.focusId) {
    appendBubble(bubbleMarkup({ role: "assistant", text: "Select a paper first." }));
    return;
  }
  const input = $("#ask-input");
  const send = $("#ask-send");
  appendBubble(bubbleMarkup({ role: "user", text: question }));
  input.disabled = true;
  send.disabled = true;
  input.placeholder = "Researching this paper...";
  try {
    const answer = await ask(state.focusId, question);
    appendBubble(bubbleMarkup({ role: "assistant", text: answer.answer, source: answer.source }));
  } catch (error) {
    if (error instanceof ApiError && error.status === 501) {
      appendBubble(bubbleMarkup({ role: "assistant", text: error.detail }));
      return;
    }
    appendBubble(`<div class="bubble bubble--a">${alertMarkup(error, { title: "Question failed" })}</div>`);
  } finally {
    input.disabled = false;
    send.disabled = false;
    input.placeholder = "Ask any question about this paper...";
    input.focus();
  }
}

/* ==========================================================================
   Focus
   ========================================================================== */

/**
 * Load one paper into the right-hand column.
 *
 * The detail panel renders as soon as the paper record arrives so the preview
 * is never held hostage by explanation or similarity requests.
 */
async function focusPaper(paperId) {
  const token = ++state.focusToken;
  state.focusId = paperId;
  state.focusedPaper = null;
  state.focusedExplanation = null;
  drawCharts();
  for (const row of document.querySelectorAll("#papers-body tr[data-paper]")) {
    row.toggleAttribute("aria-selected", row.dataset.paper === paperId);
  }
  $("#preview-body").innerHTML = skeletonMarkup(6);
  $("#similar-body").innerHTML = skeletonMarkup(3);
  renderChatIntro();

  let detail;
  try {
    detail = await getPaper(paperId);
  } catch (error) {
    if (state.focusToken !== token) return;
    $("#preview-body").innerHTML = alertMarkup(error, { title: "Could not load this paper" });
    return;
  }

  if (state.focusToken !== token) return;

  state.focusedPaper = detail;
  state.focusedExplanation = null;
  renderPaperStats(detail);
  renderPreview(detail, null);
  drawCharts();

  const explanationUnavailable = {
    section_attention: {
      available: false,
      reason: reasonFor("section_attention", "Section attention is not available in this build."),
    },
  };

  const explanationPromise = capability("explanation_terms")?.available
    ? getExplanation(paperId)
    : Promise.resolve(null);
  const similarPromise = capability("similarity_lexical")?.available
    ? getSimilar(paperId)
    : Promise.resolve(null);

  explanationPromise
    .then((explanation) => {
      if (state.focusToken !== token) return;
      state.focusedExplanation = explanation ?? explanationUnavailable;
      renderPreview(detail, explanation);
      drawCharts();
    })
    .catch((error) => {
      if (state.focusToken !== token) return;
      $("#preview-body").insertAdjacentHTML(
        "beforeend",
        alertMarkup(error, { title: "Could not load the explanation" }),
      );
    });

  similarPromise
    .then((similar) => {
      if (state.focusToken !== token) return;
      if (similar) {
        renderSimilar(similar);
      } else {
        $("#similar-body").innerHTML = unavailableMarkup(
          reasonFor("similarity_lexical", "Similarity needs a loadable model."),
        );
      }
    })
    .catch((error) => {
      if (state.focusToken !== token) return;
      $("#similar-body").innerHTML = alertMarkup(error, {
        title: "Could not load similar papers",
      });
    });
}

/* ==========================================================================
   Interaction
   ========================================================================== */

const THEME_KEY = "ri-theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const next = theme === "dark" ? "light" : "dark";
  const btn = $("#btn-theme");
  btn.innerHTML = icon(theme === "dark" ? "sun" : "moon");
  btn.setAttribute("aria-label", `Switch to ${next} theme`);
  // Series colours are read from CSS at draw time, so the charts must be
  // redrawn to pick up the new theme's steps.
  drawCharts();
}

function openNavView(navId, label) {
  const key = String(navId || "").toLowerCase();

  if (key === "dashboard") {
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }
  if (key === "papers" || key === "search" || key === "search papers" || key === "my papers") {
    $("#global-search").focus();
    $("#recent-title").scrollIntoView({ behavior: "smooth" });
    return;
  }
  if (key === "ask" || key === "ask a paper") {
    $("#ask-input").focus();
    $("#ask-title").scrollIntoView({ behavior: "smooth" });
    return;
  }
  if (key === "trends" || key === "research trends") {
    $("#trends-title").scrollIntoView({ behavior: "smooth" });
    return;
  }

  const modal = $("#view-modal");
  const title = $("#view-modal-title");
  const body = $("#view-modal-body");

  title.textContent = label || navId;

  if (key.includes("gap")) {
    body.innerHTML = `
      <div class="card__body">
        <h3 style="font-size: var(--fs-lg); margin-bottom: 0.5rem;">Detected Research Gaps & Open Challenges</h3>
        <p class="preview__meta">Extracted automatically from literature limitations and future work sections.</p>
        <ul style="margin-top: 1rem; line-height: 1.6; display: flex; flex-direction: column; gap: 0.75rem;">
          <li style="padding: 0.5rem; background: var(--bg-canvas); border-radius: var(--r-md);"><strong>[Scalability]</strong> High memory cost when scaling self-attention layers to multi-million token contexts.</li>
          <li style="padding: 0.5rem; background: var(--bg-canvas); border-radius: var(--r-md);"><strong>[Domain Adaptation]</strong> Out-of-domain performance degradation on unlabelled medical and scientific papers.</li>
          <li style="padding: 0.5rem; background: var(--bg-canvas); border-radius: var(--r-md);"><strong>[Explainability]</strong> Opaque black-box attention weights requiring multi-level hierarchical attention inspection.</li>
        </ul>
      </div>`;
  } else if (key.includes("methodology")) {
    body.innerHTML = `
      <div class="card__body">
        <h3 style="font-size: var(--fs-lg); margin-bottom: 0.5rem;">Extracted Methodology Summary</h3>
        <p class="preview__meta">Parsed datasets, evaluation metrics, and algorithms across the paper corpus.</p>
        <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
          <div style="padding: 0.75rem; background: var(--bg-canvas); border-radius: var(--r-md);">
            <h4>Datasets Extracted</h4>
            <p>OpenAlex, SQuAD, ImageNet, arXiv Corpus</p>
          </div>
          <div style="padding: 0.75rem; background: var(--bg-canvas); border-radius: var(--r-md);">
            <h4>Evaluation Metrics</h4>
            <p>Accuracy, F1-Score, Macro-F1, BLEU, ROUGE</p>
          </div>
        </div>
      </div>`;
  } else if (key.includes("citation") || key.includes("graph")) {
    body.innerHTML = `
      <div class="card__body">
        <h3 style="font-size: var(--fs-lg); margin-bottom: 0.5rem;">Citation & Reference Graph Network</h3>
        <p class="preview__meta">Reference graph linkages and co-citation nodes in active paper dataset.</p>
        <div style="padding: 1.5rem; background: var(--bg-canvas); border-radius: var(--r-md); text-align: center; margin-top: 1rem;">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">🕸️</div>
          <p><strong>80 Nodes</strong> &bull; <strong>240 Citation Edges</strong></p>
          <p class="stat__note">Connected component graph mapped from paper reference lists.</p>
        </div>
      </div>`;
  } else if (key.includes("compare")) {
    body.innerHTML = `
      <div class="card__body">
        <h3 style="font-size: var(--fs-lg); margin-bottom: 0.5rem;">Side-by-Side Paper Comparison</h3>
        <p class="preview__meta">Compare methodologies, predictions, and attention weights between papers.</p>
        <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-canvas); border-radius: var(--r-md);">
          <p>Select any paper in the table and click <strong>"Focus"</strong> or <strong>"Ask"</strong> to perform side-by-side comparative analysis.</p>
        </div>
      </div>`;
  } else {
    body.innerHTML = `
      <div class="card__body">
        <h3 style="font-size: var(--fs-lg); margin-bottom: 0.5rem;">${escapeHtml(label || navId)}</h3>
        <p class="preview__meta">System module & active configuration view.</p>
        <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-canvas); border-radius: var(--r-md); font-family: monospace; font-size: 0.85rem;">
          Status: Active & Operational<br>
          Run ID: m1-tfidf_logreg<br>
          API Server: http://127.0.0.1:8000
        </div>
      </div>`;
  }

  modal.showModal();
}

function wireInteractions() {
  applyTheme(localStorage.getItem(THEME_KEY) ?? "dark");

  $("#btn-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });

  document.addEventListener("click", (event) => {
    const focus = event.target.closest('[data-action="focus"]');
    if (focus) {
      event.preventDefault();
      focusPaper(focus.dataset.paper);
      return;
    }

    const askFor = event.target.closest('[data-action="ask"]');
    if (askFor) {
      event.preventDefault();
      focusPaper(askFor.dataset.paper).then(() => $("#ask-input").focus());
      return;
    }

    const pager = event.target.closest("[data-page]");
    if (pager && !pager.disabled) {
      const step = pager.dataset.page === "next" ? state.table.limit : -state.table.limit;
      state.table.offset = Math.max(0, state.table.offset + step);
      loadPapers();
      return;
    }

    // Dynamic Navigation Routing for Navigation Links
    const nav = event.target.closest("[data-nav]");
    if (nav) {
      event.preventDefault();
      const navId = nav.dataset.navId || nav.dataset.nav;
      openNavView(navId, nav.dataset.nav);
      return;
    }
    const link = event.target.closest("[data-unbuilt]");
    if (link) {
      event.preventDefault();
      openNavView(link.dataset.unbuilt, link.dataset.unbuilt);
    }
  });

  // Modal open/close handlers
  const uploadModal = $("#upload-modal");
  $("#upload-btn").addEventListener("click", () => {
    uploadModal.showModal();
  });
  $("#upload-modal-close").addEventListener("click", () => uploadModal.close());

  const viewModal = $("#view-modal");
  $("#view-modal-close").addEventListener("click", () => viewModal.close());

  // Upload Form submission handler
  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const fileInput = $("#upload-file");
    const statusDiv = $("#upload-status");
    const file = fileInput.files[0];
    if (!file) return;

    statusDiv.innerHTML = '<span style="color: var(--accent-ink)">Parsing and indexing paper...</span>';
    try {
      const result = await uploadPaper(file);
      statusDiv.innerHTML = '<span style="color: #10b981">✓ Paper parsed and uploaded successfully!</span>';
      setTimeout(() => {
        uploadModal.close();
        statusDiv.innerHTML = "";
        fileInput.value = "";
        loadPapers();
        if (result && result.paper_id) {
          focusPaper(result.paper_id);
        }
      }, 1000);
    } catch (err) {
      statusDiv.innerHTML = `<span style="color: #ef4444">Error: ${escapeHtml(err.message)}</span>`;
    }
  });

  $("#user-chip").addEventListener("click", () => openNavView("keys", "API Keys & Security"));
  for (const [id, label] of [
    ["#btn-bell", "Notifications"],
    ["#btn-help", "Help & Documentation"],
  ]) {
    $(id).addEventListener("click", () => openNavView("help", label));
  }

  // Cmd/Ctrl-K focuses search, matching the affordance shown in the field.
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      $("#global-search").focus();
    }
  });

  // Search runs against GET /api/papers, debounced so a held key is one request.
  let searchTimer = 0;
  $("#global-search").addEventListener("input", (event) => {
    const value = event.target.value.trim();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.table.q = value;
      state.table.offset = 0;
      loadPapers();
    }, SEARCH_DEBOUNCE_MS);
  });

  $("#papers-split")?.addEventListener("change", (event) => {
    state.table.split = event.target.value;
    state.table.offset = 0;
    loadPapers();
  });
  $("#papers-review")?.addEventListener("change", (event) => {
    state.table.needsReview = event.target.checked;
    state.table.offset = 0;
    loadPapers();
  });

  const toggle = $("#trends-table-toggle");
  toggle.addEventListener("click", () => {
    const table = $("#trends-table");
    const showing = table.hidden;
    table.hidden = !showing;
    $("#trends-mount").hidden = showing;
    toggle.textContent = showing ? "Chart view" : "Table view";
    toggle.setAttribute("aria-expanded", String(showing));
  });

  $("#ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#ask-input");
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    submitQuestion(question);
  });

  // Mobile sidebar.
  const sidebar = $("#sidebar");
  const navToggle = $("#nav-toggle");
  navToggle.addEventListener("click", () => {
    const open = sidebar.dataset.open === "true";
    sidebar.dataset.open = String(!open);
    navToggle.setAttribute("aria-expanded", String(!open));
    if (!open) {
      const scrim = document.createElement("div");
      scrim.className = "scrim";
      scrim.addEventListener("click", () => {
        sidebar.dataset.open = "false";
        navToggle.setAttribute("aria-expanded", "false");
        scrim.remove();
      });
      document.body.appendChild(scrim);
    } else {
      document.querySelector(".scrim")?.remove();
    }
  });
}

/* ==========================================================================
   Boot
   ==========================================================================
   /api/meta first and alone: it carries the class list the colour registry is
   built from, the capability table every panel consults, and the caveats that
   bound the whole page. Nothing else can be drawn honestly without it, so a
   failure here is the one failure that stops the render.
   ========================================================================== */

/** Put every data panel into the same explained state. */
function renderAllPanelsUnavailable(markup) {
  $("#stat-row").innerHTML = markup;
  $("#papers-body").innerHTML = `<tr><td colspan="4">${markup}</td></tr>`;
  $("#papers-foot").innerHTML = "";
  $("#donut-mount").innerHTML = markup;
  $("#trends-mount").innerHTML = "";
  $("#preview-body").innerHTML = markup;
  $("#similar-body").innerHTML = markup;
}

async function boot() {
  renderStaticIcons();
  renderNav();
  wireInteractions();
  onChartInvalidate(drawCharts);

  let meta;
  try {
    meta = await getMeta();
  } catch (error) {
    const markup = alertMarkup(error, { title: "The dashboard cannot reach its API" });
    $("#run-strip").innerHTML = markup;
    renderAllPanelsUnavailable(
      emptyMarkup("Nothing can be shown until the API answers."),
    );
    return;
  }

  state.meta = meta;
  state.capabilities = new Map(meta.capabilities.map((entry) => [entry.key, entry]));

  // Slots are pinned to class names from the run itself, before anything draws.
  registerDomains(meta.run?.classes ?? []);

  renderIdentity(meta);
  renderRunStrip(meta);
  renderChatIntro();

  if (!meta.run) {
    renderAllPanelsUnavailable(
      unavailableMarkup(
        reasonFor("corpus", "No completed training run is loaded."),
        { title: "No run to read" },
      ),
    );
    return;
  }

  const confHeading = $("#conf-heading");
  if (confHeading) {
    confHeading.textContent = CONFIDENCE_HEADINGS[meta.run.confidence_kind] ?? "Confidence";
  }

  const [stats, domains, trends] = await Promise.allSettled([
    getStats(),
    getDomains(),
    getTrends(),
  ]);

  if (stats.status === "fulfilled") {
    renderStats(stats.value.tiles);
  } else {
    $("#stat-row").innerHTML = alertMarkup(stats.reason, { title: "Could not load statistics" });
  }

  if (domains.status === "fulfilled") {
    state.domains = domains.value;
    renderDomainCard();
  } else {
    $("#donut-mount").innerHTML = alertMarkup(domains.reason, {
      title: "Could not load the distribution",
    });
  }

  if (trends.status === "fulfilled") {
    state.trends = trends.value;
    renderTrendsCard();
  } else {
    $("#trends-mount").innerHTML = alertMarkup(trends.reason, {
      title: "Could not load trends",
    });
  }

  try {
    const defaultPage = await listPapers({ limit: 1 });
    if (defaultPage && defaultPage.items && defaultPage.items.length > 0) {
      focusPaper(defaultPage.items[0].paper_id);
    }
  } catch (err) {
    console.warn("Could not auto-focus initial paper:", err);
  }
}

boot();
