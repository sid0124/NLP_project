/* ==========================================================================
   Application
   ==========================================================================
   Renders the dashboard from the data module and wires up interaction.

   Deliberately framework-free. The project has no Node toolchain and no
   backend to talk to yet, so adding React plus a bundler would commit Phase 5
   to a stack in order to look at a layout. Every component here is a function
   from data to markup, which is the shape that ports to JSX mechanically if
   that is the eventual choice.
   ========================================================================== */

import {
  conversation,
  domainDistribution,
  focusPaper,
  navigation,
  papers,
  similarPapers,
  stats,
  trends,
  user,
} from "./data.js";
import { domainColor, isRegisteredDomain, readToken } from "./domains.js";
import { legendMarkup, onChartInvalidate, renderDonut, renderTrends } from "./charts.js";
import { icon } from "./icons.js";

/** Section keys are snake_case in the schema; present them in title case. */
const SECTION_LABELS = {
  abstract: "Abstract",
  introduction: "Introduction",
  related_work: "Related Work",
  methodology: "Methodology",
  experiments: "Experiments",
  results: "Results",
  discussion: "Discussion",
  conclusion: "Conclusion",
};

const SECTION_ICONS = {
  abstract: "file",
  introduction: "file",
  related_work: "papers",
  methodology: "gear",
  experiments: "bars",
  results: "trend",
  discussion: "chat",
  conclusion: "file",
};

const $ = (sel) => document.querySelector(sel);

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

const pct = (value) => `${(value * 100).toFixed(1)}%`;

/**
 * A collapsible caveat attached to a panel of numbers.
 *
 * The summary line is always visible, so the qualification is never hidden --
 * only its explanation is behind the disclosure. Master spec §14/§17 require
 * these claims to be bounded where the numbers are shown, not in a footnote
 * somewhere else.
 *
 * @param {string} summary Always-visible one-liner.
 * @param {string} body Expanded explanation; may contain inline markup.
 * @returns {string} HTML markup.
 */
function noteMarkup(summary, body) {
  return `<details class="note">
    <summary>${icon("info", 14)}<span>${summary}</span><span class="note__caret">${icon("chevron", 13)}</span></summary>
    <div class="note__body">${body}</div>
  </details>`;
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
  for (const id of ["recent", "donut", "trends", "preview", "similar"]) {
    const el = $(`#${id}-arrow`);
    if (el) el.innerHTML = icon("arrow_right", 14);
  }
}

function renderUser() {
  $("#greeting").textContent = `Welcome back, ${user.first_name}! 👋`;
  $("#user-name").textContent = user.full_name;
  $("#user-role").textContent = user.role;
  $("#user-initials").textContent = user.initials;

  const { used_gb, total_gb } = user.storage;
  const percent = (used_gb / total_gb) * 100;
  $("#storage-value").textContent = `${used_gb} GB / ${total_gb} GB`;
  $("#storage-fill").style.width = `${percent}%`;
  const meter = $("#storage-meter");
  meter.setAttribute("aria-valuenow", percent.toFixed(1));
  meter.setAttribute("aria-valuetext", `${used_gb} of ${total_gb} gigabytes used`);
}

function renderNav() {
  $("#nav").innerHTML = navigation
    .map(
      (group) => `
      <div class="nav-group">
        <div class="nav-group__label">${escapeHtml(group.group)}</div>
        ${group.items
          .map(
            (item) => `
          <button class="nav-item" type="button"
            ${item.built ? 'aria-current="page"' : ""}
            data-nav="${escapeHtml(item.label)}"
            data-built="${item.built}">
            ${icon(item.icon)}
            <span>${escapeHtml(item.label)}</span>
          </button>`,
          )
          .join("")}
      </div>`,
    )
    .join("");
}

function renderStats() {
  $("#stat-row").innerHTML = stats
    .map((stat) => {
      const hue = readToken(stat.hue);
      // The arrow glyph is aria-hidden, so the direction has to reach assistive
      // tech as a word -- otherwise "4.3% vs last week" is read with no sign.
      const arrow = stat.direction === "up" ? icon("arrow_up", 13) : "";
      const direction = stat.direction === "up" ? `<span class="sr-only">up </span>` : "";
      return `
      <article class="card stat">
        <div>
          <div class="stat__label">${escapeHtml(stat.label)}</div>
          <div class="stat__value${stat.value_is_text ? " stat__value--text" : ""}">${escapeHtml(stat.value)}</div>
          <div class="stat__delta${stat.direction === "up" ? " stat__delta--up" : ""}">
            ${arrow}${direction}${escapeHtml(stat.delta)}
          </div>
        </div>
        <div class="stat__glyph" style="background:${hue}22; color:${hue}">
          ${icon(stat.icon, 21)}
        </div>
      </article>`;
    })
    .join("");
}

/* ==========================================================================
   Papers table
   ========================================================================== */

function chipMarkup(domain) {
  // Registered domains carry their reserved hue; anything else gets the
  // neutral dot, so a colour never implies a series that does not exist.
  const color = isRegisteredDomain(domain) ? domainColor(domain) : readToken("--series-other");
  return `<span class="chip"><span class="chip__dot" style="background:${color}"></span>${escapeHtml(domain)}</span>`;
}

function renderPapers() {
  $("#papers-body").innerHTML = papers
    .map((paper) => {
      /* The bar is tinted by state, not by domain: at a glance the column
         should read as "how sure", and reusing a domain hue here would make
         the same colour mean two different things in one row. */
      const barColor = paper.needs_review ? readToken("--status-warning") : readToken("--status-good");
      const review = paper.needs_review
        ? `<span class="review-flag">${icon("warn", 12)} Needs review</span>`
        : "";
      return `
      <tr>
        <td>
          <div class="paper-cell">
            <span class="file-icon" aria-hidden="true">PDF</span>
            <span>
              <span class="paper-cell__title">${escapeHtml(paper.title)}</span>
              <span class="paper-cell__meta">${escapeHtml(paper.authors_short)} · ${paper.year}</span>
            </span>
          </div>
        </td>
        <td><div class="chips">${paper.domains.map(chipMarkup).join("")}</div></td>
        <td>
          <div class="conf-cell">
            <span class="conf-cell__value">${pct(paper.confidence)}</span>
            <div class="meter meter--thin">
              <div class="meter__fill" style="width:${paper.confidence * 100}%; background:${barColor}"></div>
            </div>
            ${review}
          </div>
        </td>
        <td>
          <div class="row-actions">
            <button class="icon-btn" type="button" aria-label="View analysis for ${escapeHtml(paper.title)}">${icon("bars", 16)}</button>
            <button class="icon-btn" type="button" aria-label="Ask about ${escapeHtml(paper.title)}">${icon("chat", 16)}</button>
            <button class="icon-btn" type="button" aria-label="More actions for ${escapeHtml(paper.title)}">${icon("dots", 16)}</button>
          </div>
        </td>
      </tr>`;
    })
    .join("");
}

/* ==========================================================================
   Paper preview
   ========================================================================== */

function renderPreview() {
  const paper = focusPaper;

  const domainRows = paper.predicted_domains
    .map((d) => {
      const color = isRegisteredDomain(d.label) ? domainColor(d.label) : readToken("--series-other");
      return `
      <div class="bar-row">
        <span class="bar-row__label">${escapeHtml(d.label)}</span>
        <div class="meter"><div class="meter__fill" style="width:${d.score * 100}%; background:${color}"></div></div>
        <span class="bar-row__value">${pct(d.score)}</span>
      </div>`;
    })
    .join("");

  /* One hue for every attention bar. These weights are a single measure on a
     single scale, so they are one series -- colouring each row differently
     would imply five categories that do not exist. */
  const attentionColor = readToken("--accent");
  const attentionRows = paper.section_attention
    .map(
      (s) => `
      <div class="bar-row">
        <span class="bar-row__label">${icon(SECTION_ICONS[s.section] ?? "file", 14)}${escapeHtml(SECTION_LABELS[s.section] ?? s.section)}</span>
        <div class="meter"><div class="meter__fill" style="width:${s.weight * 100}%; background:${attentionColor}"></div></div>
        <span class="bar-row__value">${s.weight.toFixed(2)}</span>
      </div>`,
    )
    .join("");

  $("#preview-body").innerHTML = `
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
          ${escapeHtml(paper.authors_short)} (${paper.year}) ·
          <a class="link" href="#" data-unbuilt="arXiv link">${escapeHtml(paper.arxiv_id)}</a>
        </p>
      </div>
    </div>

    <div class="preview__block">
      <h4 class="section-title">Abstract</h4>
      <p class="preview__abstract" id="abstract-text">${escapeHtml(paper.abstract)}</p>
      <button class="link" type="button" id="abstract-toggle" aria-expanded="true">Read Less</button>
    </div>

    <div class="preview__block">
      <h4 class="section-title">Top Predicted Domains</h4>
      ${domainRows}
      ${noteMarkup(
        "Independent per-label scores — they do not sum to 100%.",
        `Each domain is scored separately rather than competing for a single share, so a
         paper can belong to several at once. A total above or below 100% is expected.`,
      )}
    </div>

    <div class="preview__block">
      <h4 class="section-title">Section Attention (Importance)</h4>
      ${attentionRows}
      ${noteMarkup(
        "<strong>Model evidence visualisation</strong> — not proof of causality.",
        `These weights show which sections the model attended to while predicting. A high
         weight means the section carried signal the model used; it does not establish
         that the section caused the prediction, and it is not an explanation of the
         paper's content.`,
      )}
    </div>`;

  // Abstract expand/collapse, with the full text kept in the DOM so nothing
  // depends on a second request.
  const full = paper.abstract;
  const clipped = full.slice(0, 168).trimEnd() + "...";
  const textEl = $("#abstract-text");
  const toggle = $("#abstract-toggle");
  let expanded = true;
  const sync = () => {
    textEl.textContent = expanded ? full : clipped;
    toggle.textContent = expanded ? "Read Less" : "Read More";
    toggle.setAttribute("aria-expanded", String(expanded));
  };
  toggle.addEventListener("click", () => {
    expanded = !expanded;
    sync();
  });
  expanded = false;
  sync();
}

/* ==========================================================================
   Charts
   ========================================================================== */

function renderDomainCard() {
  const data = domainDistribution;
  const total = data.slices.reduce((sum, s) => sum + s.count, 0);

  renderDonut($("#donut-mount"), data);
  $("#donut-total").textContent = String(total);
  $("#donut-unit").textContent = data.unit;

  $("#donut-legend").innerHTML = data.slices
    .map(
      (slice) => `
      <div class="legend__item">
        <span class="legend__swatch" style="background:${domainColor(slice.label)}"></span>
        <span class="legend__name">${escapeHtml(slice.label)}</span>
        <span class="legend__value">${slice.count} (${((slice.count / total) * 100).toFixed(1)}%)</span>
      </div>`,
    )
    .join("");
}

function renderTrendsCard() {
  renderTrends($("#trends-mount"), trends);
  $("#trends-legend").innerHTML = legendMarkup(trends.series);

  // The table view: the same numbers, reachable without reading a colour.
  $("#trends-table").innerHTML = `
    <table class="data-table">
      <caption>Publications per year by domain.</caption>
      <thead>
        <tr>
          <th scope="col">Domain</th>
          ${trends.years.map((y) => `<th scope="col">${y}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${trends.series
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
         Source: Section ${escapeHtml(turn.source.section)}, Page ${turn.source.page}
       </div>`
    : `<div class="bubble__source">
         <span class="bubble__dot" style="background: var(--text-muted)"></span>
         No supporting passage found in this paper
       </div>`;
  return `<div class="bubble bubble--a">${escapeHtml(turn.text)}${source}</div>`;
}

function renderChat() {
  $("#chat").innerHTML = conversation.map(bubbleMarkup).join("");
}

function renderSimilar() {
  $("#similar-body").innerHTML = `
    <div class="similar">
      ${similarPapers
        .map(
          (paper, i) => `
        <a class="similar__item" href="#" data-unbuilt="Paper detail">
          <span class="similar__rank">${i + 1}</span>
          <span>
            <span class="similar__title">${escapeHtml(paper.title)}</span>
            <span class="similar__score">Similarity: ${paper.score.toFixed(2)}</span>
          </span>
        </a>`,
        )
        .join("")}
    </div>
    ${noteMarkup(
      "<strong>Semantic</strong> similarity — not methodological equivalence.",
      `Ranked by cosine distance between document embeddings. A high score means the
       papers discuss related material; it does not mean they use the same methods, share
       assumptions, or agree in their conclusions.`,
    )}`;
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

function drawCharts() {
  renderDomainCard();
  renderTrendsCard();
}

/** Placeholder for a route that has not been built. */
function announceUnbuilt(label) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const toast = document.createElement("div");
  toast.className = "toast tooltip";
  toast.dataset.visible = "true";
  toast.style.cssText =
    "left:50%; bottom:24px; top:auto; transform:translateX(-50%); min-width:0; padding:10px 16px;";
  toast.setAttribute("role", "status");
  toast.textContent = `${label} is not built yet — this is the Dashboard shell.`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

function wireInteractions() {
  applyTheme(localStorage.getItem(THEME_KEY) ?? "dark");

  $("#btn-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });

  // Unbuilt routes: say so, rather than leaving a link that looks broken.
  document.addEventListener("click", (event) => {
    const nav = event.target.closest("[data-nav]");
    if (nav && nav.dataset.built === "false") {
      event.preventDefault();
      announceUnbuilt(nav.dataset.nav);
      return;
    }
    const link = event.target.closest("[data-unbuilt]");
    if (link) {
      event.preventDefault();
      announceUnbuilt(link.dataset.unbuilt);
    }
  });

  for (const [id, label] of [
    ["#upload-btn", "Paper upload"],
    ["#btn-bell", "Notifications"],
    ["#btn-help", "Help"],
    ["#user-chip", "Account menu"],
  ]) {
    $(id).addEventListener("click", () => announceUnbuilt(label));
  }

  // Cmd/Ctrl-K focuses search, matching the affordance shown in the field.
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      $("#global-search").focus();
    }
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

  /* The composer does not answer. Retrieval needs an index and a model, and
     neither exists — a canned reply here would be indistinguishable from a
     working feature, which is exactly the thing §20 is about. */
  $("#ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#ask-input");
    if (!input.value.trim()) return;
    announceUnbuilt("Question answering");
    input.value = "";
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
   ========================================================================== */

renderStaticIcons();
renderUser();
renderNav();
renderStats();
renderPapers();
renderPreview();
renderChat();
renderSimilar();
wireInteractions();
onChartInvalidate(drawCharts);
