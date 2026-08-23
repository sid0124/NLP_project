# Frontend Target — Research Intelligence Dashboard

Captured from a mockup supplied by the project owner on 2026-08-23. This is the
**Phase 5 / §30 / §49 deliverable**, recorded now so the requirement is not lost
and so earlier phases produce the data it needs. Nothing here is built in
Milestone 1.

## Visual direction

Dark theme (near-black canvas, slightly lighter elevated cards, hairline
borders), indigo/violet primary accent with green/amber/cyan used only to
distinguish data series, card-based layout, persistent left sidebar, global
search with a `⌘K` affordance, dense but calm information hierarchy. Progress
bars rather than gauges for score display.

## Chrome

- **Brand lockup** (sidebar top): mark + "Research Intelligence" over the
  subtitle "AI-Powered Academic Insights".
- **Top bar**: centred global search (`Search papers, topics, authors, etc...`
  with a `⌘K` chip), then notifications bell, help, light/dark toggle, and a user
  chip showing avatar, name, and role (`Researcher`) with a dropdown chevron.
- **Greeting row**: `Welcome back, {first_name}! 👋` over a one-line subtitle.
- **Sidebar footer**: storage meter — `Storage Used`, `42.3 GB / 100 GB`, bar.

## Navigation (left sidebar)

| Group | Items | Backing phase |
|---|---|---|
| Main | Dashboard, My Papers, Search Papers, Compare Papers, Ask a Paper | 5, 4, 7, 6 |
| Analytics | Research Trends, Topic Modeling, Citation Network, Research Gaps | 7, 8 |
| Tools | Methodology Extractor, Dataset Tracker, Model Tracker | 7, MLOps |
| System | Settings, API Keys, Admin Panel | 5 |

Plus a primary **Upload Paper** call to action, above the group list.

## Dashboard components

Layout: a full-width stat row, then a 2-column body (≈ 58 / 42 split) that
collapses to one column on narrow viewports.

1. **Stat tiles** (4 across) — each a label, a large value, a delta line, and a
   tinted glyph:
   - Papers Analyzed · `128` · `+12 this week`
   - Top Predicted Domain · `Machine Learning` · `38 papers`
   - Avg. Confidence · `87.4%` · `↑ 4.3% vs last week`
   - Citations Tracked · `2,543` · `+156 this week`
2. **Recent Paper Analysis** table — columns *Paper* (file-type icon, title,
   `Zhang, Y. et al.`, year), *Predicted Domains* (2–3 chips, colour-coded per
   domain), *Confidence* (percentage over an inline bar), *Actions* (view
   analysis, ask, overflow menu). Footer link `View All Papers →`.
3. **Paper Analysis Preview** — page thumbnail beside title, authors, year,
   arXiv link, abstract excerpt with `Read More`; then **Top Predicted Domains**
   (labelled bars + %) and **Section Attention (Importance)** (one row per
   canonical section, icon + bar + weight). Header link `View Full Analysis →`.
4. **Domain Distribution** — donut with the total in the hole (`128 Papers`) and
   a legend listing each domain with its count and share, plus an `Others`
   bucket. Footer link `View All →`.
5. **Research Trends** — multi-series line chart, one line per domain, years on
   the x-axis, markers at each point. Footer link `View Trends →`.
6. **Ask This Paper** — chat panel: right-aligned question bubble, left-aligned
   answer bubble whose footer carries a status dot and `Source: Section 4.2,
   Page 6`; composer reading `Ask any question about this paper...` with a send
   button.
7. **Similar Papers** — ranked list (`1`, `2`, `3`), each with title and
   `Similarity: 0.94`. Header link `View All →`.

## What this changes about *earlier* phases

The mockup is not merely cosmetic; it constrains upstream data design. Three
concrete implications, recorded here rather than discovered late:

1. **Multi-label is the primary presentation mode, not an extension.**
   Every paper row shows 2–3 domain chips, and "Top Predicted Domains" lists
   four domains at 94.2 / 91.3 / 88.7 / 85.6 % — values that do not sum to 100 %,
   so these are independent sigmoid scores, not a softmax distribution.
   *Status:* already accommodated. `labels.mode` is configuration-driven and
   `PaperDocument.labels_at()` builds score-thresholded label sets.

2. **Chunks need page numbers.** "Source: Section 4.2, Page 6" requires page
   provenance on every retrievable unit.
   *Status:* `Paragraph` has no `page_number` field yet. It is intentionally
   **not** added now — no parser exists to populate it, and an always-`None`
   field is worse than an absent one. Add it with the PDF parser.

3. **Label granularity is finer than the current default.** The chips read
   "NLP", "Computer Vision", "Reinforcement Learning", "Graph ML",
   "Medical AI" — closer to the OpenAlex *topic* level (~4,500 values) or a
   curated shortlist than to the 11 CS *subfields* now configured.
   *Status:* de-risked rather than solved. Every ingested paper retains its full
   `topic → subfield → field → domain` chain, so the label space can be re-cut
   at a different granularity **from cached data, with no re-fetch**. Choosing
   the production label set is a deliberate decision for Milestone 3, since
   ~4,500 raw topics is not a tractable classification target without curation.

4. **The section-attention panel fixes the section vocabulary.** It lists
   Abstract, Introduction, Methodology, Experiments, Results, Conclusion — one
   weight per section, which is exactly the output shape of the section-aware
   attention layer that is this project's novel contribution (§7/§12).
   *Status:* accommodated. All six already appear in `CANONICAL_SECTIONS` in
   [src/schemas/paper.py](../src/schemas/paper.py), and the schema keeps the
   `Paper → Section → Paragraph` hierarchy rather than flattening to a string,
   so the panel has somewhere to read its weights from.

## Data contract the dashboard implies

Endpoints from spec §26 that the dashboard consumes, and the phase each lands in:

- `GET /papers` + `GET /papers/{id}` — list and detail (Phase 5)
- `POST /papers/{id}/classify` → labels **with per-label confidence** (Phase 2–3)
- `GET /papers/{id}/explanation` → per-section attention weights (Phase 3)
- `GET /papers/{id}/similar` → ranked neighbours with scores (Phase 4)
- `POST /papers/{id}/ask` → answer plus section/page evidence (Phase 6)
- `GET /research/trends` → per-year, per-domain counts (Phase 7)
- Aggregate counters for the stat tiles (Phase 5)

## Honest caveats to preserve in the UI

The mockup displays confidence percentages and attention weights prominently.
Per spec §14/§15/§17 these must be labelled truthfully:

- attention weights are **model evidence visualisation**, not proof of causality;
- similarity is **semantic**, not methodological equivalence;
- low-confidence predictions must surface a human-review state rather than being
  rendered identically to high-confidence ones.
