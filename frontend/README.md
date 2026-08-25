# Frontend — Research Intelligence dashboard

A static, framework-free dashboard shell for the Academic Research Intelligence
System. It is the **presentation target** captured from the project owner's
mockup (recorded in [../docs/ui-target.md](../docs/ui-target.md)), built now so
later phases produce data in the shape the UI needs.

> **Every number on this screen is fictional.** The frontend now reads the API
> directly, and the API is pinned to the synthetic demo run in
> [`results/m1-tfidf_logreg`](../results/m1-tfidf_logreg/). Nothing here is a
> measured result — do not screenshot it as one.

## Running it

A dev server is required — the code uses native ES modules, and browsers block
`import` over `file://`. From the repository root:

```bash
python -m http.server 5173 --directory frontend
```

Then open <http://localhost:5173>. There is no build step, no bundler, and no
dependency to install. (The repo also ships a `.claude/launch.json` "frontend"
configuration that runs exactly this command.)

## What's real vs. mock

| Aspect | Status |
|---|---|
| Layout, theme, components, charts, interactions | **Real** — this is the shipping frontend code |
| Colour system + accessibility validation | **Real** — see below |
| Every displayed value (papers, %, counts, chat answers) | **Fictional** — served by the API from the synthetic demo run |
| Backend / API calls | **Real** — `js/app.js` reads `/api/*` directly |

## Files

| File | Responsibility |
|---|---|
| `index.html` | Structure only — every value is injected by JS into an empty mount |
| `css/tokens.css` | Single source of every colour / space / radius / type step. Two themes. |
| `css/app.css` | Layout and component styling; consumes tokens, declares no raw colours |
| `js/api.js` | Fetch wrapper and endpoint helpers for the live backend |
| `js/domains.js` | Maps each domain to a *reserved* series slot (see "Colour" below) |
| `js/icons.js` | Inline SVG icon set (no icon font, no sprite request) |
| `js/charts.js` | Hand-built SVG donut + multi-line trends. No charting library. |
| `js/app.js` | Render functions + interaction wiring; the entry module |

## Colour and accessibility

Two colour families are kept deliberately separate and must not be mixed:

- **Chrome** (`--bg-*`, `--text-*`, `--accent-*`) — the indigo/violet interface.
- **Series** (`--series-1..6`, `--series-other`) — data-bearing colours only.

The indigo accent identifies interactive chrome; a series hue identifies a
domain. Using the accent for a data series would make "the purple line" and "the
purple button" mean unrelated things.

**Colour follows the entity, never its rank.** `domains.js` pins each domain to
a fixed series slot, so filtering a chart or a change in paper counts never
repaints the surviving series — "the blue line" is always the same domain
between two page loads.

### The palette is validated, not eyeballed

The series values are **not** the ones in the source mockup. The mockup paired
blue `#3b82f6` with violet `#8b5cf6` as adjacent lines in the trends chart.
Measured, that pair sits at CVD ΔE 1.3 (deuteranopia) and normal-vision ΔE 12.0
— below the ΔE 15 floor, i.e. hard to separate even with full colour vision and
effectively identical for red-green colourblind readers. The validated
categorical order is used instead, re-checked against this project's own
surfaces.

The validator lives in the bundled `dataviz` skill. To re-check the dark-mode
series against the card surface they render on (`--bg-surface`):

```bash
node scripts/validate_palette.js "#3987e5,#d95926,#199e70,#c98500,#d55181,#9085e9" --mode dark --surface "#151a23"
```

And the light-mode series against the light card surface:

```bash
node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#4a3aa7" --mode light --surface "#ffffff"
```

(`scripts/validate_palette.js` is under the dataviz skill's base directory, not
this repo.) The light theme is a **selected** re-step, not a mechanical flip of
the dark values — flipping would drop three hues below 3:1 contrast. Where a
light-mode hue lands on a contrast WARN, identity is carried redundantly: the
donut legend shows counts **and** percentages, and Research Trends ships a table
view toggle, so nothing is conveyed by colour alone.

### Accessibility posture

- A **legend is always present** for the multi-series charts; identity is never
  colour-alone.
- Text wears text tokens (primary/secondary/muted ink), never a series colour —
  a coloured dot beside the label carries the hue.
- Every chart has a **table view** of the same numbers (Research Trends toggle;
  the donut legend doubles as its value table).
- Stat-delta direction reaches assistive tech as a word, not only the ↑ glyph.
- The storage meter is a `role="progressbar"` with `aria-valuetext`.
- `prefers-reduced-motion` collapses transitions.
- Dark is the default; the light theme persists to `localStorage` under
  `ri-theme`.

## Honest-caveat notes (spec §14/§15/§17)

Three claims the UI must not overstate are rendered as collapsible notes — the
qualification stays visible as a one-line summary, the explanation is one click
away:

- "Top Predicted Domains" scores are independent per-label values and **do not
  sum to 100%** — they are sigmoid scores, not a softmax distribution.
- Section attention is **model evidence visualisation, not proof of causality**.
- "Similar Papers" is **semantic** similarity, **not** methodological
  equivalence.

The "Ask This Paper" composer deliberately does **not** fabricate an answer on
submit. A canned reply would be indistinguishable from a working RAG feature,
which is exactly what spec §20 forbids; the mock conversation already includes an
"Information not found in the provided paper." turn to show that refusal path.

## Relationship to Milestone 1

This frontend is **not** part of Milestone 1 (§62 explicitly excludes the
frontend from M1). It was built ahead of its phase at the project owner's
request. It is now connected to the Python API, but the API is still serving the
synthetic demo run, so the whole experience remains fictional while the wiring is
real.
