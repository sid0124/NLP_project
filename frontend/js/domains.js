/* ==========================================================================
   Domain colour registry
   ==========================================================================
   Maps a domain name to a fixed categorical slot.

   The mapping is by ENTITY, not by rank. If it were assigned by sort order,
   filtering the chart or a change in paper counts would repaint the surviving
   series -- "the blue line" would silently become a different domain between
   two page loads. Because the slot is pinned to the name, a domain keeps its
   colour across the donut, the trends chart, and every table chip.

   The names come from the server, not from this file: `registerDomains` is
   called once at boot with `run.classes` from GET /api/meta. That list is the
   run's own class order, which is alphabetical and therefore stable across
   reloads and independent of how many papers each class holds -- the two
   properties the assignment needs. Hard-coding names here instead would break
   the moment the corpus was rebuilt at a different taxonomy level.

   Only six domains are coloured. That is not an oversight: six is the number of
   adjacent slots validated against both surfaces. Anything past the registry --
   including the donut's "Others" overflow bucket, which is never a class name
   and so is never registered -- falls back to a neutral treatment rather than
   getting an invented hue. Same rule as folding a 9th series into "Other".
   ========================================================================== */

/** Validated categorical order. Assigned in sequence, never cycled. */
const SLOTS = [
  "--series-1",
  "--series-2",
  "--series-3",
  "--series-4",
  "--series-5",
  "--series-6",
];

const NEUTRAL_SLOT = "--series-other";

/** Domain name -> CSS custom property holding its hue. */
let registry = new Map();

/**
 * Pin each domain to a categorical slot.
 *
 * Replaces the registry outright rather than extending it, so switching runs
 * cannot leave a stale name holding a slot the new run needs.
 *
 * @param {string[]} labels Class names in the run's own stable order.
 * @returns {number} How many received a reserved hue; the rest go neutral.
 */
export function registerDomains(labels) {
  registry = new Map();
  for (const label of labels) {
    if (registry.size >= SLOTS.length) break;
    if (typeof label !== "string" || !label || registry.has(label)) continue;
    registry.set(label, SLOTS[registry.size]);
  }
  return registry.size;
}

/**
 * Resolve a domain to a live CSS colour value.
 *
 * Reads the computed custom property rather than hard-coding hex, so a theme
 * switch moves every mark without re-rendering the data layer.
 *
 * @param {string} domain Domain label, e.g. "Computer Vision".
 * @returns {string} A CSS colour.
 */
export function domainColor(domain) {
  return readToken(registry.get(domain) ?? NEUTRAL_SLOT);
}

/**
 * Read a CSS custom property off the document root.
 *
 * @param {string} name Property name including the leading `--`.
 * @returns {string} Trimmed value, or an empty string if unset.
 */
export function readToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * True when a domain has a reserved hue rather than the neutral fallback.
 *
 * @param {string} domain Domain label.
 * @returns {boolean}
 */
export function isRegisteredDomain(domain) {
  return registry.has(domain);
}
