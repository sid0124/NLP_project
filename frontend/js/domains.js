/* ==========================================================================
   Domain colour registry
   ==========================================================================
   Maps a domain name to a fixed categorical slot.

   The mapping is by ENTITY, not by rank. If it were assigned by sort order,
   filtering the chart or a change in paper counts would repaint the surviving
   series -- "the blue line" would silently become a different domain between
   two page loads. Because the slot is pinned to the name, a domain keeps its
   colour across the donut, the trends chart, and every table chip.

   Only six domains are coloured. That is not an oversight: six is the number
   of adjacent slots validated against both surfaces. Anything past the
   registry folds into a neutral treatment rather than getting an invented
   hue, which is the same rule as folding a 9th series into "Other".
   ========================================================================== */

/** Domain name -> CSS custom property holding its hue. */
const DOMAIN_SLOTS = {
  "Machine Learning": "--series-1",
  "Computer Vision": "--series-2",
  NLP: "--series-3",
  Robotics: "--series-4",
  Bioinformatics: "--series-5",
  "Deep Learning": "--series-6",
};

const NEUTRAL_SLOT = "--series-other";

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
  const slot = DOMAIN_SLOTS[domain] ?? NEUTRAL_SLOT;
  return readToken(slot);
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
  return domain in DOMAIN_SLOTS;
}
