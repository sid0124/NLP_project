/* ==========================================================================
   Icons
   ==========================================================================
   Inline SVG path data, drawn on a 24x24 grid with a 1.7 stroke so the whole
   set has one optical weight. Inline rather than an icon font or a sprite
   sheet: no extra request, no FOUT, and `currentColor` inherits so an icon
   recolours with its container on a theme switch.
   ========================================================================== */

const PATHS = {
  home: "M3 10.5 12 3l9 7.5M5.5 9.5V20h13V9.5",
  papers: "M7 3h7l4 4v14H7zM14 3v4h4M10 12h6M10 16h6",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM16.5 16.5 21 21",
  compare: "M12 3v18M7 7H4v10h3M17 7h3v10h-3",
  chat: "M4 5h16v11H9l-5 4z",
  trend: "M3 17l5.5-5.5 4 4L21 7M21 7h-5M21 7v5",
  pie: "M12 3a9 9 0 1 0 9 9h-9z",
  graph: "M6 6.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM18 22.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM18 11.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM8 5.5l8 3M8 6.5l8 11",
  gap: "M12 3v6M12 15v6M3 12h6M15 12h6",
  extract: "M5 3h9l5 5v13H5zM14 3v5h5M9 13h6M9 17h4",
  database: "M12 3c4 0 7 1.2 7 2.7S16 8.4 12 8.4 5 7.2 5 5.7 8 3 12 3zM5 5.7v12.6c0 1.5 3 2.7 7 2.7s7-1.2 7-2.7V5.7M5 12c0 1.5 3 2.7 7 2.7s7-1.2 7-2.7",
  cube: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
  gear: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM12 2.5v2.6M12 18.9v2.6M4.2 7l2.2 1.3M17.6 15.7l2.2 1.3M4.2 17l2.2-1.3M17.6 8.3l2.2-1.3",
  key: "M15 3a6 6 0 1 1-5.2 9L8 14H5v3H2v-4l7.8-7.8A6 6 0 0 1 15 3z",
  shield: "M12 3l7 3v6c0 4-3 7.4-7 9-4-1.6-7-5-7-9V6z",
  bell: "M12 3a5 5 0 0 0-5 5c0 5-2 6-2 6h14s-2-1-2-6a5 5 0 0 0-5-5zM10.5 20a1.8 1.8 0 0 0 3 0",
  help: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9.6 9.2A2.5 2.5 0 0 1 14.5 10c0 1.7-2.5 2-2.5 3.8M12 17.2h.01",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z",
  chevron: "M6 9l6 6 6-6",
  arrow_right: "M4 12h15M13 6l6 6-6 6",
  arrow_up: "M12 19V5M6 11l6-6 6 6",
  plus: "M12 5v14M5 12h14",
  send: "M4 12l16-8-6 16-3-6z",
  bars: "M5 20V10M12 20V4M19 20v-7",
  dots: "M12 6.5h.01M12 12h.01M12 17.5h.01",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v5.5M12 7.8h.01",
  warn: "M12 4l9 15.5H3zM12 10v4M12 16.8h.01",
  menu: "M4 7h16M4 12h16M4 17h16",
  file: "M7 3h7l4 4v14H7zM14 3v4h4",
  flag: "M6 21V3.5M6 4h12l-2.2 4.2L18 12.5H6",
  lock: "M5.5 11h13v10h-13zM8.5 11V7.5a3.5 3.5 0 0 1 7 0V11",
  refresh: "M19.5 12a7.5 7.5 0 1 1-2.4-5.5M20 4v3.6h-3.6",
};

/**
 * Render an icon as an SVG string.
 *
 * `aria-hidden` is the default because these sit beside their own text label;
 * an icon that is the sole content of a control gets its name from an
 * `aria-label` on the button instead.
 *
 * @param {string} name Key from the icon set.
 * @param {number} [size=18] Pixel width and height.
 * @returns {string} SVG markup, or an empty string for an unknown name.
 */
export function icon(name, size = 18) {
  const path = PATHS[name];
  if (!path) return "";
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true"><path d="${path}"/></svg>`;
}
