/* ==========================================================================
   Charts
   ==========================================================================
   Hand-built SVG. No charting library: two chart types do not justify a
   dependency, and the master spec's "no libraries without justification" rule
   applies to the frontend as much as the Python side.

   Both charts ship three things by default:
     - a hover layer (an SVG chart is interactive; a static one is a picture),
     - a legend, so identity is never carried by colour alone,
     - a table view of the same numbers, which is also the documented relief
       for the sub-3:1 light-mode hues.
   ========================================================================== */

import { domainColor, readToken } from "./domains.js";

/** Shared floating tooltip, created once. */
let tooltipEl = null;

function tooltip() {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tooltip";
    tooltipEl.setAttribute("role", "status");
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}

function showTooltip(html, x, y) {
  const el = tooltip();
  el.innerHTML = html;
  el.dataset.visible = "true";
  // Measure, then flip near the viewport edge so the tip is never clipped.
  const box = el.getBoundingClientRect();
  const left = Math.min(x + 14, window.innerWidth - box.width - 8);
  const top = Math.max(8, Math.min(y - box.height / 2, window.innerHeight - box.height - 8));
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
}

function hideTooltip() {
  if (tooltipEl) tooltipEl.dataset.visible = "false";
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Create an SVG element with attributes applied.
 *
 * @param {string} tag Element name.
 * @param {Record<string, string|number>} [attrs] Attributes.
 * @returns {SVGElement}
 */
function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

/* ==========================================================================
   Donut
   ========================================================================== */

/**
 * Render a donut chart with a centred total.
 *
 * Wedges are separated by a 2px gap in the surface colour rather than a
 * stroke: a stroke would add ink that is not data, and the gap is what makes
 * neighbouring wedges read as distinct.
 *
 * @param {HTMLElement} mount Container element.
 * @param {{total: number, unit: string, slices: {label: string, count: number}[]}} data
 */
export function renderDonut(mount, data) {
  const size = 168;
  const stroke = 26;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const total = data.slices.reduce((sum, s) => sum + s.count, 0);

  const root = svg("svg", {
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`,
    class: "chart",
    role: "img",
    "aria-label": `Domain distribution across ${total} papers.`,
  });

  // Rotate so the first wedge starts at 12 o'clock.
  const group = svg("g", { transform: `rotate(-90 ${size / 2} ${size / 2})` });
  const GAP_PX = 2;
  let offset = 0;

  data.slices.forEach((slice) => {
    const fraction = slice.count / total;
    const length = fraction * circumference;
    const arc = svg("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: domainColor(slice.label),
      "stroke-width": stroke,
      // The gap is subtracted from the arc, so the wedge angle stays truthful.
      "stroke-dasharray": `${Math.max(0, length - GAP_PX)} ${circumference - Math.max(0, length - GAP_PX)}`,
      "stroke-dashoffset": -offset,
    });
    arc.style.cursor = "pointer";

    const share = ((fraction * 100).toFixed(1) + "%").padStart(1);
    arc.addEventListener("mousemove", (event) => {
      showTooltip(
        `<div class="tooltip__title">${escapeHtml(slice.label)}</div>
         <div class="tooltip__row">
           <span class="tooltip__swatch" style="background:${domainColor(slice.label)}"></span>
           <span>Papers</span>
           <span class="tooltip__value">${slice.count} · ${share}</span>
         </div>`,
        event.clientX,
        event.clientY,
      );
    });
    arc.addEventListener("mouseleave", hideTooltip);

    group.appendChild(arc);
    offset += length;
  });

  root.appendChild(group);
  mount.replaceChildren(root);
}

/* ==========================================================================
   Multi-line trends
   ========================================================================== */

/**
 * Render a multi-series line chart with a crosshair and a shared tooltip.
 *
 * Single y-axis by construction: all series are the same measure (papers per
 * year), so they belong on one scale. A second axis would let two unrelated
 * scales imply a crossing that is an artefact of scaling.
 *
 * @param {HTMLElement} mount Container element.
 * @param {{years: number[], series: {label: string, values: number[]}[]}} data
 */
export function renderTrends(mount, data) {
  const width = mount.clientWidth || 460;
  const height = 214;
  const pad = { top: 10, right: 14, bottom: 24, left: 34 };
  const plotW = Math.max(10, width - pad.left - pad.right);
  const plotH = height - pad.top - pad.bottom;

  const peak = Math.max(...data.series.flatMap((s) => s.values));
  // Round the axis top to a clean number so ticks land on readable values.
  const step = peak > 50 ? 25 : 10;
  const yMax = Math.ceil(peak / step) * step;

  const xAt = (i) =>
    pad.left + (data.years.length === 1 ? plotW / 2 : (i / (data.years.length - 1)) * plotW);
  const yAt = (v) => pad.top + plotH - (v / yMax) * plotH;

  const root = svg("svg", {
    width: "100%",
    height,
    viewBox: `0 0 ${width} ${height}`,
    class: "chart",
    preserveAspectRatio: "none",
    role: "img",
    "aria-label": `Publications per year by domain, ${data.years[0]} to ${data.years.at(-1)}.`,
  });

  // Gridlines: hairline, solid, one step off the surface — recessive.
  const grid = svg("g", { class: "chart__grid" });
  for (let v = 0; v <= yMax; v += step) {
    const y = yAt(v);
    grid.appendChild(svg("line", { x1: pad.left, y1: y, x2: width - pad.right, y2: y }));
    const label = svg("text", { x: pad.left - 8, y: y + 3, "text-anchor": "end" });
    label.textContent = String(v);
    grid.appendChild(label);
  }
  root.appendChild(grid);

  data.years.forEach((year, i) => {
    const label = svg("text", { x: xAt(i), y: height - 6, "text-anchor": "middle" });
    label.textContent = String(year);
    root.appendChild(label);
  });

  root.appendChild(
    svg("line", {
      class: "chart__axis",
      x1: pad.left,
      y1: pad.top + plotH,
      x2: width - pad.right,
      y2: pad.top + plotH,
    }),
  );

  const crosshair = svg("line", {
    class: "chart__crosshair",
    y1: pad.top,
    y2: pad.top + plotH,
    opacity: 0,
  });
  root.appendChild(crosshair);

  data.series.forEach((series) => {
    const color = domainColor(series.label);
    const d = series.values.map((v, i) => `${i ? "L" : "M"}${xAt(i)} ${yAt(v)}`).join(" ");
    root.appendChild(svg("path", { class: "chart__line", d, stroke: color }));
    series.values.forEach((v, i) => {
      root.appendChild(
        svg("circle", { class: "chart__dot", cx: xAt(i), cy: yAt(v), r: 4, fill: color }),
      );
    });
  });

  // One hit band per year, full plot height — a target far larger than the
  // 8px dots, so hovering never requires precision aiming.
  const bandW = plotW / Math.max(1, data.years.length - 1);
  data.years.forEach((year, i) => {
    const band = svg("rect", {
      class: "chart__hit",
      x: xAt(i) - bandW / 2,
      y: pad.top,
      width: bandW,
      height: plotH,
    });
    band.addEventListener("mousemove", (event) => {
      crosshair.setAttribute("x1", xAt(i));
      crosshair.setAttribute("x2", xAt(i));
      crosshair.setAttribute("opacity", "1");
      const rows = data.series
        .map(
          (s) =>
            `<div class="tooltip__row">
               <span class="tooltip__swatch" style="background:${domainColor(s.label)}"></span>
               <span>${escapeHtml(s.label)}</span>
               <span class="tooltip__value">${s.values[i]}</span>
             </div>`,
        )
        .join("");
      showTooltip(
        `<div class="tooltip__title">${year}</div>${rows}`,
        event.clientX,
        event.clientY,
      );
    });
    band.addEventListener("mouseleave", () => {
      crosshair.setAttribute("opacity", "0");
      hideTooltip();
    });
    root.appendChild(band);
  });

  mount.replaceChildren(root);
}

/**
 * Build a legend where each item is a coloured key plus a text label.
 *
 * @param {{label: string}[]} items Series or slices.
 * @returns {string} HTML markup.
 */
export function legendMarkup(items) {
  return items
    .map(
      (item) =>
        `<span class="chart-legend__item">
           <span class="chart-legend__key" style="background:${domainColor(item.label)}"></span>
           ${escapeHtml(item.label)}
         </span>`,
    )
    .join("");
}

/**
 * Redraw charts on viewport resize.
 *
 * Width is baked into the SVG viewBox at draw time, so a resize needs a
 * redraw rather than a reflow. Theme changes also invalidate the charts --
 * colours are read from CSS custom properties when the marks are built -- but
 * that redraw is triggered by the theme toggle in app.js, which owns the
 * `data-theme` switch, not here.
 *
 * @param {() => void} redraw Callback that re-renders every chart.
 */
export function onChartInvalidate(redraw) {
  let frame = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(redraw);
  });
}

export { readToken };
