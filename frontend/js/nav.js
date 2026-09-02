/* ==========================================================================
   Sidebar navigation
   ==========================================================================
   The route table, kept out of the render code so adding a page is a data
   change.

   `built` describes THIS FRONTEND: whether a page exists to navigate to. It is
   deliberately separate from `capability`, which describes the BACKEND. The two
   are independent, and conflating them would make the nav lie in one direction
   or the other -- "My Papers" has a working corpus endpoint behind it and still
   has no page, while "Research Gaps" has neither.

   When an unbuilt item also names an unavailable capability, the click handler
   in app.js prefers the server's own reason over the generic message, so the
   explanation a user reads comes from src/api/capabilities.py rather than being
   duplicated here.
   ========================================================================== */

/**
 * @typedef {object} NavItem
 * @property {string} id Stable key.
 * @property {string} label Visible text.
 * @property {string} icon Key into js/icons.js.
 * @property {boolean} built Whether this frontend has the page.
 * @property {string} [capability] Matching key from GET /api/meta capabilities.
 */

/** @type {{group: string, items: NavItem[]}[]} */
export const NAV_GROUPS = [
  {
    group: "Main",
    items: [
      { id: "dashboard", label: "Dashboard", icon: "home", built: true },
      { id: "papers", label: "My Papers", icon: "papers", built: true, capability: "corpus" },
      { id: "search", label: "Search Papers", icon: "search", built: true, capability: "corpus" },
      {
        id: "compare",
        label: "Compare Papers",
        icon: "compare",
        built: true,
        capability: "comparison",
      },
      { id: "ask", label: "Ask a Paper", icon: "chat", built: true, capability: "rag_ask" },
    ],
  },
  {
    group: "Analytics",
    items: [
      { id: "trends", label: "Research Trends", icon: "trend", built: true, capability: "trends" },
      { id: "topics", label: "Topic Modeling", icon: "pie", built: true },
      { id: "citations", label: "Citation Network", icon: "graph", built: true },
      { id: "gaps", label: "Research Gaps", icon: "gap", built: true, capability: "research_gaps" },
    ],
  },
  {
    group: "Tools",
    items: [
      { id: "methodology", label: "Methodology Extractor", icon: "extract", built: true },
      { id: "datasets", label: "Dataset Tracker", icon: "database", built: true },
      { id: "models", label: "Model Tracker", icon: "cube", built: true },
    ],
  },
  {
    group: "System",
    items: [
      { id: "settings", label: "Settings", icon: "gear", built: true },
      { id: "keys", label: "API Keys", icon: "key", built: true, capability: "authentication" },
      { id: "admin", label: "Admin Panel", icon: "shield", built: true },
    ],
  },
];
