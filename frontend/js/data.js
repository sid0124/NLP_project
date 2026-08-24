/* ==========================================================================
   Mock data
   ==========================================================================
   EVERY NUMBER IN THIS FILE IS FICTIONAL.

   Nothing here was produced by a model. No classifier has been trained in this
   project yet -- Milestone 1 stops at the dataset build -- so these are the
   values from the design mockup, transcribed to drive the layout. They are not
   predictions, and the confidences and attention weights in particular are
   invented. Do not screenshot this as a result.

   The shapes, however, are real. Each export mirrors one endpoint from the
   master spec §26 contract recorded in docs/ui-target.md, so wiring the real
   backend means replacing the body of each function in api.js with a fetch --
   not reshaping the components that consume it.
   ========================================================================== */

/** Signed-in user. Real source: the session/auth layer (Phase 5). */
export const user = {
  first_name: "Sarah",
  full_name: "Sarah Johnson",
  role: "Researcher",
  initials: "SJ",
  storage: { used_gb: 42.3, total_gb: 100 },
};

/** Aggregate counters. Real source: GET /stats. */
export const stats = [
  {
    id: "papers",
    label: "Papers Analyzed",
    value: "128",
    delta: "+12 this week",
    direction: "up",
    icon: "papers",
    hue: "--series-6",
  },
  {
    id: "top_domain",
    label: "Top Predicted Domain",
    value: "Machine Learning",
    value_is_text: true,
    delta: "38 papers",
    direction: "flat",
    icon: "pie",
    hue: "--series-3",
  },
  {
    id: "confidence",
    label: "Avg. Confidence",
    value: "87.4%",
    delta: "4.3% vs last week",
    direction: "up",
    icon: "trend",
    hue: "--series-1",
  },
  {
    id: "citations",
    label: "Citations Tracked",
    value: "2,543",
    delta: "+156 this week",
    direction: "up",
    icon: "graph",
    hue: "--series-2",
  },
];

/**
 * Recent papers. Real source: GET /papers?limit=5&sort=-analyzed_at.
 *
 * `confidence` is the model's score for the top label. `needs_review` is
 * derived server-side by comparing it against the configured decision
 * threshold -- the UI must not re-derive it, or the threshold would live in
 * two places. Master spec §15: a low-confidence prediction is required to
 * render differently from a confident one.
 */
export const papers = [
  {
    paper_id: "W2401-12345",
    title: "Vision Transformer for Medical Image Segmentation: A Survey",
    authors_short: "Zhang, Y. et al.",
    year: 2024,
    domains: ["Computer Vision", "Medical AI", "Deep Learning"],
    confidence: 0.942,
    needs_review: false,
  },
  {
    paper_id: "W2404-08812",
    title: "A Novel Graph Neural Network for Drug Discovery",
    authors_short: "Kumar, A. et al.",
    year: 2024,
    domains: ["Graph ML", "Drug Discovery", "Bioinformatics"],
    confidence: 0.911,
    needs_review: false,
  },
  {
    paper_id: "W2309-55120",
    title: "Large Language Models are Few-Shot Text Classifiers",
    authors_short: "Brown, T. et al.",
    year: 2023,
    domains: ["NLP", "Transformers", "Few-shot Learning"],
    confidence: 0.893,
    needs_review: false,
  },
  {
    paper_id: "W2311-90233",
    title: "Robust Reinforcement Learning for Autonomous Driving",
    authors_short: "Lee, J. et al.",
    year: 2023,
    domains: ["Reinforcement Learning", "Autonomous Systems", "Robotics"],
    confidence: 0.867,
    needs_review: false,
  },
  /* Deliberately included: an interdisciplinary paper the model is not
     confident about. The mockup showed only 86-94% rows, which would have let
     the review state ship untested. */
  {
    paper_id: "W2402-31708",
    title: "Neuromorphic Substrates for Energy-Efficient Edge Inference",
    authors_short: "Okafor, C. et al.",
    year: 2024,
    domains: ["Robotics", "Machine Learning"],
    confidence: 0.412,
    needs_review: true,
  },
];

/**
 * The focused paper. Real source: GET /papers/{id}.
 *
 * `predicted_domains` is POST /papers/{id}/classify. The scores are
 * independent per-label sigmoid outputs, which is why they do not sum to 100%
 * -- see implication 1 in docs/ui-target.md.
 *
 * `section_attention` is GET /papers/{id}/explanation. Keys are drawn from
 * CANONICAL_SECTIONS in src/schemas/paper.py, so the panel renders whatever
 * the section-aware attention layer emits without a translation table.
 */
export const focusPaper = {
  paper_id: "W2401-12345",
  title: "Vision Transformer for Medical Image Segmentation: A Survey",
  authors_short: "Zhang, Y. et al.",
  year: 2024,
  arxiv_id: "arXiv:2401.12345",
  abstract:
    "This survey comprehensively reviews the recent advances in vision transformer models for medical image segmentation. We categorize existing methods, analyze their strengths and limitations, and identify open problems in data efficiency and cross-modality generalisation.",
  predicted_domains: [
    { label: "Computer Vision", score: 0.942 },
    { label: "Medical AI", score: 0.913 },
    { label: "Deep Learning", score: 0.887 },
    { label: "Image Segmentation", score: 0.856 },
  ],
  section_attention: [
    { section: "abstract", weight: 0.89 },
    { section: "introduction", weight: 0.76 },
    { section: "methodology", weight: 0.94 },
    { section: "experiments", weight: 0.81 },
    { section: "results", weight: 0.78 },
    { section: "conclusion", weight: 0.72 },
  ],
};

/**
 * Corpus composition. Real source: GET /stats/domains.
 *
 * "Others" is the overflow bucket, and it is a bucket rather than a 7th hue
 * on purpose -- see js/domains.js.
 */
export const domainDistribution = {
  total: 128,
  unit: "Papers",
  slices: [
    { label: "Machine Learning", count: 38 },
    { label: "Computer Vision", count: 32 },
    { label: "NLP", count: 20 },
    { label: "Robotics", count: 12 },
    { label: "Bioinformatics", count: 10 },
    { label: "Others", count: 16 },
  ],
};

/**
 * Publication counts per domain per year. Real source: GET /research/trends.
 *
 * Four series, which is the cap for a legible multi-line chart; a fifth would
 * go to small multiples rather than another line.
 */
export const trends = {
  years: [2020, 2021, 2022, 2023, 2024],
  series: [
    { label: "Machine Learning", values: [22, 35, 48, 62, 88] },
    { label: "Computer Vision", values: [18, 28, 38, 48, 64] },
    { label: "NLP", values: [10, 15, 22, 30, 40] },
    { label: "Robotics", values: [8, 11, 15, 18, 22] },
  ],
};

/**
 * Grounded question answering. Real source: POST /papers/{id}/ask.
 *
 * The second exchange is the refusal case, and it is seeded here on purpose.
 * Master spec §20 forbids fabricating an answer: when retrieval finds nothing
 * relevant the system must say so. Showing only successful answers would let
 * that path ship without a design.
 */
export const conversation = [
  { role: "user", text: "What dataset did the authors use for evaluation?" },
  {
    role: "assistant",
    text: "The authors used the BraTS 2021 dataset for brain tumor segmentation tasks.",
    source: { section: "4.2", page: 6 },
  },
  { role: "user", text: "What was the training cost in GPU-hours?" },
  {
    role: "assistant",
    text: "Information not found in the provided paper.",
    source: null,
  },
];

/**
 * Nearest neighbours in embedding space. Real source: GET /papers/{id}/similar.
 *
 * `score` is cosine similarity between document embeddings. Master spec §17:
 * this is semantic proximity, not a claim of methodological equivalence, and
 * the UI has to say so rather than letting "0.94" imply the papers do the
 * same thing.
 */
export const similarPapers = [
  {
    paper_id: "W2201-01122",
    title: "Swin UNETR: Swin Transformers for Semantic Segmentation",
    score: 0.94,
  },
  {
    paper_id: "W2103-04567",
    title: "TransBTS: Multimodal Brain Tumor Segmentation",
    score: 0.91,
  },
  {
    paper_id: "W2205-77310",
    title: "Medical Image Segmentation Using Hybrid Transformers",
    score: 0.89,
  },
];

/** Sidebar navigation. `built: false` renders as visibly unavailable. */
export const navigation = [
  {
    group: "Main",
    items: [
      { id: "dashboard", label: "Dashboard", icon: "home", built: true },
      { id: "papers", label: "My Papers", icon: "papers", built: false },
      { id: "search", label: "Search Papers", icon: "search", built: false },
      { id: "compare", label: "Compare Papers", icon: "compare", built: false },
      { id: "ask", label: "Ask a Paper", icon: "chat", built: false },
    ],
  },
  {
    group: "Analytics",
    items: [
      { id: "trends", label: "Research Trends", icon: "trend", built: false },
      { id: "topics", label: "Topic Modeling", icon: "pie", built: false },
      { id: "citations", label: "Citation Network", icon: "graph", built: false },
      { id: "gaps", label: "Research Gaps", icon: "gap", built: false },
    ],
  },
  {
    group: "Tools",
    items: [
      { id: "methodology", label: "Methodology Extractor", icon: "extract", built: false },
      { id: "datasets", label: "Dataset Tracker", icon: "database", built: false },
      { id: "models", label: "Model Tracker", icon: "cube", built: false },
    ],
  },
  {
    group: "System",
    items: [
      { id: "settings", label: "Settings", icon: "gear", built: false },
      { id: "keys", label: "API Keys", icon: "key", built: false },
      { id: "admin", label: "Admin Panel", icon: "shield", built: false },
    ],
  },
];
