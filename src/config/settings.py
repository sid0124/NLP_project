"""Typed, layered configuration.

Configuration resolves in three layers, later layers overriding earlier ones:

1. ``configs/config.yaml``, ``configs/dataset.yaml``, ``configs/model.yaml``,
   ``configs/api.yaml``
2. Environment variables (``.env`` or the real environment)
3. Explicit keyword overrides passed by a CLI script

Every model sets ``extra="forbid"``, so a typo in a YAML key fails loudly at
load time instead of being silently ignored and leaving a stale default in
force. Nothing in the codebase reads a tunable from anywhere but here
(master spec §32).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.io import PROJECT_ROOT, read_yaml, resolve_path

__all__ = [
    "ApiConfig",
    "AppConfig",
    "BaselineConfig",
    "CorsConfig",
    "DatasetConfig",
    "DecisionConfig",
    "EnvSettings",
    "LabelsConfig",
    "ModelConfig",
    "PathsConfig",
    "SecurityConfig",
    "ServerConfig",
    "Settings",
    "SplitConfig",
    "load_settings",
]

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)

LabelMode = Literal["multiclass", "multilabel"]
TaxonomyLevel = Literal["field", "subfield"]


# ===========================================================================
# configs/config.yaml
# ===========================================================================
class ProjectConfig(BaseModel):
    """Project identity and the global random seed."""

    model_config = _STRICT

    name: str
    version: str
    seed: int = Field(ge=0)


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths resolve against the project root."""

    model_config = _STRICT

    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    sample_dir: Path
    external_dir: Path
    results_dir: Path
    logs_dir: Path

    def resolved(self, name: str) -> Path:
        """Return an absolute path for the named directory attribute.

        Args:
            name: Attribute name, e.g. ``"processed_dir"``.

        Raises:
            AttributeError: If ``name`` is not a configured path.
        """
        value = getattr(self, name)
        return resolve_path(value)


class IngestionConfig(BaseModel):
    """Selects which registered ingestion source is active."""

    model_config = _STRICT

    source: str


class MultiLabelConfig(BaseModel):
    """Thresholds governing multi-label target construction."""

    model_config = _STRICT

    min_topic_score: float = Field(ge=0.0, le=1.0)
    max_labels_per_paper: int = Field(ge=1)


class LabelsConfig(BaseModel):
    """Label space definition and filtering rules."""

    model_config = _STRICT

    mode: LabelMode
    taxonomy_level: TaxonomyLevel
    min_class_count: int = Field(ge=1)
    exclude_classes: list[str] = Field(default_factory=list)
    multilabel: MultiLabelConfig

    @property
    def is_multilabel(self) -> bool:
        """True when the pipeline should emit label sets rather than one label."""
        return self.mode == "multilabel"


class SplitConfig(BaseModel):
    """Train/validation/test proportions."""

    model_config = _STRICT

    train_ratio: float = Field(gt=0.0, lt=1.0)
    val_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    stratify: bool

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        """Reject split ratios that do not sum to 1.

        A silent normalisation here would produce splits that disagree with the
        documented configuration, so this is an error rather than a warning.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        return self


class TextConfig(BaseModel):
    """Which paper fields form the classifier input, and how they are cleaned."""

    model_config = _STRICT

    fields: list[str] = Field(min_length=1)
    lowercase: bool
    strip_urls: bool
    normalize_unicode: bool
    collapse_whitespace: bool


class LoggingConfig(BaseModel):
    """Root log level."""

    model_config = _STRICT

    level: str = "INFO"


class AppConfig(BaseModel):
    """Root model for ``configs/config.yaml``."""

    model_config = _STRICT

    project: ProjectConfig
    paths: PathsConfig
    ingestion: IngestionConfig
    labels: LabelsConfig
    split: SplitConfig
    text: TextConfig
    logging: LoggingConfig


# ===========================================================================
# configs/dataset.yaml
# ===========================================================================
class LabelTarget(BaseModel):
    """One class in the label space, with its OpenAlex short id."""

    model_config = _STRICT

    id: str
    name: str


class TaxonomyConfig(BaseModel):
    """A label space drawn from one level of the OpenAlex topic hierarchy."""

    model_config = _STRICT

    filter_key: str
    targets: list[LabelTarget] = Field(min_length=2)

    @property
    def class_names(self) -> list[str]:
        """Class display names, in configured order."""
        return [t.name for t in self.targets]


class OpenAlexFilters(BaseModel):
    """Server-side filters applied to every OpenAlex works query."""

    model_config = _STRICT

    from_publication_date: str
    to_publication_date: str
    language: str
    type: str
    has_abstract: bool


class RequestConfig(BaseModel):
    """HTTP retry and rate-limiting behaviour."""

    model_config = _STRICT

    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_factor: float = Field(gt=0)
    retry_on_status: list[int]
    sleep_between_requests: float = Field(ge=0)


class OpenAlexConfig(BaseModel):
    """Everything needed to fetch a labelled corpus from OpenAlex."""

    model_config = _STRICT

    base_url: str
    polite_pool: bool
    per_page: int = Field(gt=0, le=200)  # 200 is the OpenAlex maximum
    max_records_per_class: int = Field(gt=0)
    filters: OpenAlexFilters
    select: list[str] = Field(min_length=1)
    request: RequestConfig
    subfield: TaxonomyConfig
    field: TaxonomyConfig

    def taxonomy(self, level: TaxonomyLevel) -> TaxonomyConfig:
        """Return the label-space definition for the given taxonomy level."""
        return self.subfield if level == "subfield" else self.field


class ValidationConfig(BaseModel):
    """Data-quality thresholds (master spec §36)."""

    model_config = _STRICT

    min_title_chars: int = Field(ge=0)
    min_abstract_chars: int = Field(ge=0)
    max_abstract_chars: int = Field(gt=0)
    allowed_languages: list[str]
    max_invalid_fraction: float = Field(ge=0.0, le=1.0)
    imbalance_warn_ratio: float = Field(gt=1.0)

    @model_validator(mode="after")
    def _abstract_bounds_ordered(self) -> ValidationConfig:
        """Ensure the abstract-length window is non-empty."""
        if self.min_abstract_chars >= self.max_abstract_chars:
            raise ValueError(
                f"min_abstract_chars ({self.min_abstract_chars}) must be less than "
                f"max_abstract_chars ({self.max_abstract_chars})"
            )
        return self


class ExactDedupConfig(BaseModel):
    """Exact-duplicate detection keys, tried in order."""

    model_config = _STRICT

    enabled: bool
    keys: list[str]


class NearDuplicateConfig(BaseModel):
    """Shingle-based near-duplicate detection with bottom-k sketch blocking."""

    model_config = _STRICT

    enabled: bool
    shingle_size: int = Field(ge=1)
    jaccard_threshold: float = Field(gt=0.0, le=1.0)
    #: Bottom-k shingle hashes indexed per document for candidate generation.
    #: ``0`` disables blocking and compares every pair — correct but quadratic,
    #: so it is intended for small corpora and tests only.
    sketch_size: int = Field(ge=0)
    #: Sketch buckets larger than this are skipped as boilerplate.
    max_bucket_size: int = Field(ge=2)


class DedupConfig(BaseModel):
    """Deduplication settings; runs before splitting to prevent leakage."""

    model_config = _STRICT

    exact: ExactDedupConfig
    near_duplicate: NearDuplicateConfig


class DatasetConfig(BaseModel):
    """Root model for ``configs/dataset.yaml``."""

    model_config = _STRICT

    openalex: OpenAlexConfig
    validation: ValidationConfig
    dedup: DedupConfig


# ===========================================================================
# configs/model.yaml
# ===========================================================================
class VectorizerConfig(BaseModel):
    """A named feature extractor. ``params`` passes straight to scikit-learn."""

    model_config = _STRICT

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ClassifierConfig(BaseModel):
    """A classifier head. ``params`` passes straight to scikit-learn."""

    model_config = _STRICT

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class BaselineConfig(BaseModel):
    """One end-to-end baseline: a named vectorizer plus a classifier."""

    model_config = _STRICT

    display_name: str
    vectorizer: str
    classifier: ClassifierConfig


class TrainingConfig(BaseModel):
    """Run-level training behaviour."""

    model_config = _STRICT

    eval_split: Literal["val", "test"]
    score_test: bool
    save_model: bool


class PlotsConfig(BaseModel):
    """Which figures to render, and how."""

    model_config = _STRICT

    confusion_matrix: bool
    class_distribution: bool
    normalize_confusion_matrix: bool
    dpi: int = Field(gt=0)
    figure_format: str


class EvaluationConfig(BaseModel):
    """Metrics and figures produced for every run (master spec §13)."""

    model_config = _STRICT

    averages: list[str] = Field(min_length=1)
    primary_metric: str
    per_class_metrics: bool
    hamming_loss: bool
    plots: PlotsConfig


class ModelConfig(BaseModel):
    """Root model for ``configs/model.yaml``."""

    model_config = _STRICT

    vectorizers: dict[str, VectorizerConfig]
    baselines: dict[str, BaselineConfig]
    training: TrainingConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _baselines_reference_known_vectorizers(self) -> ModelConfig:
        """Fail fast when a baseline names a vectorizer that does not exist."""
        known = set(self.vectorizers)
        for name, baseline in self.baselines.items():
            if baseline.vectorizer not in known:
                raise ValueError(
                    f"baseline '{name}' references unknown vectorizer "
                    f"'{baseline.vectorizer}'; defined vectorizers: {sorted(known)}"
                )
        return self

    def baseline(self, name: str) -> BaselineConfig:
        """Look up a baseline by key.

        Raises:
            KeyError: If no baseline with that key is configured.
        """
        try:
            return self.baselines[name]
        except KeyError:
            raise KeyError(
                f"Unknown baseline '{name}'. Available: {sorted(self.baselines)}"
            ) from None

    def vectorizer_for(self, baseline_name: str) -> VectorizerConfig:
        """Return the vectorizer configuration used by a baseline."""
        return self.vectorizers[self.baseline(baseline_name).vectorizer]


# ===========================================================================
# configs/api.yaml
# ===========================================================================
class ServerConfig(BaseModel):
    """Where the API listens, and whether it also serves the dashboard."""

    model_config = _STRICT

    host: str
    port: int = Field(gt=0, le=65535)
    serve_frontend: bool
    frontend_dir: Path


class CorsConfig(BaseModel):
    """Cross-origin policy (master spec §40)."""

    model_config = _STRICT

    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]

    @property
    def allows_any_origin(self) -> bool:
        """True when the configuration permits requests from any origin."""
        return "*" in self.allow_origins

    @model_validator(mode="after")
    def _wildcard_origin_forbids_credentials(self) -> CorsConfig:
        """Reject the wildcard-plus-credentials combination.

        Browsers refuse it anyway, so allowing it here would only produce
        requests that fail at the client with an opaque CORS error instead of a
        configuration error naming the cause.
        """
        if self.allows_any_origin and self.allow_credentials:
            raise ValueError(
                "cors.allow_origins ['*'] cannot be combined with "
                "allow_credentials: true — list the exact origins instead."
            )
        return self


class SecurityConfig(BaseModel):
    """Request limits and the authentication-ready hook (master spec §40)."""

    model_config = _STRICT

    require_api_key: bool
    api_key_header: str = Field(min_length=1)
    max_request_bytes: int = Field(gt=0)
    max_text_chars: int = Field(gt=0)
    send_security_headers: bool


class RunsConfig(BaseModel):
    """Which training run the API reads."""

    model_config = _STRICT

    #: ``None`` falls back to ``paths.results_dir`` so the layout is defined once.
    results_dir: Path | None = None
    #: ``None`` selects the most recently finished run.
    default_run_id: str | None = None


class PaginationConfig(BaseModel):
    """Page-size bounds for list endpoints."""

    model_config = _STRICT

    default_page_size: int = Field(gt=0)
    max_page_size: int = Field(gt=0)

    @model_validator(mode="after")
    def _default_within_max(self) -> PaginationConfig:
        """Ensure the default page size is actually requestable."""
        if self.default_page_size > self.max_page_size:
            raise ValueError(
                f"pagination.default_page_size ({self.default_page_size}) exceeds "
                f"max_page_size ({self.max_page_size})"
            )
        return self


class DecisionConfig(BaseModel):
    """Human-review thresholds (master spec §15).

    Two thresholds because the two configured classifiers expose different
    quantities: a probability in [0, 1] from logistic regression, and an
    unbounded decision margin from LinearSVC. One bar cannot serve both.
    """

    model_config = _STRICT

    review_threshold: float = Field(ge=0.0, le=1.0)
    review_margin_threshold: float = Field(ge=0.0)


class SimilarityConfig(BaseModel):
    """Nearest-neighbour retrieval over the run's fitted TF-IDF space."""

    model_config = _STRICT

    top_k: int = Field(gt=0)
    min_score: float = Field(ge=0.0, le=1.0)


class ExplanationConfig(BaseModel):
    """How many per-term contributions an explanation returns."""

    model_config = _STRICT

    top_k_terms: int = Field(gt=0)


class TrendsConfig(BaseModel):
    """Per-year, per-class publication counts."""

    model_config = _STRICT

    max_series: int = Field(gt=0)
    min_year: int = Field(gt=0)


class DistributionConfig(BaseModel):
    """Class-distribution bucketing for the donut chart."""

    model_config = _STRICT

    max_slices: int = Field(gt=0)
    other_label: str = Field(min_length=1)


class StorageConfig(BaseModel):
    """Disk-usage meter: measured usage against an operator-set quota."""

    model_config = _STRICT

    quota_gb: float = Field(gt=0)
    measured_dirs: list[Path] = Field(min_length=1)


class ApiConfig(BaseModel):
    """Root model for ``configs/api.yaml``."""

    model_config = _STRICT

    server: ServerConfig
    cors: CorsConfig
    security: SecurityConfig
    runs: RunsConfig
    pagination: PaginationConfig
    decision: DecisionConfig
    similarity: SimilarityConfig
    explanation: ExplanationConfig
    trends: TrendsConfig
    distribution: DistributionConfig
    storage: StorageConfig


# ===========================================================================
# Environment overlay
# ===========================================================================
class EnvSettings(BaseSettings):
    """Environment-variable overrides, optionally sourced from ``.env``.

    Only settings that genuinely vary per machine or per deployment live here.
    Everything structural stays in YAML so it is reviewable in version control.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openalex_mailto: str | None = None
    aris_seed: int | None = None
    aris_log_level: str | None = None
    aris_data_dir: Path | None = None

    # -- API deployment overrides -------------------------------------------
    #: Deployment label, surfaced by ``/api/health`` and the dashboard footer.
    #: Names the environment only; it must never gate behaviour, or the code path
    #: that runs in production becomes the one least exercised in development.
    aris_env: str = "development"
    #: Shared secret for the API. A secret, so it lives only here — never in a
    #: YAML file under version control, and never in a response body or a
    #: results artifact (master spec §32).
    aris_api_key: str | None = None
    aris_api_host: str | None = None
    aris_api_port: int | None = None
    #: Comma-separated origin list, for a deployment whose front end is served
    #: from somewhere other than the API itself.
    aris_cors_origins: str | None = None
    #: Pins the run the dashboard reads, overriding ``runs.default_run_id``.
    aris_run_id: str | None = None
    #: Groq is used only for grounded paper Q&A. It is never serialized in API
    #: responses or copied into a run artifact.
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    @property
    def cors_origin_list(self) -> list[str] | None:
        """Parse :attr:`aris_cors_origins`, or ``None`` when unset."""
        if self.aris_cors_origins is None:
            return None
        return [origin.strip() for origin in self.aris_cors_origins.split(",") if origin.strip()]


# ===========================================================================
# Composite settings object
# ===========================================================================
class Settings(BaseModel):
    """Fully resolved configuration for a run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    app: AppConfig
    dataset: DatasetConfig
    model: ModelConfig
    api: ApiConfig
    env: EnvSettings
    config_dir: Path
    project_root: Path = PROJECT_ROOT

    # -- convenience accessors used across the pipeline ---------------------
    @property
    def seed(self) -> int:
        """Effective random seed."""
        return self.app.project.seed

    @property
    def paths(self) -> PathsConfig:
        """Configured filesystem layout."""
        return self.app.paths

    @property
    def labels(self) -> LabelsConfig:
        """Label-space configuration."""
        return self.app.labels

    @property
    def taxonomy(self) -> TaxonomyConfig:
        """Active label space, selected by ``labels.taxonomy_level``."""
        return self.dataset.openalex.taxonomy(self.app.labels.taxonomy_level)

    @property
    def class_names(self) -> list[str]:
        """Active class names, with configured exclusions removed."""
        excluded = set(self.app.labels.exclude_classes)
        return [n for n in self.taxonomy.class_names if n not in excluded]

    @property
    def log_level(self) -> str:
        """Effective root log level."""
        return self.app.logging.level

    @property
    def results_dir(self) -> Path:
        """Directory holding per-run output directories.

        ``api.runs.results_dir`` wins when set; otherwise the single definition
        in ``paths.results_dir`` applies, so the two cannot silently disagree.
        """
        override = self.api.runs.results_dir
        if override is not None:
            return resolve_path(override)
        return self.paths.resolved("results_dir")


def _apply_env_overrides(app: AppConfig, env: EnvSettings) -> AppConfig:
    """Overlay environment variables onto the YAML-derived app config.

    ``ARIS_DATA_DIR`` rebases every data subdirectory that sits beneath the
    configured ``data_dir``, so relocating the corpus to another drive does not
    require editing five separate path entries.
    """
    if env.aris_seed is not None:
        app.project.seed = env.aris_seed

    if env.aris_log_level is not None:
        app.logging.level = env.aris_log_level.upper()

    if env.aris_data_dir is not None:
        old_root, new_root = app.paths.data_dir, env.aris_data_dir
        for attr in ("raw_dir", "interim_dir", "processed_dir", "sample_dir", "external_dir"):
            current: Path = getattr(app.paths, attr)
            try:
                relative = current.relative_to(old_root)
            except ValueError:
                continue  # configured outside data_dir; leave untouched
            setattr(app.paths, attr, new_root / relative)
        app.paths.data_dir = new_root

    return app


def _apply_api_env_overrides(api: ApiConfig, env: EnvSettings) -> ApiConfig:
    """Overlay deployment-varying environment values onto the API config.

    Only bind address, allowed origins, and the pinned run id are overridable:
    those genuinely differ between a laptop and a deployment. Everything
    behavioural stays in YAML so it is reviewable in version control.

    The API key is deliberately absent — it is a secret, read straight from
    :class:`EnvSettings` at request time and never copied into a config object
    that gets serialised (master spec §32).
    """
    if env.aris_api_host is not None:
        api.server.host = env.aris_api_host
    if env.aris_api_port is not None:
        api.server.port = env.aris_api_port
    if (origins := env.cors_origin_list) is not None:
        api.cors.allow_origins = origins
    if env.aris_run_id is not None:
        api.runs.default_run_id = env.aris_run_id
    return api


def load_settings(
    config_dir: Path | str = "configs",
    *,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load, overlay, and validate the full configuration.

    Args:
        config_dir: Directory holding ``config.yaml``, ``dataset.yaml``,
            ``model.yaml``, and ``api.yaml``. Relative paths resolve against the
            project root.
        overrides: Optional nested overrides applied to ``config.yaml`` data
            before validation, e.g. ``{"labels": {"mode": "multilabel"}}``.
            Used by CLI flags so a script never mutates config files on disk.

    Returns:
        A validated :class:`Settings` instance.

    Raises:
        FileNotFoundError: If any expected config file is missing.
        ValueError: If a config file is malformed or fails validation.
    """
    directory = resolve_path(config_dir)
    app_data = read_yaml(directory / "config.yaml")
    dataset_data = read_yaml(directory / "dataset.yaml")
    model_data = read_yaml(directory / "model.yaml")
    api_data = read_yaml(directory / "api.yaml")

    if overrides:
        app_data = _deep_merge(app_data, overrides)

    env = EnvSettings()
    app = _apply_env_overrides(AppConfig(**app_data), env)
    api = _apply_api_env_overrides(ApiConfig(**api_data), env)

    return Settings(
        app=app,
        dataset=DatasetConfig(**dataset_data),
        model=ModelConfig(**model_data),
        api=api,
        env=env,
        config_dir=directory,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new mapping.

    Nested mappings merge key-by-key; every other type is replaced outright.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, loaded once and cached.

    Convenience for long-lived processes such as the API in a later milestone.
    Pipeline scripts call :func:`load_settings` directly so that CLI overrides
    are honoured and tests stay isolated from one another.
    """
    return load_settings()
