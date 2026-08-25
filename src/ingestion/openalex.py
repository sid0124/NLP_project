"""OpenAlex ingestion adapter.

OpenAlex is used as the primary metadata source because it requires no API key,
releases its data under CC0 (so a snapshot can be committed for offline
reproducibility), and exposes a full ``topic -> subfield -> field -> domain``
hierarchy that makes the label space configurable rather than hard-coded.

The API contract below was verified against the live service on 2026-08-23:

* cursor pagination — ``cursor=*``, then follow ``meta.next_cursor``
* ``select`` sparse fieldsets and ``per-page`` (max 200) are honoured
* ``filter`` accepts comma-separated AND clauses
* filter *values* use short ids (``fields/17``) while responses echo back full
  URLs (``https://openalex.org/fields/17``), so ids are normalised on parse
* abstracts arrive as ``abstract_inverted_index``: ``{term: [positions]}``
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.settings import Settings
from src.ingestion.base import (
    ClassFetchStat,
    FetchManifest,
    IngestionError,
    IngestionSource,
    register_source,
)
from src.schemas.paper import Author, PaperDocument, TopicAssignment
from src.utils.io import ensure_dir, sha256_file, write_json, write_jsonl
from src.utils.logging import get_logger

__all__ = ["OpenAlexSource", "reconstruct_abstract", "slugify"]

logger = get_logger(__name__)

_OPENALEX_URL_PREFIX = "https://openalex.org/"
_DOI_URL_PREFIX = "https://doi.org/"


def slugify(value: str) -> str:
    """Convert a class name into a filesystem-safe lowercase slug.

    ``"Computer Vision and Pattern Recognition"`` becomes
    ``"computer_vision_and_pattern_recognition"``.
    """
    cleaned = [char.lower() if char.isalnum() else "_" for char in value]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unnamed"


def _short_id(value: Any) -> str | None:
    """Strip the OpenAlex URL prefix from an identifier.

    ``https://openalex.org/W123`` becomes ``W123``; non-strings yield ``None``.
    """
    if not isinstance(value, str) or not value:
        return None
    return value.removeprefix(_OPENALEX_URL_PREFIX)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild abstract text from an OpenAlex inverted index.

    OpenAlex stores abstracts as ``{term: [token positions]}`` rather than as
    plain text. Reconstruction places each occurrence of every term at its
    recorded position and joins the result in positional order.

    Duplicate positions are tolerated (last term wins) and non-integer or
    negative positions are ignored, so a malformed payload degrades to a partial
    abstract instead of raising mid-fetch.

    Args:
        inverted_index: The mapping as returned by OpenAlex, possibly ``None``.

    Returns:
        The reconstructed abstract, or ``None`` if nothing usable was present.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return None

    positioned: dict[int, str] = {}
    for term, positions in inverted_index.items():
        if not isinstance(term, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int):
                continue
            if position < 0:
                continue
            positioned[position] = term

    if not positioned:
        return None
    text = " ".join(positioned[index] for index in sorted(positioned))
    return text or None


def _parse_topic(raw: Any) -> TopicAssignment | None:
    """Convert one OpenAlex topic object into a :class:`TopicAssignment`."""
    if not isinstance(raw, dict):
        return None
    name = raw.get("display_name")
    if not isinstance(name, str) or not name.strip():
        return None

    raw_score = raw.get("score")
    numeric_score = isinstance(raw_score, int | float) and not isinstance(raw_score, bool)
    score = float(raw_score) if numeric_score else 0.0

    def nested(key: str) -> str | None:
        node = raw.get(key)
        return node.get("display_name") if isinstance(node, dict) else None

    return TopicAssignment(
        display_name=name.strip(),
        # Clamped because the schema constrains scores to [0, 1] and a single
        # out-of-range value should not invalidate an otherwise good record.
        score=min(max(score, 0.0), 1.0),
        topic_id=_short_id(raw.get("id")),
        subfield=nested("subfield"),
        field=nested("field"),
        domain=nested("domain"),
    )


def _parse_authors(authorships: Any) -> list[Author]:
    """Extract authors from an OpenAlex ``authorships`` array.

    Falls back to ``raw_author_name`` when the disambiguated author record has
    no display name, and skips entries with neither.
    """
    if not isinstance(authorships, list):
        return []

    authors: list[Author] = []
    for entry in authorships:
        if not isinstance(entry, dict):
            continue
        author_node = entry.get("author") if isinstance(entry.get("author"), dict) else {}
        name = author_node.get("display_name") or entry.get("raw_author_name")
        if not isinstance(name, str) or not name.strip():
            continue

        institutions = entry.get("institutions")
        affiliations = (
            [
                inst["display_name"]
                for inst in institutions
                if isinstance(inst, dict) and isinstance(inst.get("display_name"), str)
            ]
            if isinstance(institutions, list)
            else []
        )

        authors.append(
            Author(
                name=name.strip(),
                author_id=_short_id(author_node.get("id")),
                affiliations=affiliations,
                position=entry.get("author_position"),
                orcid=author_node.get("orcid") or entry.get("raw_orcid"),
            )
        )
    return authors


@register_source
class OpenAlexSource(IngestionSource):
    """Fetches and parses labelled paper metadata from the OpenAlex Works API."""

    name = "openalex"

    def __init__(self, settings: Settings) -> None:
        """Initialise the adapter.

        Args:
            settings: Resolved configuration supplying query parameters, the
                active label space, and the polite-pool contact address.
        """
        self.settings = settings
        self.config = settings.dataset.openalex
        self.taxonomy = settings.taxonomy
        self._session: requests.Session | None = None

    # -- HTTP ---------------------------------------------------------------
    @property
    def session(self) -> requests.Session:
        """Lazily built session with retry/backoff on transient failures."""
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def _build_session(self) -> requests.Session:
        """Create a session that retries idempotent GETs with exponential backoff."""
        request_config = self.config.request
        session = requests.Session()
        retry = Retry(
            total=request_config.max_retries,
            connect=request_config.max_retries,
            read=request_config.max_retries,
            status=request_config.max_retries,
            backoff_factor=request_config.backoff_factor,
            status_forcelist=tuple(request_config.retry_on_status),
            allowed_methods=frozenset({"GET"}),
            # Honour the server's own throttling guidance ahead of our backoff.
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=4)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        mailto = self.settings.env.openalex_mailto
        user_agent = f"academic-research-intelligence/{self.settings.app.project.version}"
        if mailto:
            user_agent = f"{user_agent} (mailto:{mailto})"
        session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        return session

    def _base_params(self) -> dict[str, Any]:
        """Query parameters common to every request."""
        params: dict[str, Any] = {
            "select": ",".join(self.config.select),
            "per-page": self.config.per_page,
        }
        mailto = self.settings.env.openalex_mailto
        if self.config.polite_pool and mailto:
            params["mailto"] = mailto
        return params

    def _build_filter(self, target_id: str) -> str:
        """Compose the comma-separated OpenAlex filter string for one class."""
        filters = self.config.filters
        clauses = [
            f"{self.taxonomy.filter_key}:{target_id}",
            f"from_publication_date:{filters.from_publication_date}",
            f"to_publication_date:{filters.to_publication_date}",
            f"language:{filters.language}",
            f"type:{filters.type}",
            f"has_abstract:{str(filters.has_abstract).lower()}",
        ]
        return ",".join(clauses)

    def _get_page(self, filter_string: str, cursor: str) -> dict[str, Any]:
        """Fetch one page of works.

        Raises:
            IngestionError: On transport failure, a non-200 status that survived
                retries, or a body that is not a JSON object.
        """
        params = self._base_params() | {"filter": filter_string, "cursor": cursor}
        try:
            response = self.session.get(
                self.config.base_url,
                params=params,
                timeout=self.config.request.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise IngestionError(
                f"OpenAlex request failed for filter '{filter_string}': {exc}"
            ) from exc

        if response.status_code != 200:
            raise IngestionError(
                f"OpenAlex returned HTTP {response.status_code} for filter "
                f"'{filter_string}': {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise IngestionError(f"OpenAlex returned non-JSON payload: {exc}") from exc
        if not isinstance(payload, dict):
            raise IngestionError(
                f"Expected a JSON object from OpenAlex, got {type(payload).__name__}"
            )
        return payload

    def _fetch_class(
        self, target_id: str, class_name: str, limit: int
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Page through all works for one class, up to ``limit`` records.

        Returns:
            The retrieved raw records, and the total the API reported as
            available (``None`` if the API omitted it).
        """
        filter_string = self._build_filter(target_id)
        records: list[dict[str, Any]] = []
        cursor: str | None = "*"
        available: int | None = None
        pages = 0

        while cursor and len(records) < limit:
            payload = self._get_page(filter_string, cursor)
            meta = payload.get("meta") or {}
            if available is None:
                count = meta.get("count")
                available = count if isinstance(count, int) else None

            results = payload.get("results")
            if not isinstance(results, list) or not results:
                break

            records.extend(results[: limit - len(records)])
            pages += 1
            cursor = meta.get("next_cursor")
            logger.debug(
                "%s: page %d -> %d/%d records (available=%s)",
                class_name, pages, len(records), limit, available,
            )
            if cursor and len(records) < limit:
                time.sleep(self.config.request.sleep_between_requests)

        return records, available

    def fetch(
        self, destination: Path, *, run_id: str, limit_per_class: int | None = None
    ) -> FetchManifest:
        """Fetch one JSONL shard per class, plus a manifest, into ``destination``.

        Fetching per class produces a stratified corpus by construction, which
        keeps the label distribution controlled rather than dominated by the
        largest class.

        Args:
            destination: Directory for the shards and ``manifest.json``.
            run_id: Correlation id recorded in the manifest and logs.
            limit_per_class: Overrides ``max_records_per_class``; useful for
                building small snapshots and smoke tests.

        Returns:
            The fetch manifest, also written to ``destination/manifest.json``.

        Raises:
            IngestionError: If no records could be retrieved for any class.
        """
        target_dir = ensure_dir(destination)
        limit = limit_per_class or self.config.max_records_per_class
        excluded = set(self.settings.app.labels.exclude_classes)
        targets = [t for t in self.taxonomy.targets if t.name not in excluded]

        notes: list[str] = []
        if not self.settings.env.openalex_mailto:
            message = (
                "OPENALEX_MAILTO is not set; requests use the common pool and may be "
                "throttled. Set it in .env to join the faster polite pool."
            )
            logger.warning(message)
            notes.append(message)
        if excluded:
            notes.append(f"Excluded classes (labels.exclude_classes): {sorted(excluded)}")

        stats: list[ClassFetchStat] = []
        total = 0
        for index, target in enumerate(targets, start=1):
            logger.info(
                "[%d/%d] Fetching up to %d records for '%s' (%s)",
                index, len(targets), limit, target.name, target.id,
            )
            records, available = self._fetch_class(target.id, target.name, limit)
            shard = target_dir / f"{slugify(target.name)}.jsonl"
            written = write_jsonl(shard, records)
            total += written

            if written == 0:
                notes.append(f"No records retrieved for class '{target.name}' ({target.id}).")
                logger.warning("No records retrieved for '%s'", target.name)

            stats.append(
                ClassFetchStat(
                    class_name=target.name,
                    target_id=target.id,
                    requested=limit,
                    retrieved=written,
                    available=available,
                    file=shard.name,
                    sha256=sha256_file(shard),
                )
            )

        if total == 0:
            raise IngestionError(
                "OpenAlex fetch produced zero records across all classes. "
                "Check network access and the filters in configs/dataset.yaml."
            )

        manifest = FetchManifest(
            source=self.name,
            run_id=run_id,
            fetched_at=datetime.now(UTC),
            taxonomy_level=self.settings.app.labels.taxonomy_level,
            query={
                "base_url": self.config.base_url,
                "filter_key": self.taxonomy.filter_key,
                "filters": self.config.filters.model_dump(),
                "select": self.config.select,
                "per_page": self.config.per_page,
                "limit_per_class": limit,
            },
            total_records=total,
            classes=stats,
            config_snapshot={
                "taxonomy_level": self.settings.app.labels.taxonomy_level,
                "label_mode": self.settings.app.labels.mode,
                "n_target_classes": len(targets),
                "project_version": self.settings.app.project.version,
            },
            notes=notes,
        )
        write_json(target_dir / "manifest.json", manifest.model_dump(mode="json"))
        logger.info("Fetched %d records across %d classes into %s", total, len(targets), target_dir)
        return manifest

    # -- parsing (pure, offline) -------------------------------------------
    @staticmethod
    def parse_record(raw: dict[str, Any]) -> PaperDocument | None:
        """Convert one OpenAlex work into a :class:`PaperDocument`.

        Tolerant by design: only a missing identifier makes a record
        unrepresentable. Everything else (absent abstract, no topics, blank
        title) parses through and is judged later by the validation stage, which
        keeps quality rules in exactly one place.

        Args:
            raw: A single OpenAlex work object.

        Returns:
            The parsed document, or ``None`` if it has no usable identifier.
        """
        if not isinstance(raw, dict):
            return None
        paper_id = _short_id(raw.get("id"))
        if not paper_id:
            return None

        doi = raw.get("doi")
        doi_clean = doi.removeprefix(_DOI_URL_PREFIX) if isinstance(doi, str) else None

        primary_location = raw.get("primary_location")
        source_node = primary_location.get("source") if isinstance(primary_location, dict) else None
        venue = source_node.get("display_name") if isinstance(source_node, dict) else None

        keywords_raw = raw.get("keywords")
        keywords = (
            [
                kw["display_name"]
                for kw in keywords_raw
                if isinstance(kw, dict) and isinstance(kw.get("display_name"), str)
            ]
            if isinstance(keywords_raw, list)
            else []
        )

        topics_raw = raw.get("topics")
        topics = (
            [topic for topic in (_parse_topic(t) for t in topics_raw) if topic is not None]
            if isinstance(topics_raw, list)
            else []
        )

        references_raw = raw.get("referenced_works")
        references = (
            [ref for ref in (_short_id(r) for r in references_raw) if ref]
            if isinstance(references_raw, list)
            else []
        )

        year = raw.get("publication_year")
        return PaperDocument(
            paper_id=paper_id,
            source=OpenAlexSource.name,
            doi=doi_clean,
            title=raw.get("title") or "",
            abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
            authors=_parse_authors(raw.get("authorships")),
            keywords=keywords,
            publication_date=raw.get("publication_date") or None,
            publication_year=year if isinstance(year, int) else None,
            venue=venue,
            language=raw.get("language"),
            document_type=raw.get("type"),
            references=references,
            primary_topic=_parse_topic(raw.get("primary_topic")),
            topics=topics,
        )

    def parse_records(self, raws: Iterator[dict[str, Any]]) -> Iterator[PaperDocument]:
        """Parse a stream of works, logging how many payloads were unusable."""
        skipped = 0
        for raw in raws:
            document = self.parse_record(raw)
            if document is None:
                skipped += 1
                continue
            yield document
        if skipped:
            logger.warning("Skipped %d OpenAlex payload(s) lacking a usable identifier", skipped)
