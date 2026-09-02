"""Citation Network Builder.

Builds reference graph networks and co-citation links across paper records in the corpus.
"""

from __future__ import annotations

from typing import Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.schemas.paper import DatasetRecord, PaperDocument

__all__ = ["CitationEdge", "CitationGraph", "CitationNetworkBuilder", "CitationNode"]

_RESPONSE = ConfigDict(extra="forbid", protected_namespaces=())


class CitationNode(BaseModel):
    """A paper node in the citation graph."""

    model_config = _RESPONSE

    id: str
    label: str
    domain: str | None = None


class CitationEdge(BaseModel):
    """A directed reference link between two papers."""

    model_config = _RESPONSE

    source: str
    target: str


class CitationGraph(BaseModel):
    """Full citation network graph payload."""

    model_config = _RESPONSE

    nodes: list[CitationNode]
    edges: list[CitationEdge]


class CitationNetworkBuilder:
    """Builder for paper citation graphs."""

    def build_graph(self, records: Sequence[DatasetRecord | PaperDocument]) -> CitationGraph:
        """Build citation graph from paper records."""
        nodes: list[CitationNode] = []
        edges: list[CitationEdge] = []
        known_ids = {r.paper_id for r in records}

        for rec in records:
            domain = getattr(rec, "label", None) or (rec.primary_topic.subfield if hasattr(rec, "primary_topic") and rec.primary_topic else None)
            nodes.append(
                CitationNode(
                    id=rec.paper_id,
                    label=getattr(rec, "title", rec.paper_id),
                    domain=domain,
                )
            )
            refs = getattr(rec, "references", []) or getattr(rec, "meta", {}).get("references", [])
            for ref_id in refs:
                if ref_id in known_ids:
                    edges.append(CitationEdge(source=rec.paper_id, target=ref_id))

        return CitationGraph(nodes=nodes, edges=edges)

