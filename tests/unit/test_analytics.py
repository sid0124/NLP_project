"""Unit tests for Research Intelligence Analytics engines."""

from src.analytics.citations import CitationNetworkBuilder
from src.analytics.gaps import ResearchGapDetector
from src.analytics.methodology import MethodologyExtractor
from src.schemas.paper import DatasetRecord, PaperDocument


def test_methodology_extractor():
    text = "We evaluate SciBERT on ImageNet dataset using F1 score and Adam optimizer."
    extractor = MethodologyExtractor()
    result = extractor.extract(text)

    assert "ImageNet" in result.datasets
    assert "F1" in result.metrics
    assert "SciBERT" in result.architectures


def test_research_gap_detector():
    text = "Our model suffers from memory bottleneck and out-of-domain scalability issues."
    detector = ResearchGapDetector()
    gaps = detector.detect(text)

    assert len(gaps) > 0
    categories = [g.category for g in gaps]
    assert "scalability" in categories or "domain_adaptation" in categories


def test_citation_network_builder():
    doc1 = PaperDocument(paper_id="p1", source="test", title="Paper 1", references=["p2"])
    doc2 = PaperDocument(paper_id="p2", source="test", title="Paper 2")

    builder = CitationNetworkBuilder()
    graph = builder.build_graph([doc1, doc2])

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "p1"
    assert graph.edges[0].target == "p2"

