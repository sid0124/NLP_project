"""Unit tests for VectorStore."""

from src.api.vector_store import VectorStore
from src.schemas.paper import PaperDocument, PaperSection, Paragraph


def test_vector_store_indexing_and_search(tmp_path):
    doc1 = PaperDocument(
        paper_id="doc1",
        source="test",
        title="Deep Convolutional Networks for Computer Vision",
        abstract="ResNet and CNN models for image classification.",
    )
    doc2 = PaperDocument(
        paper_id="doc2",
        source="test",
        title="Attention Mechanism in Natural Language Processing",
        abstract="Transformers and BERT models for text processing.",
    )

    store = VectorStore(store_dir=tmp_path)
    store.add_documents([doc1, doc2])

    results = store.search("computer vision convolutional image", top_k=2)
    assert len(results) > 0
    assert results[0][0].paper_id == "doc1"

