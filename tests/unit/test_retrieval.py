"""Unit tests for PaperQAEngine retrieval and Q&A."""

from src.api.retrieval import PaperQAEngine


def test_paper_qa_engine_matching():
    title = "Hierarchical Attention Networks for Document Classification"
    text = """1. Introduction
Document classification is an important task in natural language processing.

2. Methodology
We introduce a novel hierarchical attention network that operates at word, sentence, and section levels.
The model weights each section according to its predictive contribution.

3. Experiments
We evaluate our method on academic papers dataset using macro F1 score."""

    engine = PaperQAEngine(paper_id="paper-101", title=title, text=text)
    response = engine.answer_question("What methodology does this paper introduce?")

    assert response.paper_id == "paper-101"
    assert response.confidence > 0.0
    assert "hierarchical attention" in response.answer.lower()
    assert response.source is not None
    assert len(response.evidence) > 0


def test_paper_qa_engine_out_of_scope_refusal():
    title = "Graph Neural Networks in Bioinformatics"
    text = "Abstract: We apply graph neural networks to predict protein structures."

    engine = PaperQAEngine(paper_id="paper-202", title=title, text=text)
    response = engine.answer_question("What is the recipe for baking a chocolate cake?")

    assert response.answer == "Information not found in the provided paper."
    assert response.confidence == 0.0
    assert response.source is None

