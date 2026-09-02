"""Extractive paper Q&A retrieval engine.

Segments paper text into candidate section/paragraph passages, scores passage relevance against
the input question using TF-IDF and keyword matching, and produces grounded answers with section provenance.
Refuses gracefully when no relevant passage is found (master spec §20).
"""

from __future__ import annotations

import re
from typing import Sequence

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.api.schemas import AskResponse, PassageEvidence
from src.preprocessing.sections import parse_text_into_sections
from src.schemas.paper import PaperDocument, PaperSection

__all__ = ["PaperQAEngine"]


class PaperQAEngine:
    """Passage-level retrieval and Q&A engine for an academic paper."""

    def __init__(
        self,
        paper_id: str,
        title: str,
        text: str,
        sections: Sequence[PaperSection] | None = None,
        groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-120b",
    ) -> None:
        self.paper_id = paper_id
        self.title = title
        self.text = text
        self.sections = parse_text_into_sections(text, title=title, existing_sections=sections)
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

    @classmethod
    def from_document(cls, paper: PaperDocument) -> PaperQAEngine:
        """Instantiate engine directly from a PaperDocument."""
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            text=paper.full_text or paper.text_for(("title", "abstract")),
            sections=paper.sections,
        )

    def answer_question(self, question: str, min_confidence: float = 0.08) -> AskResponse:
        """Answer a question using passage retrieval.

        Args:
            question: The user's natural language question.
            min_confidence: Minimum score threshold required to return an answer.

        Returns:
            An :class:`src.api.schemas.AskResponse` payload.
        """
        passages: list[tuple[str, str]] = []  # (section_name, passage_text)
        for sec in self.sections:
            sec_name = sec.section_name or sec.canonical_name or "Body"
            for para in sec.paragraphs:
                if para.text.strip():
                    passages.append((sec_name, para.text.strip()))

        if not passages:
            passages = [("Paper Content", self.text[:4000] if self.text else self.title)]

        # Vectorize question and candidate passages
        corpus = [p[1] for p in passages]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([question])
            scores = (tfidf_matrix * query_vec.T).toarray().ravel()
        except Exception:
            scores = np.zeros(len(passages))

        best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        best_score = float(scores[best_idx]) if len(scores) > 0 else 0.0

        # Summary questions often use words that do not occur in the paper
        # ("what is it about?", "main topic", "overview"). Let Groq summarize
        # the strongest retrieved passages for those intents; unrelated queries
        # still receive an explicit grounded refusal.
        summary_question = bool(
            re.search(
                r"\b(about|topic|overview|summary|summarize|purpose|objective|"
                r"research question|main idea|contribution)\b",
                question,
                re.IGNORECASE,
            )
        )
        if best_score < min_confidence and not summary_question:
            return AskResponse(
                paper_id=self.paper_id,
                question=question,
                answer="Information not found in the provided paper.",
                confidence=0.0,
            )

        source_section, best_passage = passages[best_idx]

        # Give Groq a few relevant passages, not the whole document. This keeps
        # the generated answer grounded and leaves the evidence visible in the UI.
        ranked_indexes = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        context = [
            (passages[index][0], passages[index][1], float(scores[index]))
            for index in ranked_indexes[:3]
            if float(scores[index]) >= min_confidence or summary_question
        ]
        if not context:
            context = [(source_section, best_passage, best_score)]

        # Start with a deterministic local answer. Groq improves the wording
        # when configured, but a provider outage must not make the panel blank.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_passage) if s.strip()]
        answer_text = sentences[0] if sentences else best_passage

        if self.groq_api_key:
            try:
                context_text = "\n\n".join(
                    f"[{section}]\n{passage}" for section, passage, _ in context
                )
                prompt = (
                    "You are a grounded academic research assistant. Answer the question "
                    "using only the paper passages below. If the passages do not support "
                    "an answer, reply exactly: Information not found in the provided paper. "
                    "Do not invent facts, citations, datasets, or results. Keep the answer "
                    "concise and mention uncertainty when appropriate.\n\n"
                    f"Paper title: {self.title}\n"
                    f"Question: {question}\n\n"
                    f"Retrieved paper passages:\n{context_text}"
                )
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Answer only from supplied research-paper context.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        # GPT-OSS uses some completion tokens for reasoning;
                        # 600 leaves room for visible answer content too.
                        "max_tokens": 600,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
                if isinstance(content, str) and content.strip():
                    answer_text = content.strip()
            except (requests.RequestException, ValueError, IndexError, KeyError):
                # Do not expose provider or credential details in the response.
                pass

        evidence = [
            PassageEvidence(
                source_section=source_section,
                passage=best_passage,
                confidence=round(best_score, 4),
            )
        ]

        return AskResponse(
            paper_id=self.paper_id,
            question=question,
            answer=answer_text,
            source=f"Section: {source_section}",
            evidence=evidence,
            confidence=round(best_score, 4),
        )
