"""Research Gap Detector.

Identifies open research challenges, limitations, and future work from paper text.
"""

from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DetectedGap", "ResearchGapDetector"]

_RESPONSE = ConfigDict(extra="forbid", protected_namespaces=())


class DetectedGap(BaseModel):
    """An open research challenge or limitation extracted from literature."""

    model_config = _RESPONSE

    category: str
    statement: str
    confidence: float


class ResearchGapDetector:
    """Detector for research limitations and open challenges."""

    _GAP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("scalability", re.compile(r"\b(scalable|scalability|computational cost|memory bottleneck)\b", re.IGNORECASE)),
        ("domain_adaptation", re.compile(r"\b(generaliz|out-of-domain|domain shift|unseen data)\b", re.IGNORECASE)),
        ("interpretability", re.compile(r"\b(black-box|explainability|interpretability|opaque)\b", re.IGNORECASE)),
        ("data_scarcity", re.compile(r"\b(data scarcity|limited annotations|small sample|labeling cost)\b", re.IGNORECASE)),
    )

    def detect(self, text: str) -> list[DetectedGap]:
        """Detect research gaps and limitations in text."""
        gaps: list[DetectedGap] = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

        for sent in sentences:
            for category, pattern in self._GAP_PATTERNS:
                if pattern.search(sent):
                    gaps.append(
                        DetectedGap(
                            category=category,
                            statement=sent[:250],
                            confidence=0.85,
                        )
                    )
                    break
        return gaps[:10]

