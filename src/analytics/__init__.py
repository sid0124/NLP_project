"""Research Intelligence Analytics package.

Provides methodology extraction, open research gap detection, and citation network analysis.
"""

from src.analytics.citations import CitationNetworkBuilder
from src.analytics.gaps import ResearchGapDetector
from src.analytics.methodology import MethodologyExtractor

__all__ = ["CitationNetworkBuilder", "MethodologyExtractor", "ResearchGapDetector"]

