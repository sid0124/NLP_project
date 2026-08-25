"""Route modules, one per area of the dashboard.

Split by what the client asks for rather than by HTTP verb: ``system`` answers
"what is this server and what can it do", ``dashboard`` answers the aggregate
panels, and ``papers`` answers everything about an individual paper.
"""

from __future__ import annotations

__all__ = ["dashboard", "papers", "system"]
