"""Make ``python scripts/<name>.py`` work without an editable install.

Every script imports this first. Two things happen here, both of which must
precede any other project import:

1. The project root joins ``sys.path``, so ``import src...`` resolves whether
   the package was installed or the repository was merely cloned.
2. ``stdout``/``stderr`` are switched to UTF-8. Academic metadata is full of
   non-ASCII author names, and a Windows console defaults to cp1252, where a
   single ``print`` of such a name raises ``UnicodeEncodeError``.

Because this runs import-time side effects by design, scripts import it as
``import _bootstrap  # noqa: F401`` before importing ``src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import force_utf8_streams  # noqa: E402

force_utf8_streams()

__all__ = ["PROJECT_ROOT"]
