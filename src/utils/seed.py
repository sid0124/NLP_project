"""Deterministic seeding for reproducible experiments (master spec §51)."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

__all__ = ["set_seed"]

logger = get_logger(__name__)


def set_seed(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed every random source the pipeline draws on.

    Covers Python's ``random``, NumPy's legacy global generator (which
    scikit-learn consults when ``random_state`` is unset), and ``PYTHONHASHSEED``.
    PyTorch is seeded too when installed, so that Milestone 2 inherits
    reproducibility without a second seeding path.

    Note:
        ``PYTHONHASHSEED`` only affects interpreters started *after* it is set;
        it is exported here for subprocesses. Reproducibility within this
        process does not depend on it, since no code relies on ``hash()``
        ordering.

    Args:
        seed: Non-negative seed value.
        deterministic_torch: When PyTorch is present, also request deterministic
            cuDNN kernels. Slower, but removes GPU run-to-run variance.

    Returns:
        The seed that was applied, for logging into run metadata.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional: keeps Milestone 2 (transformers) reproducible via the same call.
    try:
        import torch
    except ImportError:
        pass
    else:  # pragma: no cover - torch is not a Milestone 1 dependency
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.debug("Random seed set to %d", seed)
    return seed
