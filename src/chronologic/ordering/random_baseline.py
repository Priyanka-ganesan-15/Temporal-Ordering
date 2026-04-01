"""Random baseline for ChronoLogic ordering experiments."""

from __future__ import annotations

import random


def random_permutation(n: int, seed: int | None = None) -> list[int]:
    """Return a reproducible random permutation of indices [0, ..., n - 1]."""
    if n <= 0:
        raise ValueError("n must be greater than 0")

    permutation = list(range(n))
    random.Random(seed).shuffle(permutation)
    return permutation


__all__ = ["random_permutation"]