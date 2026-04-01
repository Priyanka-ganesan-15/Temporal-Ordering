"""Legacy compatibility wrapper for the random ordering baseline."""

from __future__ import annotations

import random

from chronologic.ordering.random_baseline import random_permutation


def random_ordering_baseline(
    n_frames: int,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    """Return a random permutation of frame indices for a sequence."""
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both")
    if rng is not None:
        if n_frames <= 0:
            raise ValueError("n_frames must be greater than 0")
        permutation = list(range(n_frames))
        rng.shuffle(permutation)
        return permutation
    try:
        return random_permutation(n_frames, seed=seed)
    except ValueError as exc:
        raise ValueError("n_frames must be greater than 0") from exc