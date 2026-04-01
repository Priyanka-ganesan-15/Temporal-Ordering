"""Greedy nearest-neighbor ordering utilities for ChronoLogic."""

from __future__ import annotations

import numpy as np


def adjacency_path_score(path: list[int], similarity_matrix: np.ndarray) -> float:
    """Return the total adjacent similarity score for a full path."""
    validate_similarity_matrix(similarity_matrix)
    _validate_candidate_path(path, similarity_matrix.shape[0])

    if len(path) < 2:
        return 0.0

    return float(
        sum(similarity_matrix[current, nxt] for current, nxt in zip(path, path[1:]))
    )


def validate_similarity_matrix(similarity_matrix: np.ndarray) -> None:
    """Validate that the similarity matrix is square and non-empty."""
    if not isinstance(similarity_matrix, np.ndarray):
        raise TypeError("similarity_matrix must be a numpy.ndarray")
    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be 2-dimensional")

    n_rows, n_cols = similarity_matrix.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("similarity_matrix must be non-empty")
    if n_rows != n_cols:
        raise ValueError("similarity_matrix must have shape (n, n)")


def greedy_path_from_start(
    similarity_matrix: np.ndarray,
    start_idx: int,
) -> tuple[list[int], float]:
    """Build a greedy path from one start node and return the path plus score."""
    validate_similarity_matrix(similarity_matrix)

    n_frames = similarity_matrix.shape[0]
    if not 0 <= start_idx < n_frames:
        raise ValueError("start_idx is out of range")

    path = [start_idx]
    unused = set(range(n_frames))
    unused.remove(start_idx)

    while unused:
        current = path[-1]
        next_idx = max(unused, key=lambda frame: similarity_matrix[current, frame])
        path.append(next_idx)
        unused.remove(next_idx)

    return path, adjacency_path_score(path, similarity_matrix)


def best_greedy_path(similarity_matrix: np.ndarray) -> tuple[list[int], float]:
    """Return the best greedy path across all starting nodes."""
    validate_similarity_matrix(similarity_matrix)

    best_path: list[int] | None = None
    best_score = float("-inf")
    for start_idx in range(similarity_matrix.shape[0]):
        path, score = greedy_path_from_start(similarity_matrix, start_idx)
        if score > best_score:
            best_path = path
            best_score = score

    if best_path is None:
        raise RuntimeError("Failed to construct a greedy ordering path")
    return best_path, best_score


def score_candidate_permutation(
    similarity_matrix: np.ndarray,
    candidate: list[int],
) -> float:
    """Score a full candidate permutation by adjacency continuity."""
    return adjacency_path_score(candidate, similarity_matrix)


def _validate_candidate_path(path: list[int], n_frames: int) -> None:
    if len(path) != n_frames:
        raise ValueError("candidate path must include each frame exactly once")
    if sorted(path) != list(range(n_frames)):
        raise ValueError("candidate path must be a permutation of frame indices")


__all__ = [
    "adjacency_path_score",
    "best_greedy_path",
    "greedy_path_from_start",
    "score_candidate_permutation",
    "validate_similarity_matrix",
]