"""Legacy compatibility wrappers for greedy nearest-neighbor ordering."""

from __future__ import annotations

import numpy as np

from chronologic.ordering.nearest_neighbor import (
    adjacency_path_score,
    best_greedy_path,
    greedy_path_from_start as _greedy_path_from_start,
    score_candidate_permutation,
    validate_similarity_matrix,
)


def path_adjacency_score(similarity_matrix: np.ndarray, path: list[int]) -> float:
    """Return the sum of adjacent similarities along a path."""
    return adjacency_path_score(path, similarity_matrix)


def greedy_path_from_start(similarity_matrix: np.ndarray, start_frame: int) -> list[int]:
    """Build a greedy nearest-neighbor path from one starting frame."""
    path, _ = _greedy_path_from_start(similarity_matrix, start_frame)
    return path


def greedy_nearest_neighbor_ordering(
    similarity_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Try every start frame and return the highest-scoring greedy path."""
    return best_greedy_path(similarity_matrix)