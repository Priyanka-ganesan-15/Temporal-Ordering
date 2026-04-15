"""Insertion-sort temporal ordering for image sequences."""

from __future__ import annotations

import numpy as np

from chronologic.ordering.nearest_neighbor import adjacency_path_score, validate_similarity_matrix


def insertion_sort_ordering(
    similarity_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Order frames using a greedy insertion-sort strategy.

    Builds the sequence one frame at a time: starts with frame 0, then for
    each remaining frame tries every possible insertion position and keeps the
    one that maximises the total adjacency path score.

    Unlike greedy nearest-neighbor, which extends the path only at the right
    end, insertion sort can slot new frames anywhere — producing paths that
    avoid the 'dangling tail' problem where a poorly-chosen start node forces
    all subsequent frames into suboptimal positions.

    Returns
    -------
    ordering : list[int]
        Frame indices in the predicted temporal order.
    path_score : float
        Total adjacent-similarity score of the returned ordering.
    """
    validate_similarity_matrix(similarity_matrix)
    n = similarity_matrix.shape[0]

    if n == 1:
        return [0], 0.0

    path = [0]
    unplaced = list(range(1, n))

    for frame in unplaced:
        # Compute insertion delta: gain from new adjacencies minus
        # the adjacency that the insertion breaks.  This is equivalent to
        # maximising total path score because all other pairs are unchanged.
        best_delta = float("-inf")
        best_pos = 0

        for pos in range(len(path) + 1):
            before = float(similarity_matrix[path[pos - 1], frame]) if pos > 0 else 0.0
            after = float(similarity_matrix[frame, path[pos]]) if pos < len(path) else 0.0
            removed = (
                float(similarity_matrix[path[pos - 1], path[pos]])
                if 0 < pos < len(path)
                else 0.0
            )
            delta = before + after - removed
            if delta > best_delta:
                best_delta = delta
                best_pos = pos

        path = path[:best_pos] + [frame] + path[best_pos:]

    return path, adjacency_path_score(path, similarity_matrix)


def best_insertion_sort_ordering(
    similarity_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Return the best insertion-sort path across all possible starting frames.

    Runs :func:`insertion_sort_ordering` once for each possible starting frame
    (by permuting the similarity matrix columns/rows so the start is always
    at index 0), then returns the path with the highest adjacency score.

    This is O(n²·n!) in the worst case but in practice each insertion pass is
    O(n²), so the total is O(n³) — tractable for n ≤ 20.

    Returns
    -------
    ordering : list[int]
        Frame indices in the predicted temporal order.
    path_score : float
        Total adjacent-similarity score of the returned ordering.
    """
    validate_similarity_matrix(similarity_matrix)
    n = similarity_matrix.shape[0]

    best_path: list[int] | None = None
    best_score = float("-inf")

    for start in range(n):
        # Re-index so that `start` is visited first by insertion_sort_ordering
        idx = [start] + [i for i in range(n) if i != start]
        sub_matrix = similarity_matrix[np.ix_(idx, idx)]

        local_path, local_score = insertion_sort_ordering(sub_matrix)

        # Map back to original frame indices
        original_path = [idx[i] for i in local_path]
        score = adjacency_path_score(original_path, similarity_matrix)

        if score > best_score:
            best_score = score
            best_path = original_path

    assert best_path is not None
    return best_path, best_score


__all__ = ["insertion_sort_ordering", "best_insertion_sort_ordering"]
