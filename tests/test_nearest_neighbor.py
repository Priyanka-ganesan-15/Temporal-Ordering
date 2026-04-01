import numpy as np

from chronologic.ordering.nearest_neighbor import (
    adjacency_path_score,
    best_greedy_path,
    greedy_path_from_start,
)


def test_adjacency_path_score_sums_adjacent_similarities() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    assert np.isclose(adjacency_path_score([0, 1, 2], similarity_matrix), 1.7)


def test_greedy_path_from_start_returns_path_and_score() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.8, 0.3],
            [0.2, 0.8, 1.0, 0.7],
            [0.1, 0.3, 0.7, 1.0],
        ],
        dtype=np.float32,
    )

    path, score = greedy_path_from_start(similarity_matrix, 0)
    assert path == [0, 1, 2, 3]
    assert np.isclose(score, 2.4)


def test_best_greedy_path_tries_all_start_nodes() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.1, 0.6, 0.2],
            [0.1, 1.0, 0.3, 0.95],
            [0.6, 0.3, 1.0, 0.9],
            [0.2, 0.95, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    best_path, best_score = best_greedy_path(similarity_matrix)
    assert best_path in ([0, 2, 3, 1], [1, 3, 2, 0])
    assert np.isclose(best_score, 2.45)