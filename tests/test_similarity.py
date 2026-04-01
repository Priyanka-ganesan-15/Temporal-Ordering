import numpy as np
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

similarity_module = importlib.import_module("temporal_ordering.similarity")
cosine_similarity_matrix = similarity_module.cosine_similarity_matrix
greedy_nearest_neighbor_ordering = similarity_module.greedy_nearest_neighbor_ordering
greedy_path_from_start = similarity_module.greedy_path_from_start
path_adjacency_score = similarity_module.path_adjacency_score
random_ordering_baseline = similarity_module.random_ordering_baseline
temporal_structure_score = similarity_module.temporal_structure_score


def test_cosine_similarity_matrix_identity_for_orthogonal_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = cosine_similarity_matrix(embeddings)
    assert sim.shape == (2, 2)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.isclose(sim[0, 1], 0.0)


def test_temporal_structure_score_expected_contrast() -> None:
    sim = np.array(
        [
            [1.0, 0.9, 0.5, 0.3],
            [0.9, 1.0, 0.8, 0.4],
            [0.5, 0.8, 1.0, 0.85],
            [0.3, 0.4, 0.85, 1.0],
        ],
        dtype=np.float32,
    )
    near, far, contrast = temporal_structure_score(sim, near_gap=1, far_gap=3)
    assert near > far
    assert np.isclose(contrast, near - far)


def test_random_ordering_baseline_returns_permutation() -> None:
    permutation = random_ordering_baseline(5)
    assert len(permutation) == 5
    assert sorted(permutation) == [0, 1, 2, 3, 4]


def test_random_ordering_baseline_is_reproducible_with_seed() -> None:
    first = random_ordering_baseline(8, seed=123)
    second = random_ordering_baseline(8, seed=123)
    assert first == second


def test_random_ordering_baseline_rejects_non_positive_frame_count() -> None:
    try:
        random_ordering_baseline(0)
    except ValueError as exc:
        assert str(exc) == "n_frames must be greater than 0"
    else:
        raise AssertionError("Expected ValueError for non-positive frame count")


def test_greedy_path_from_start_avoids_revisiting_frames() -> None:
    sim = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.8, 0.3],
            [0.2, 0.8, 1.0, 0.7],
            [0.1, 0.3, 0.7, 1.0],
        ],
        dtype=np.float32,
    )

    path = greedy_path_from_start(sim, start_frame=0)
    assert path == [0, 1, 2, 3]
    assert len(path) == 4
    assert len(set(path)) == 4


def test_path_adjacency_score_sums_adjacent_pairs() -> None:
    sim = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    score = path_adjacency_score(sim, [0, 1, 2])
    assert np.isclose(score, 1.7)


def test_greedy_nearest_neighbor_ordering_tries_all_start_frames() -> None:
    sim = np.array(
        [
            [1.0, 0.1, 0.6, 0.2],
            [0.1, 1.0, 0.3, 0.95],
            [0.6, 0.3, 1.0, 0.9],
            [0.2, 0.95, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    best_path, best_score = greedy_nearest_neighbor_ordering(sim)
    assert best_path in ([0, 2, 3, 1], [1, 3, 2, 0])
    assert np.isclose(best_score, 2.45)


def test_greedy_nearest_neighbor_ordering_rejects_non_square_input() -> None:
    sim = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.7]], dtype=np.float32)

    try:
        greedy_nearest_neighbor_ordering(sim)
    except ValueError as exc:
        assert str(exc) == "similarity_matrix must have shape (n, n)"
    else:
        raise AssertionError("Expected ValueError for non-square similarity matrix")
