import numpy as np

from chronologic.ordering.continuity import (
    best_continuity_only_path,
    best_continuity_plus_text_direction_path,
    best_continuity_path,
    best_oriented_continuity_plus_text_direction_path,
    build_directional_evidence,
    continuity_only,
    continuity_plus_text_direction,
    disambiguate_reversal,
    permutation_score_components,
    score_permutation_with_continuity,
)


def test_score_permutation_with_continuity_rewards_smooth_steps() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.92, 0.50, 0.10],
            [0.92, 1.0, 0.88, 0.45],
            [0.50, 0.88, 1.0, 0.86],
            [0.10, 0.45, 0.86, 1.0],
        ],
        dtype=np.float32,
    )

    smooth = score_permutation_with_continuity(similarity_matrix, [0, 1, 2, 3])
    jumpy = score_permutation_with_continuity(similarity_matrix, [0, 2, 1, 3])

    assert smooth > jumpy


def test_best_continuity_path_returns_permutation_and_score() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.95, 0.35, 0.10],
            [0.95, 1.0, 0.90, 0.30],
            [0.35, 0.90, 1.0, 0.88],
            [0.10, 0.30, 0.88, 1.0],
        ],
        dtype=np.float32,
    )

    best_path, best_score = best_continuity_path(similarity_matrix)

    assert sorted(best_path) == [0, 1, 2, 3]
    assert np.isfinite(best_score)
    assert best_path in ([0, 1, 2, 3], [3, 2, 1, 0])


def test_best_continuity_path_rejects_oversized_search() -> None:
    similarity_matrix = np.eye(9, dtype=np.float32)

    try:
        best_continuity_path(similarity_matrix, max_bruteforce_frames=8)
    except ValueError as exc:
        assert "supports at most 8 frames" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized continuity brute-force")


def test_permutation_score_components_include_direction_and_endpoint() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.90, 0.20, 0.05],
            [0.90, 1.0, 0.92, 0.15],
            [0.20, 0.92, 1.0, 0.91],
            [0.05, 0.15, 0.91, 1.0],
        ],
        dtype=np.float32,
    )
    frame_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def text_provider(_: list[str]) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    evidence = build_directional_evidence(frame_embeddings, "making tea", text_provider)
    components = permutation_score_components(
        similarity_matrix,
        [0, 1, 2, 3],
        frame_embeddings=frame_embeddings,
        directional_evidence=evidence,
    )

    assert set(components.keys()) == {"adjacency", "continuity", "direction", "endpoint"}
    assert components["adjacency"] > 0.0
    assert components["continuity"] >= 0.0
    assert np.isfinite(components["direction"])
    assert components["endpoint"] > 0.0


def test_disambiguate_reversal_prefers_forward_prompt_alignment() -> None:
    frame_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def text_provider(_: list[str]) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    evidence = build_directional_evidence(frame_embeddings, "assembling kit", text_provider)
    chosen, _ = disambiguate_reversal(
        candidate=[3, 2, 1, 0],
        directional_evidence=evidence,
        frame_embeddings=frame_embeddings,
    )

    assert chosen == [0, 1, 2, 3]


def test_continuity_only_returns_components_and_total_score() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.85],
            [0.2, 0.85, 1.0],
        ],
        dtype=np.float32,
    )

    components = continuity_only(similarity_matrix, [0, 1, 2], alpha=1.2, beta=0.7)

    assert components["alpha"] == 1.2
    assert components["beta"] == 0.7
    assert components["gamma"] == 0.0
    assert "adjacency_score" in components
    assert "continuity_score" in components
    assert "direction_score" in components
    assert "total_score" in components


def test_continuity_plus_text_direction_prefers_forward_order() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.88],
            [0.2, 0.88, 1.0],
        ],
        dtype=np.float32,
    )
    frame_to_prompt_similarity = {
        "start": np.array([0.95, 0.4, 0.1], dtype=np.float32),
        "middle": np.array([0.2, 0.96, 0.2], dtype=np.float32),
        "end": np.array([0.1, 0.3, 0.94], dtype=np.float32),
    }

    forward = continuity_plus_text_direction(
        similarity_matrix,
        [0, 1, 2],
        frame_to_prompt_similarity,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )
    reverse = continuity_plus_text_direction(
        similarity_matrix,
        [2, 1, 0],
        frame_to_prompt_similarity,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert forward["direction_score"] > reverse["direction_score"]
    assert forward["total_score"] > reverse["total_score"]


def test_best_continuity_plus_text_direction_path_returns_components() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.88],
            [0.2, 0.88, 1.0],
        ],
        dtype=np.float32,
    )
    frame_to_prompt_similarity = {
        "start": np.array([0.95, 0.4, 0.1], dtype=np.float32),
        "middle": np.array([0.2, 0.96, 0.2], dtype=np.float32),
        "end": np.array([0.1, 0.3, 0.94], dtype=np.float32),
    }

    best_path, components = best_continuity_plus_text_direction_path(
        similarity_matrix,
        frame_to_prompt_similarity,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert best_path == [0, 1, 2]
    assert np.isfinite(components["total_score"])


def test_best_continuity_only_path_returns_components() -> None:
    similarity_matrix = np.array(
        [
            [1.0, 0.95, 0.3],
            [0.95, 1.0, 0.9],
            [0.3, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    best_path, components = best_continuity_only_path(similarity_matrix)

    assert best_path in ([0, 1, 2], [2, 1, 0])
    assert np.isfinite(components["total_score"])


def test_best_oriented_path_selects_forward_via_text_direction() -> None:
    """Text direction should select forward orientation when continuity scores tie."""
    similarity_matrix = np.array(
        [
            [1.0, 0.92, 0.35, 0.10],
            [0.92, 1.0, 0.90, 0.32],
            [0.35, 0.90, 1.0, 0.88],
            [0.10, 0.32, 0.88, 1.0],
        ],
        dtype=np.float32,
    )
    # Text similarity strongly supports forward ordering [0, 1, 2, 3]
    frame_to_prompt_similarity = {
        "start": np.array([0.95, 0.40, 0.15, 0.05], dtype=np.float32),
        "middle": np.array([0.10, 0.80, 0.80, 0.10], dtype=np.float32),
        "end": np.array([0.05, 0.15, 0.40, 0.95], dtype=np.float32),
    }

    chosen_path, components = best_oriented_continuity_plus_text_direction_path(
        similarity_matrix,
        frame_to_prompt_similarity,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert chosen_path == [0, 1, 2, 3]
    # Symmetric similarity matrix → base scores always tie → direction tiebreak used
    assert components["is_ambiguous"] == 1.0
    assert components["used_direction_tiebreak"] == 1.0
    assert components["selected_is_reversed"] == 0.0


def test_best_oriented_path_reverses_when_text_supports_reverse() -> None:
    """Text direction should flip to reverse orientation when reverse scores higher."""
    similarity_matrix = np.array(
        [
            [1.0, 0.91, 0.38, 0.12],
            [0.91, 1.0, 0.89, 0.30],
            [0.38, 0.89, 1.0, 0.87],
            [0.12, 0.30, 0.87, 1.0],
        ],
        dtype=np.float32,
    )
    # Text similarity strongly supports reverse ordering [3, 2, 1, 0]
    frame_to_prompt_similarity = {
        "start": np.array([0.05, 0.15, 0.40, 0.95], dtype=np.float32),
        "middle": np.array([0.10, 0.80, 0.80, 0.10], dtype=np.float32),
        "end": np.array([0.95, 0.40, 0.15, 0.05], dtype=np.float32),
    }

    chosen_path, components = best_oriented_continuity_plus_text_direction_path(
        similarity_matrix,
        frame_to_prompt_similarity,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert chosen_path == [3, 2, 1, 0]
    assert components["selected_is_reversed"] == 1.0


def test_best_oriented_path_component_dict_has_expected_keys() -> None:
    """Returned component dict must contain scoring fields and orientation metadata."""
    similarity_matrix = np.array(
        [
            [1.0, 0.88, 0.20],
            [0.88, 1.0, 0.85],
            [0.20, 0.85, 1.0],
        ],
        dtype=np.float32,
    )
    frame_to_prompt_similarity = {
        "start": np.array([0.90, 0.30, 0.10], dtype=np.float32),
        "middle": np.array([0.20, 0.90, 0.20], dtype=np.float32),
        "end": np.array([0.10, 0.30, 0.90], dtype=np.float32),
    }

    _, components = best_oriented_continuity_plus_text_direction_path(
        similarity_matrix, frame_to_prompt_similarity
    )

    expected_keys = {
        # scoring components
        "alpha", "beta", "gamma",
        "adjacency_score", "continuity_score", "direction_score", "total_score",
        # orientation metadata
        "forward_score", "reverse_score", "score_gap", "abs_score_gap",
        "epsilon", "is_ambiguous", "used_direction_tiebreak",
        "forward_direction_score", "reverse_direction_score", "direction_gap",
        "selected_is_reversed", "selected_base_score", "selected_direction_score",
    }
    assert expected_keys.issubset(components.keys())
