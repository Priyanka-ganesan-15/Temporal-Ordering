"""Continuity-aware exhaustive ordering for short frame sequences."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable

import numpy as np

from chronologic.ordering.nearest_neighbor import validate_similarity_matrix
from chronologic.ordering.reverse_disambiguation import choose_oriented_path
from chronologic.ordering.text_direction import temporal_direction_score

TextEmbeddingProvider = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class ContinuityScoreWeights:
    """Weights for each objective component in the ordering score."""

    adjacency: float = 1.0
    continuity: float = 1.0
    direction: float = 0.0
    endpoint: float = 0.0


@dataclass(frozen=True)
class DirectionalEvidence:
    """Per-frame direction clues used by direction and endpoint terms."""

    frame_phase_similarities: np.ndarray | None
    frame_progress: np.ndarray | None


def build_directional_evidence(
    frame_embeddings: np.ndarray,
    caption: str,
    text_embedding_provider: TextEmbeddingProvider | None,
) -> DirectionalEvidence:
    """Build direction-sensitive clues from start/middle/end text prompts."""
    if text_embedding_provider is None:
        return DirectionalEvidence(frame_phase_similarities=None, frame_progress=None)

    caption_text = caption.strip()
    if not caption_text:
        return DirectionalEvidence(frame_phase_similarities=None, frame_progress=None)

    normalized_frames = _normalize_rows(frame_embeddings)
    prompts = [
        f"start of sequence: {caption_text}",
        f"middle of sequence: {caption_text}",
        f"end of sequence: {caption_text}",
    ]
    text_embeddings = np.asarray(text_embedding_provider(prompts), dtype=np.float64)
    if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 3:
        raise ValueError("text_embedding_provider must return array with shape (3, embedding_dim)")

    normalized_text = _normalize_rows(text_embeddings)
    phase_similarities = normalized_frames @ normalized_text.T
    frame_progress = phase_similarities[:, 2] - phase_similarities[:, 0]
    return DirectionalEvidence(
        frame_phase_similarities=phase_similarities,
        frame_progress=frame_progress,
    )


def permutation_score_components(
    similarity_matrix: np.ndarray,
    candidate: list[int],
    frame_embeddings: np.ndarray | None = None,
    directional_evidence: DirectionalEvidence | None = None,
) -> dict[str, float]:
    """Return modular score components for a candidate permutation."""
    validate_similarity_matrix(similarity_matrix)
    _validate_candidate_path(candidate, similarity_matrix.shape[0])

    adjacency_score = _adjacency_score(similarity_matrix, candidate)
    continuity_penalty = _continuity_penalty(similarity_matrix, candidate)

    direction_score = 0.0
    if frame_embeddings is not None:
        direction_score = _direction_score(
            frame_embeddings=frame_embeddings,
            candidate=candidate,
            directional_evidence=directional_evidence,
        )

    endpoint_score = _endpoint_score(candidate, directional_evidence)

    return {
        "adjacency": adjacency_score,
        "continuity": continuity_penalty,
        "direction": direction_score,
        "endpoint": endpoint_score,
    }


def score_permutation_with_continuity(
    similarity_matrix: np.ndarray,
    candidate: list[int],
    adjacency_weight: float = 1.0,
    continuity_weight: float = 1.0,
    direction_weight: float = 0.0,
    endpoint_weight: float = 0.0,
    frame_embeddings: np.ndarray | None = None,
    directional_evidence: DirectionalEvidence | None = None,
) -> float:
    """Score a candidate permutation using adjacency reward and continuity penalty.

    The objective is:
        adjacency_weight * adjacency
        - continuity_weight * continuity
        + direction_weight * direction
        + endpoint_weight * endpoint

    where d_t = 1 - sim(i_t, i_{t+1}) is cosine distance between adjacent frames.
    """
    components = permutation_score_components(
        similarity_matrix=similarity_matrix,
        candidate=candidate,
        frame_embeddings=frame_embeddings,
        directional_evidence=directional_evidence,
    )
    return (
        adjacency_weight * components["adjacency"]
        - continuity_weight * components["continuity"]
        + direction_weight * components["direction"]
        + endpoint_weight * components["endpoint"]
    )


def continuity_only(
    similarity_matrix: np.ndarray,
    path: list[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, float]:
    """Score a path with adjacency + continuity only.

    This keeps continuity-only scoring explicit and ablation-friendly.
    continuity_score is defined as a reward (higher is better) by negating
    the continuity penalty.
    """
    return _weighted_score_components(
        similarity_matrix=similarity_matrix,
        path=path,
        alpha=alpha,
        beta=beta,
        gamma=0.0,
        frame_to_prompt_similarity=None,
    )


def continuity_plus_text_direction(
    similarity_matrix: np.ndarray,
    path: list[int],
    frame_to_prompt_similarity: dict[str, np.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Score a path with adjacency + continuity + text-direction alignment.

    total_score =
      alpha * adjacency_score
      + beta * continuity_score
      + gamma * direction_score
    """
    return _weighted_score_components(
        similarity_matrix=similarity_matrix,
        path=path,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_to_prompt_similarity=frame_to_prompt_similarity,
    )


def best_continuity_plus_text_direction_path(
    similarity_matrix: np.ndarray,
    frame_to_prompt_similarity: dict[str, np.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    max_bruteforce_frames: int = 8,
) -> tuple[list[int], dict[str, float]]:
    """Return best path under continuity_plus_text_direction objective."""
    return _best_weighted_path(
        similarity_matrix=similarity_matrix,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_to_prompt_similarity=frame_to_prompt_similarity,
        max_bruteforce_frames=max_bruteforce_frames,
    )


def best_continuity_only_path(
    similarity_matrix: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_bruteforce_frames: int = 8,
) -> tuple[list[int], dict[str, float]]:
    """Return best path under continuity_only objective."""
    return _best_weighted_path(
        similarity_matrix=similarity_matrix,
        alpha=alpha,
        beta=beta,
        gamma=0.0,
        frame_to_prompt_similarity=None,
        max_bruteforce_frames=max_bruteforce_frames,
    )


def best_oriented_continuity_plus_text_direction_path(
    similarity_matrix: np.ndarray,
    frame_to_prompt_similarity: dict[str, np.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    max_bruteforce_frames: int = 8,
) -> tuple[list[int], dict[str, float]]:
    """Find best path via two-stage continuity search + explicit orientation resolution.

    Stage 1 — Path search:
        Exhaustive search over all permutations scored by continuity_only
        (alpha * adjacency + beta * continuity, no direction term) to find
        the best unoriented path.  Because continuity scoring is symmetric,
        both orientations of the optimal sequence achieve equal scores; the
        search simply surfaces one of the two.

    Stage 2 — Orientation resolution (choose_oriented_path):
        Compare the candidate path and its reverse under the continuity_only
        base score.  When the gap is within epsilon (always true for symmetric
        similarity matrices), temporal_direction_score breaks the tie.

    The chosen path is then scored with the full continuity_plus_text_direction
    objective (alpha * adjacency + beta * continuity + gamma * direction).

    Returns:
        chosen_path : list[int]
            Frame indices in the predicted temporal order.
        components : dict[str, float]
            Scoring components (adjacency_score, continuity_score,
            direction_score, total_score, weights) merged with orientation
            metadata from choose_oriented_path (forward_score, reverse_score,
            is_ambiguous, used_direction_tiebreak, selected_is_reversed, etc.).
    """
    best_path, _ = _best_weighted_path(
        similarity_matrix=similarity_matrix,
        alpha=alpha,
        beta=beta,
        gamma=0.0,
        frame_to_prompt_similarity=None,
        max_bruteforce_frames=max_bruteforce_frames,
    )

    def _base_score_fn(p: list[int]) -> float:
        return _weighted_score_components(
            similarity_matrix=similarity_matrix,
            path=p,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            frame_to_prompt_similarity=None,
        )["total_score"]

    def _direction_score_fn(p: list[int]) -> float:
        return temporal_direction_score(p, frame_to_prompt_similarity)

    chosen_path, orientation_metrics = choose_oriented_path(
        best_path,
        _base_score_fn,
        direction_score_fn=_direction_score_fn,
        epsilon=epsilon,
    )

    components = _weighted_score_components(
        similarity_matrix=similarity_matrix,
        path=chosen_path,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        frame_to_prompt_similarity=frame_to_prompt_similarity,
    )
    components.update(orientation_metrics)
    return chosen_path, components


def best_continuity_path(
    similarity_matrix: np.ndarray,
    adjacency_weight: float = 1.0,
    continuity_weight: float = 1.0,
    direction_weight: float = 0.0,
    endpoint_weight: float = 0.0,
    frame_embeddings: np.ndarray | None = None,
    directional_evidence: DirectionalEvidence | None = None,
    max_bruteforce_frames: int = 8,
) -> tuple[list[int], float]:
    """Return the highest-scoring path under the continuity-aware objective.

    Exhaustive search is used for short sequences up to max_bruteforce_frames.
    """
    validate_similarity_matrix(similarity_matrix)
    n_frames = similarity_matrix.shape[0]
    if n_frames > max_bruteforce_frames:
        raise ValueError(
            "continuity exhaustive search supports at most "
            f"{max_bruteforce_frames} frames, got {n_frames}"
        )

    best_path: list[int] | None = None
    best_score = float("-inf")
    for candidate_tuple in permutations(range(n_frames)):
        candidate = list(candidate_tuple)
        score = score_permutation_with_continuity(
            similarity_matrix,
            candidate,
            adjacency_weight=adjacency_weight,
            continuity_weight=continuity_weight,
            direction_weight=direction_weight,
            endpoint_weight=endpoint_weight,
            frame_embeddings=frame_embeddings,
            directional_evidence=directional_evidence,
        )
        if score > best_score:
            best_path = candidate
            best_score = score

    if best_path is None:
        raise RuntimeError("Failed to construct a continuity-aware ordering path")
    return best_path, best_score


def disambiguate_reversal(
    candidate: list[int],
    directional_evidence: DirectionalEvidence | None,
    frame_embeddings: np.ndarray | None = None,
    direction_weight: float = 1.0,
    endpoint_weight: float = 1.0,
) -> tuple[list[int], float]:
    """Compare candidate with its reverse and keep the direction-preferred orientation."""
    reverse_candidate = list(reversed(candidate))

    candidate_support = _directional_support_score(
        candidate,
        directional_evidence=directional_evidence,
        frame_embeddings=frame_embeddings,
        direction_weight=direction_weight,
        endpoint_weight=endpoint_weight,
    )
    reverse_support = _directional_support_score(
        reverse_candidate,
        directional_evidence=directional_evidence,
        frame_embeddings=frame_embeddings,
        direction_weight=direction_weight,
        endpoint_weight=endpoint_weight,
    )

    if candidate_support >= reverse_support:
        return candidate, candidate_support
    return reverse_candidate, reverse_support


def _adjacency_score(similarity_matrix: np.ndarray, candidate: list[int]) -> float:
    if len(candidate) < 2:
        return 0.0
    edge_similarities = np.array(
        [similarity_matrix[current, nxt] for current, nxt in zip(candidate, candidate[1:])],
        dtype=np.float64,
    )
    return float(edge_similarities.sum())


def _continuity_penalty(similarity_matrix: np.ndarray, candidate: list[int]) -> float:
    if len(candidate) < 3:
        return 0.0
    edge_similarities = np.array(
        [similarity_matrix[current, nxt] for current, nxt in zip(candidate, candidate[1:])],
        dtype=np.float64,
    )
    edge_distances = 1.0 - edge_similarities
    return float(np.abs(np.diff(edge_distances)).sum())


def _direction_score(
    frame_embeddings: np.ndarray,
    candidate: list[int],
    directional_evidence: DirectionalEvidence | None,
) -> float:
    if len(candidate) < 3:
        return 0.0

    ordered_embeddings = np.asarray(frame_embeddings[candidate], dtype=np.float64)
    deltas = np.diff(ordered_embeddings, axis=0)
    delta_consistency = 0.0
    if deltas.shape[0] >= 2:
        normalized_deltas = _normalize_rows(deltas)
        alignment = np.sum(normalized_deltas[:-1] * normalized_deltas[1:], axis=1)
        delta_consistency = float(np.mean(alignment))

    progress_monotonicity = 0.0
    phase_alignment = 0.0
    if directional_evidence is not None and directional_evidence.frame_progress is not None:
        progress = directional_evidence.frame_progress[candidate]
        progress_deltas = np.diff(progress)
        scale = float(np.std(directional_evidence.frame_progress)) + 1e-8
        progress_monotonicity = float(np.mean(np.tanh(progress_deltas / scale)))

    if directional_evidence is not None and directional_evidence.frame_phase_similarities is not None:
        phase_sim = directional_evidence.frame_phase_similarities[candidate]
        n_frames = len(candidate)
        positions = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
        w_start = 1.0 - positions
        w_middle = np.maximum(0.0, 1.0 - 2.0 * np.abs(positions - 0.5))
        w_end = positions
        phase_curve = (
            w_start * phase_sim[:, 0]
            + w_middle * phase_sim[:, 1]
            + w_end * phase_sim[:, 2]
        )
        phase_alignment = float(np.mean(phase_curve))

    return delta_consistency + progress_monotonicity + phase_alignment


def _endpoint_score(
    candidate: list[int],
    directional_evidence: DirectionalEvidence | None,
) -> float:
    if directional_evidence is None or directional_evidence.frame_phase_similarities is None:
        return 0.0
    phase_sim = directional_evidence.frame_phase_similarities
    first_frame = candidate[0]
    last_frame = candidate[-1]
    return float(
        (phase_sim[first_frame, 0] + phase_sim[last_frame, 2])
        - (phase_sim[first_frame, 2] + phase_sim[last_frame, 0])
    )


def _directional_support_score(
    candidate: list[int],
    directional_evidence: DirectionalEvidence | None,
    frame_embeddings: np.ndarray | None,
    direction_weight: float,
    endpoint_weight: float,
) -> float:
    direction_score = 0.0
    if frame_embeddings is not None:
        direction_score = _direction_score(
            frame_embeddings=frame_embeddings,
            candidate=candidate,
            directional_evidence=directional_evidence,
        )
    endpoint_score = _endpoint_score(candidate, directional_evidence)
    return (direction_weight * direction_score) + (endpoint_weight * endpoint_score)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-8)
    return vectors / safe_norms


def _weighted_score_components(
    similarity_matrix: np.ndarray,
    path: list[int],
    alpha: float,
    beta: float,
    gamma: float,
    frame_to_prompt_similarity: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    validate_similarity_matrix(similarity_matrix)
    _validate_candidate_path(path, similarity_matrix.shape[0])

    adjacency_score = _adjacency_score(similarity_matrix, path)
    continuity_score = -_continuity_penalty(similarity_matrix, path)
    direction_score = (
        temporal_direction_score(path, frame_to_prompt_similarity)
        if frame_to_prompt_similarity is not None
        else 0.0
    )
    total_score = (
        alpha * adjacency_score
        + beta * continuity_score
        + gamma * direction_score
    )
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "adjacency_score": float(adjacency_score),
        "continuity_score": float(continuity_score),
        "direction_score": float(direction_score),
        "total_score": float(total_score),
    }


def _best_weighted_path(
    similarity_matrix: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    frame_to_prompt_similarity: dict[str, np.ndarray] | None,
    max_bruteforce_frames: int,
) -> tuple[list[int], dict[str, float]]:
    validate_similarity_matrix(similarity_matrix)
    n_frames = similarity_matrix.shape[0]
    if n_frames > max_bruteforce_frames:
        raise ValueError(
            "continuity exhaustive search supports at most "
            f"{max_bruteforce_frames} frames, got {n_frames}"
        )

    best_path: list[int] | None = None
    best_components: dict[str, float] | None = None
    best_score = float("-inf")

    for candidate_tuple in permutations(range(n_frames)):
        candidate = list(candidate_tuple)
        components = _weighted_score_components(
            similarity_matrix=similarity_matrix,
            path=candidate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            frame_to_prompt_similarity=frame_to_prompt_similarity,
        )
        score = components["total_score"]
        if score > best_score:
            best_path = candidate
            best_components = components
            best_score = score

    if best_path is None or best_components is None:
        raise RuntimeError("Failed to construct a continuity-aware ordering path")
    return best_path, best_components


def _validate_candidate_path(path: list[int], n_frames: int) -> None:
    if len(path) != n_frames:
        raise ValueError("candidate path must include each frame exactly once")
    if sorted(path) != list(range(n_frames)):
        raise ValueError("candidate path must be a permutation of frame indices")


__all__ = [
    "ContinuityScoreWeights",
    "DirectionalEvidence",
    "best_continuity_only_path",
    "best_continuity_path",
    "best_continuity_plus_text_direction_path",
    "best_oriented_continuity_plus_text_direction_path",
    "build_directional_evidence",
    "continuity_only",
    "continuity_plus_text_direction",
    "disambiguate_reversal",
    "permutation_score_components",
    "score_permutation_with_continuity",
]
