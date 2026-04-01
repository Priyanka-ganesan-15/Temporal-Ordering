"""Helpers to resolve forward-vs-reverse orientation ambiguity."""

from __future__ import annotations

from typing import Callable


def compare_forward_reverse_scores(
    path: list[int],
    base_score_fn: Callable[[list[int]], float],
) -> dict[str, float]:
    """Compare base scores for a path and its reversed orientation.

    Returns a float-only metrics dictionary for simple downstream logging.
    """
    if not path:
        raise ValueError("path must not be empty")

    reverse_path = list(reversed(path))
    forward_score = float(base_score_fn(path))
    reverse_score = float(base_score_fn(reverse_path))
    score_gap = forward_score - reverse_score

    return {
        "forward_score": forward_score,
        "reverse_score": reverse_score,
        "score_gap": score_gap,
        "abs_score_gap": abs(score_gap),
    }


def choose_oriented_path(
    path: list[int],
    base_score_fn: Callable[[list[int]], float],
    direction_score_fn: Callable[[list[int]], float] | None = None,
    epsilon: float = 1e-6,
) -> tuple[list[int], dict[str, float]]:
    """Choose a canonical orientation using base score and optional direction tie-break.

    Selection rules:
    1. Compute base score for forward and reverse.
    2. If the absolute base-score gap is larger than epsilon, choose the higher base score.
    3. If base scores are ambiguous (gap <= epsilon):
       - use direction_score_fn if provided and choose higher direction score
       - otherwise default to forward orientation
    """
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")

    reverse_path = list(reversed(path))
    base_metrics = compare_forward_reverse_scores(path, base_score_fn)

    forward_base = base_metrics["forward_score"]
    reverse_base = base_metrics["reverse_score"]
    base_gap = base_metrics["score_gap"]
    ambiguous = abs(base_gap) <= epsilon

    direction_used = 0.0
    forward_direction = 0.0
    reverse_direction = 0.0
    direction_gap = 0.0

    selected_path = path
    selected_is_reversed = 0.0

    if not ambiguous:
        if reverse_base > forward_base:
            selected_path = reverse_path
            selected_is_reversed = 1.0
    else:
        if direction_score_fn is not None:
            direction_used = 1.0
            forward_direction = float(direction_score_fn(path))
            reverse_direction = float(direction_score_fn(reverse_path))
            direction_gap = forward_direction - reverse_direction
            if reverse_direction > forward_direction:
                selected_path = reverse_path
                selected_is_reversed = 1.0

    selected_base_score = reverse_base if selected_is_reversed else forward_base
    selected_direction_score = reverse_direction if selected_is_reversed else forward_direction

    metrics: dict[str, float] = {
        "forward_score": forward_base,
        "reverse_score": reverse_base,
        "score_gap": base_gap,
        "abs_score_gap": abs(base_gap),
        "epsilon": float(epsilon),
        "is_ambiguous": 1.0 if ambiguous else 0.0,
        "used_direction_tiebreak": direction_used,
        "forward_direction_score": forward_direction,
        "reverse_direction_score": reverse_direction,
        "direction_gap": direction_gap,
        "selected_is_reversed": selected_is_reversed,
        "selected_base_score": selected_base_score,
        "selected_direction_score": selected_direction_score,
    }
    return selected_path, metrics


__all__ = [
    "choose_oriented_path",
    "compare_forward_reverse_scores",
]
