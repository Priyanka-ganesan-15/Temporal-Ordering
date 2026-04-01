"""Metrics for ChronoLogic ordering evaluation."""

from __future__ import annotations

from itertools import combinations


def validate_permutation(pred: list[int], true: list[int]) -> None:
    """Validate two permutations contain the same unique items."""
    if len(pred) != len(true):
        raise ValueError("pred and true must have the same length")
    if len(set(pred)) != len(pred) or len(set(true)) != len(true):
        raise ValueError("pred and true must each contain unique indices")
    if sorted(pred) != sorted(true):
        raise ValueError("pred and true must contain the same indices")


def exact_match_accuracy(pred: list[int], true: list[int]) -> float:
    """Return 1.0 if the predicted permutation matches the truth exactly."""
    validate_permutation(pred, true)
    return 1.0 if pred == true else 0.0


def pairwise_order_accuracy(pred: list[int], true: list[int]) -> float:
    """Return the fraction of correctly ordered index pairs."""
    validate_permutation(pred, true)

    n_items = len(true)
    if n_items < 2:
        return 1.0

    predicted_positions = {item: index for index, item in enumerate(pred)}
    correct_pairs = 0
    total_pairs = 0
    for left_item, right_item in combinations(true, 2):
        total_pairs += 1
        if predicted_positions[left_item] < predicted_positions[right_item]:
            correct_pairs += 1

    return correct_pairs / total_pairs


def normalized_kendall_agreement(pred: list[int], true: list[int]) -> float:
    """Return normalized inversion agreement in the range [0, 1]."""
    validate_permutation(pred, true)

    max_inversions = _max_inversions(len(true))
    if max_inversions == 0:
        return 1.0
    return 1.0 - (count_inversions(pred, true) / max_inversions)


def kendall_tau_rank_agreement(pred: list[int], true: list[int]) -> float:
    """Return Kendall tau in the range [-1, 1]."""
    validate_permutation(pred, true)

    max_inversions = _max_inversions(len(true))
    if max_inversions == 0:
        return 1.0
    return 1.0 - (2.0 * count_inversions(pred, true) / max_inversions)


def count_inversions(pred: list[int], true: list[int]) -> int:
    """Count discordant pairs between predicted and true permutations."""
    validate_permutation(pred, true)

    predicted_positions = {item: index for index, item in enumerate(pred)}
    inversions = 0
    for left_item, right_item in combinations(true, 2):
        if predicted_positions[left_item] > predicted_positions[right_item]:
            inversions += 1
    return inversions


def compute_metrics(pred: list[int], true: list[int]) -> dict[str, float]:
    """Return the requested benchmark metrics for one prediction."""
    validate_permutation(pred, true)
    return {
        "exact_match_accuracy": exact_match_accuracy(pred, true),
        "pairwise_order_accuracy": pairwise_order_accuracy(pred, true),
        "normalized_kendall_agreement": normalized_kendall_agreement(pred, true),
    }


def evaluate_ordering_prediction(pred: list[int], true: list[int]) -> dict[str, float]:
    """Return an extended metric bundle for one prediction."""
    inversions = count_inversions(pred, true)
    return {
        "exact_match_accuracy": exact_match_accuracy(pred, true),
        "pairwise_order_accuracy": pairwise_order_accuracy(pred, true),
        "normalized_kendall_agreement": normalized_kendall_agreement(pred, true),
        "kendall_tau": kendall_tau_rank_agreement(pred, true),
        "inversion_count": float(inversions),
    }


def _max_inversions(n_items: int) -> int:
    return n_items * (n_items - 1) // 2


__all__ = [
    "count_inversions",
    "compute_metrics",
    "evaluate_ordering_prediction",
    "exact_match_accuracy",
    "kendall_tau_rank_agreement",
    "normalized_kendall_agreement",
    "pairwise_order_accuracy",
    "validate_permutation",
]