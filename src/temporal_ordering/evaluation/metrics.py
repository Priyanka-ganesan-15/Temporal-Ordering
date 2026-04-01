"""Legacy compatibility wrappers for temporal order metrics."""

from __future__ import annotations

from chronologic.evaluation.metrics import (
    count_inversions as _count_inversions,
    evaluate_ordering_prediction as _evaluate_ordering_prediction,
    exact_match_accuracy as _exact_match_accuracy,
    kendall_tau_rank_agreement as _kendall_tau_rank_agreement,
    normalized_kendall_agreement,
    pairwise_order_accuracy,
    validate_permutation,
)


def evaluate_ordering_prediction(
    ground_truth_order: list[int],
    predicted_order: list[int],
) -> dict[str, float]:
    """Return a standard metric bundle for one predicted ordering."""
    metrics = _evaluate_ordering_prediction(predicted_order, ground_truth_order)
    return {
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "pairwise_ordering_accuracy": metrics["pairwise_order_accuracy"],
        "kendall_tau": metrics["kendall_tau"],
        "normalized_inversion_score": metrics["normalized_kendall_agreement"],
        "inversion_count": metrics["inversion_count"],
    }


def exact_match_accuracy(ground_truth_order: list[int], predicted_order: list[int]) -> float:
    """Return 1.0 if the full order matches exactly, otherwise 0.0."""
    validate_permutation(predicted_order, ground_truth_order)
    return _exact_match_accuracy(predicted_order, ground_truth_order)


def pairwise_ordering_accuracy(
    ground_truth_order: list[int],
    predicted_order: list[int],
) -> float:
    """Return the fraction of pairs with the correct relative order."""
    validate_permutation(predicted_order, ground_truth_order)
    return pairwise_order_accuracy(predicted_order, ground_truth_order)


def kendall_tau_rank_agreement(
    ground_truth_order: list[int],
    predicted_order: list[int],
    inversion_count: int | None = None,
) -> float:
    """Return Kendall tau in the range [-1, 1]."""
    validate_permutation(predicted_order, ground_truth_order)
    if inversion_count is not None:
        max_inversions = _max_inversions(len(ground_truth_order))
        if max_inversions == 0:
            return 1.0
        return 1.0 - (2.0 * inversion_count / max_inversions)
    return _kendall_tau_rank_agreement(predicted_order, ground_truth_order)


def normalized_inversion_score(
    ground_truth_order: list[int],
    predicted_order: list[int],
    inversion_count: int | None = None,
) -> float:
    """Return 1.0 for perfect order and 0.0 for maximal inversion."""
    validate_permutation(predicted_order, ground_truth_order)
    if inversion_count is not None:
        max_inversions = _max_inversions(len(ground_truth_order))
        if max_inversions == 0:
            return 1.0
        return 1.0 - (inversion_count / max_inversions)
    return normalized_kendall_agreement(predicted_order, ground_truth_order)


def count_inversions(ground_truth_order: list[int], predicted_order: list[int]) -> int:
    """Count discordant pairs between two permutations."""
    validate_permutation(predicted_order, ground_truth_order)
    return _count_inversions(predicted_order, ground_truth_order)


def _max_inversions(n_items: int) -> int:
    return n_items * (n_items - 1) // 2
