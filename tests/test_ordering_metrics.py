import pytest

from chronologic.evaluation.metrics import (
    compute_metrics,
    normalized_kendall_agreement,
    validate_permutation,
)


def test_compute_metrics_for_perfect_order() -> None:
    metrics = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])

    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["pairwise_order_accuracy"] == 1.0
    assert metrics["normalized_kendall_agreement"] == 1.0


def test_compute_metrics_for_reversed_order() -> None:
    metrics = compute_metrics([3, 2, 1, 0], [0, 1, 2, 3])

    assert metrics["exact_match_accuracy"] == 0.0
    assert metrics["pairwise_order_accuracy"] == 0.0
    assert metrics["normalized_kendall_agreement"] == 0.0
    assert normalized_kendall_agreement([3, 2, 1, 0], [0, 1, 2, 3]) == 0.0


def test_validate_permutation_rejects_mismatched_indices() -> None:
    with pytest.raises(ValueError, match="same indices"):
        validate_permutation([0, 1, 2], [0, 1, 3])