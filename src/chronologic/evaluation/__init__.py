"""Evaluation helpers exposed under the ChronoLogic package name."""

from chronologic.evaluation.metrics import (
    compute_metrics,
    evaluate_ordering_prediction,
    exact_match_accuracy,
    kendall_tau_rank_agreement,
    normalized_kendall_agreement,
    pairwise_order_accuracy,
    validate_permutation,
)
from chronologic.evaluation.runner import (
    evaluate_method_on_sequence,
    evaluate_sequence,
    evaluate_sequences,
    parse_evaluation_args,
    run_evaluation_cli,
    run_full_evaluation,
    save_results_dataframe,
)

__all__ = [
    "compute_metrics",
    "evaluate_ordering_prediction",
    "evaluate_method_on_sequence",
    "evaluate_sequence",
    "evaluate_sequences",
    "exact_match_accuracy",
    "kendall_tau_rank_agreement",
    "normalized_kendall_agreement",
    "parse_evaluation_args",
    "pairwise_order_accuracy",
    "run_evaluation_cli",
    "run_full_evaluation",
    "save_results_dataframe",
    "validate_permutation",
]