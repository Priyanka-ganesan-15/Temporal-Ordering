"""Legacy compatibility exports for temporal ordering evaluation."""

from temporal_ordering.evaluation.metrics import evaluate_ordering_prediction
from temporal_ordering.evaluation.runner import (
    evaluate_sequences,
    parse_evaluation_args,
    run_evaluation_cli,
)

__all__ = [
    "evaluate_ordering_prediction",
    "evaluate_sequences",
    "parse_evaluation_args",
    "run_evaluation_cli",
]