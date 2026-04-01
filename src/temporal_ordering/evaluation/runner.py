"""Legacy compatibility wrappers for the ChronoLogic evaluation runner."""

from chronologic.evaluation.runner import (
    build_method_summary,
    build_sequence_summary,
    evaluate_method_on_sequence,
    evaluate_sequence,
    evaluate_sequences,
    parse_evaluation_args,
    print_method_summary,
    print_sequence_difficulty_summary,
    run_evaluation_cli,
    run_full_evaluation,
    save_results_dataframe,
    shuffled_similarity_inputs,
    write_csv,
)

__all__ = [
    "build_method_summary",
    "build_sequence_summary",
    "evaluate_method_on_sequence",
    "evaluate_sequence",
    "evaluate_sequences",
    "parse_evaluation_args",
    "print_method_summary",
    "print_sequence_difficulty_summary",
    "run_evaluation_cli",
    "run_full_evaluation",
    "save_results_dataframe",
    "shuffled_similarity_inputs",
    "write_csv",
]
