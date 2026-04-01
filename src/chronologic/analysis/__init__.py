"""Diagnostics and failure-mode analysis utilities for ChronoLogic."""

from chronologic.analysis.alignment import plot_order_alignment
from chronologic.analysis.endpoint_analysis import (
    compute_endpoint_distinctiveness,
    plot_endpoint_distinctiveness,
)
from chronologic.analysis.error_taxonomy import (
    classify_prediction_error,
    summarize_error_taxonomy,
    write_error_taxonomy_summary,
)
from chronologic.analysis.forward_reverse import (
    compute_forward_reverse_scores,
    plot_forward_reverse_gap,
    write_forward_reverse_scores,
)
from chronologic.analysis.pairwise_errors import (
    compute_pairwise_error_matrix,
    pairwise_error_rows,
    plot_pairwise_error_matrix,
    write_pairwise_error_rows,
)
from chronologic.analysis.trajectory import (
    compute_adjacency_similarity_profile,
    compute_second_order_jump_profile,
    plot_embedding_trajectories,
    plot_sequence_profiles,
)

__all__ = [
    "classify_prediction_error",
    "compute_adjacency_similarity_profile",
    "compute_endpoint_distinctiveness",
    "compute_forward_reverse_scores",
    "compute_pairwise_error_matrix",
    "compute_second_order_jump_profile",
    "pairwise_error_rows",
    "plot_embedding_trajectories",
    "plot_endpoint_distinctiveness",
    "plot_forward_reverse_gap",
    "plot_order_alignment",
    "plot_pairwise_error_matrix",
    "plot_sequence_profiles",
    "summarize_error_taxonomy",
    "write_error_taxonomy_summary",
    "write_forward_reverse_scores",
    "write_pairwise_error_rows",
]
