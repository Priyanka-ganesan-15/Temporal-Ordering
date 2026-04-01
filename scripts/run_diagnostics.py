"""Run ChronoLogic diagnostics for ordering failure modes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronologic.analysis.alignment import plot_order_alignment
from chronologic.analysis.endpoint_analysis import (
    compute_endpoint_distinctiveness,
    plot_endpoint_distinctiveness,
    write_endpoint_rows,
)
from chronologic.analysis.error_taxonomy import (
    classify_prediction_error,
    plot_error_taxonomy_summary,
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
    pca_project_2d,
    plot_embedding_trajectories,
    plot_sequence_profiles,
)
from chronologic.ordering.continuity import best_continuity_path
from chronologic.ordering.nearest_neighbor import best_greedy_path
from chronologic.ordering.random_baseline import random_permutation
from temporal_ordering.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_ORDERING_OUTPUT_DIR,
    DEFAULT_PRETRAINED,
    DEFAULT_RANDOM_SEED,
    project_root,
    resolve_project_path,
)
from temporal_ordering.data_loader import load_sequences
from temporal_ordering.embedding import OpenCLIPEmbedder
from temporal_ordering.similarity import cosine_similarity_matrix, sequence_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChronoLogic ordering diagnostics")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_ORDERING_OUTPUT_DIR) / "diagnostics"),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = load_sequences(args.manifest)
    embedder = OpenCLIPEmbedder(model_name=args.model, pretrained=args.pretrained)

    cache_dir = resolve_project_path(args.cache_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_reverse_rows: list[dict[str, str | float]] = []
    pairwise_rows_all: list[dict[str, str | int | float]] = []
    endpoint_rows_all: list[dict[str, str | int | float]] = []
    taxonomy_rows: list[dict[str, str | float]] = []
    prediction_rows: list[dict[str, str | float]] = []

    for sequence_index, sequence in enumerate(sequences):
        embeddings = sequence_embeddings(
            sequence,
            embedder,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            force_recompute=args.force_embeddings,
        )
        similarity_matrix = cosine_similarity_matrix(embeddings)
        n_frames = sequence.num_frames
        true_order = list(range(n_frames))

        random_order = random_permutation(n_frames, seed=(args.seed + sequence_index) * 101 + 1)
        greedy_order, greedy_score = best_greedy_path(similarity_matrix)
        continuity_order, continuity_score = best_continuity_path(similarity_matrix)

        method_predictions: dict[str, list[int]] = {
            "random": random_order,
            "greedy_nearest_neighbor": greedy_order,
            "continuity": continuity_order,
        }
        method_scores: dict[str, float | None] = {
            "random": None,
            "greedy_nearest_neighbor": greedy_score,
            "continuity": continuity_score,
        }

        forward_reverse_rows.append(
            compute_forward_reverse_scores(
                sequence_id=sequence.sequence_id,
                similarity_matrix=similarity_matrix,
            )
        )

        plot_order_alignment(
            sequence_id=sequence.sequence_id,
            method_predictions=method_predictions,
            output_path=output_dir / "alignment" / f"{sequence.sequence_id}_alignment.png",
        )

        for method, predicted_order in method_predictions.items():
            matrix = compute_pairwise_error_matrix(predicted_order, true_order)
            pairwise_rows_all.extend(
                pairwise_error_rows(
                    sequence_id=sequence.sequence_id,
                    method=method,
                    predicted_order=predicted_order,
                    true_order=true_order,
                )
            )
            plot_pairwise_error_matrix(
                sequence_id=sequence.sequence_id,
                method=method,
                pairwise_matrix=matrix,
                output_path=(
                    output_dir
                    / "pairwise_errors"
                    / f"{sequence.sequence_id}_{method}_pairwise.png"
                ),
            )

            taxonomy_rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "method": method,
                    "taxonomy": classify_prediction_error(predicted_order, true_order),
                }
            )
            prediction_rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "method": method,
                    "predicted_order": _serialize_int_list(predicted_order),
                    "score": float(method_scores[method]) if method_scores[method] is not None else float("nan"),
                }
            )

        adjacency_profile = compute_adjacency_similarity_profile(embeddings, true_order)
        second_order_profile = compute_second_order_jump_profile(embeddings, true_order)
        plot_sequence_profiles(
            sequence_id=sequence.sequence_id,
            adjacency_profile=adjacency_profile,
            second_order_profile=second_order_profile,
            output_path=output_dir / "trajectory" / f"{sequence.sequence_id}_profiles.png",
        )

        projected = pca_project_2d(embeddings)
        plot_embedding_trajectories(
            sequence_id=sequence.sequence_id,
            projected_2d=projected,
            true_order=true_order,
            method_predictions=method_predictions,
            output_path=output_dir / "trajectory" / f"{sequence.sequence_id}_trajectory.png",
        )

        endpoint_rows = compute_endpoint_distinctiveness(
            sequence_id=sequence.sequence_id,
            embeddings=embeddings,
            similarity_matrix=similarity_matrix,
        )
        endpoint_rows_all.extend(endpoint_rows)
        plot_endpoint_distinctiveness(
            sequence_id=sequence.sequence_id,
            rows=endpoint_rows,
            output_path=output_dir / "endpoint" / f"{sequence.sequence_id}_endpoint.png",
        )

    forward_reverse_rows.sort(key=lambda row: str(row["sequence_id"]))
    write_forward_reverse_scores(
        output_dir / "forward_reverse" / "forward_reverse_scores.csv",
        forward_reverse_rows,
    )
    plot_forward_reverse_gap(
        forward_reverse_rows,
        output_dir / "forward_reverse" / "forward_reverse_gap.png",
    )

    write_pairwise_error_rows(
        output_dir / "pairwise_errors" / "pairwise_error_rows.csv",
        pairwise_rows_all,
    )
    write_endpoint_rows(
        output_dir / "endpoint" / "endpoint_distinctiveness.csv",
        endpoint_rows_all,
    )

    write_rows_csv(
        output_dir / "predictions.csv",
        prediction_rows,
    )

    write_error_taxonomy_summary(
        output_dir / "error_taxonomy" / "taxonomy_rows.csv",
        taxonomy_rows,
    )
    taxonomy_summary = summarize_error_taxonomy(taxonomy_rows)
    write_error_taxonomy_summary(
        output_dir / "error_taxonomy" / "taxonomy_summary.csv",
        taxonomy_summary,
    )
    plot_error_taxonomy_summary(
        taxonomy_summary,
        output_dir / "error_taxonomy" / "taxonomy_summary.png",
    )

    root = project_root()
    print("Diagnostics completed. Outputs:")
    print(f"  {(output_dir / 'forward_reverse' / 'forward_reverse_scores.csv').relative_to(root)}")
    print(f"  {(output_dir / 'forward_reverse' / 'forward_reverse_gap.png').relative_to(root)}")
    print(f"  {(output_dir / 'alignment').relative_to(root)}")
    print(f"  {(output_dir / 'pairwise_errors' / 'pairwise_error_rows.csv').relative_to(root)}")
    print(f"  {(output_dir / 'trajectory').relative_to(root)}")
    print(f"  {(output_dir / 'endpoint' / 'endpoint_distinctiveness.csv').relative_to(root)}")
    print(f"  {(output_dir / 'error_taxonomy' / 'taxonomy_summary.csv').relative_to(root)}")
    print(f"  {(output_dir / 'error_taxonomy' / 'taxonomy_summary.png').relative_to(root)}")


def write_rows_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    """Write generic tabular rows to CSV."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_int_list(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


if __name__ == "__main__":
    main()
