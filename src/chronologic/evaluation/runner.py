"""Evaluation runner for ChronoLogic ordering experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from chronologic.evaluation.metrics import evaluate_ordering_prediction
from chronologic.ordering.continuity import (
    best_continuity_path,
    build_directional_evidence,
    disambiguate_reversal,
)
from chronologic.ordering.nearest_neighbor import best_greedy_path, validate_similarity_matrix
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
from temporal_ordering.models import Sequence
from temporal_ordering.similarity import (
    cosine_similarity_matrix,
    sequence_embeddings,
    temporal_structure_score,
)

EmbeddingProvider = Callable[[Sequence], np.ndarray]
TextEmbeddingProvider = Callable[[list[str]], np.ndarray]
MetricRow = dict[str, str | float]

CONTINUITY_DIRECTION_WEIGHT = 0.75
CONTINUITY_ENDPOINT_WEIGHT = 0.75
REVERSE_DISAMBIGUATION_DIRECTION_WEIGHT = 1.0
REVERSE_DISAMBIGUATION_ENDPOINT_WEIGHT = 1.0

METRIC_COLUMNS = [
    "exact_match_accuracy",
    "pairwise_order_accuracy",
    "normalized_kendall_agreement",
    "kendall_tau",
    "inversion_count",
]


def parse_evaluation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal ordering baselines")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_ORDERING_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--near-gap", type=int, default=1)
    parser.add_argument("--far-gap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    return parser.parse_args()


def evaluate_method_on_sequence(
    sequence: Sequence,
    embedding_provider: EmbeddingProvider,
    seed: int,
    text_embedding_provider: TextEmbeddingProvider | None = None,
) -> list[MetricRow]:
    """Evaluate all registered methods on a single sequence."""
    results_rows, _, _ = evaluate_sequences(
        [sequence],
        embedding_provider=embedding_provider,
        seed=seed,
        text_embedding_provider=text_embedding_provider,
    )
    return results_rows


def evaluate_sequence(
    sequence: Sequence,
    embedding_provider: EmbeddingProvider,
    seed: int,
    text_embedding_provider: TextEmbeddingProvider | None = None,
) -> list[MetricRow]:
    """Alias for sequence-level evaluation."""
    return evaluate_method_on_sequence(
        sequence,
        embedding_provider=embedding_provider,
        seed=seed,
        text_embedding_provider=text_embedding_provider,
    )


def run_full_evaluation(
    sequences: list[Sequence],
    embedding_provider: EmbeddingProvider,
    seed: int,
    text_embedding_provider: TextEmbeddingProvider | None = None,
) -> tuple[list[MetricRow], list[MetricRow], list[MetricRow]]:
    """Run the full ordering benchmark."""
    return evaluate_sequences(
        sequences,
        embedding_provider=embedding_provider,
        seed=seed,
        text_embedding_provider=text_embedding_provider,
    )


def evaluate_sequences(
    sequences: list[Sequence],
    embedding_provider: EmbeddingProvider,
    seed: int = DEFAULT_RANDOM_SEED,
    near_gap: int = 1,
    far_gap: int = 4,
    text_embedding_provider: TextEmbeddingProvider | None = None,
) -> tuple[list[MetricRow], list[MetricRow], list[MetricRow]]:
    """Evaluate the registered ordering baselines across all sequences."""
    results_rows: list[MetricRow] = []
    sequence_metric_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    method_metric_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for sequence_index, sequence in enumerate(sequences):
        embeddings = embedding_provider(sequence)
        similarity_matrix = cosine_similarity_matrix(embeddings)
        validate_similarity_matrix(similarity_matrix)

        shuffled_order, shuffled_matrix = shuffled_similarity_inputs(
            similarity_matrix,
            seed=seed + sequence_index,
        )
        shuffle_permutation = np.array(shuffled_order)
        shuffled_embeddings = embeddings[shuffle_permutation]
        ground_truth_order = list(range(sequence.num_frames))
        ground_truth_order_serialized = _serialize_int_list(ground_truth_order)
        near_mean, far_mean, contrast = temporal_structure_score(
            similarity_matrix,
            near_gap=near_gap,
            far_gap=far_gap,
        )

        directional_evidence = build_directional_evidence(
            frame_embeddings=shuffled_embeddings,
            caption=sequence.caption,
            text_embedding_provider=text_embedding_provider,
        )

        base_continuity_prediction = _continuity_prediction(shuffled_matrix)
        method_predictions = {
            "random": _random_baseline_prediction(
                shuffled_matrix,
                seed=(seed + sequence_index) * 101 + 1,
            ),
            "greedy_nearest_neighbor": _greedy_baseline_prediction(shuffled_matrix),
            "continuity": base_continuity_prediction,
            "continuity_plus_direction": _continuity_plus_direction_prediction(
                shuffled_matrix,
                shuffled_embeddings,
                directional_evidence,
            ),
            "continuity_plus_endpoint": _continuity_plus_endpoint_prediction(
                shuffled_matrix,
                shuffled_embeddings,
                directional_evidence,
            ),
            "continuity_plus_reverse_disambiguation": _continuity_plus_reverse_disambiguation_prediction(
                shuffled_matrix,
                shuffled_embeddings,
                directional_evidence,
                base_continuity_prediction,
            ),
        }

        for method_name, (predicted_shuffled_order, path_score) in method_predictions.items():
            predicted_original_order = [shuffled_order[idx] for idx in predicted_shuffled_order]
            metrics = evaluate_ordering_prediction(predicted_original_order, ground_truth_order)
            row: MetricRow = {
                "sequence_id": sequence.sequence_id,
                "category": sequence.category,
                "difficulty": sequence.difficulty,
                "sequence_type": sequence.sequence_type,
                "method": method_name,
                "num_frames": float(sequence.num_frames),
                "seed": float(seed + sequence_index),
                "shuffle_order": _serialize_int_list(shuffled_order),
                "predicted_shuffled_order": _serialize_int_list(predicted_shuffled_order),
                "predicted_order": _serialize_int_list(predicted_original_order),
                "ground_truth_order": ground_truth_order_serialized,
                "path_score": float(path_score) if path_score is not None else "",
                "temporal_near_mean": near_mean,
                "temporal_far_mean": far_mean,
                "temporal_contrast_score": contrast,
            }
            row.update(metrics)
            results_rows.append(row)

            sequence_key = (
                f"{sequence.sequence_id}|{sequence.category}|{sequence.difficulty}|"
                f"{sequence.sequence_type}|{contrast}"
            )
            for metric_name in METRIC_COLUMNS:
                metric_value = float(metrics[metric_name])
                method_metric_groups[method_name][metric_name].append(metric_value)
                sequence_metric_groups[sequence_key][f"{method_name}__{metric_name}"].append(metric_value)

    summary_by_method = build_method_summary(method_metric_groups)
    summary_by_sequence = build_sequence_summary(sequence_metric_groups)
    return results_rows, summary_by_method, summary_by_sequence


def run_evaluation_cli(args: argparse.Namespace) -> None:
    sequences = load_sequences(args.manifest)
    embedder = OpenCLIPEmbedder(model_name=args.model, pretrained=args.pretrained)
    cache_dir = resolve_project_path(args.cache_dir)
    output_dir = resolve_project_path(args.output_dir)

    def embedding_provider(sequence: Sequence) -> np.ndarray:
        return sequence_embeddings(
            sequence,
            embedder,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            force_recompute=args.force_embeddings,
        )

    def text_embedding_provider(prompts: list[str]) -> np.ndarray:
        return embedder.embed_texts(prompts)

    results_rows, summary_by_method, summary_by_sequence = evaluate_sequences(
        sequences,
        embedding_provider=embedding_provider,
        seed=args.seed,
        near_gap=args.near_gap,
        far_gap=args.far_gap,
        text_embedding_provider=text_embedding_provider,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ordering_results.csv"
    contrast_results_path = output_dir / "ordering_results_with_contrast.csv"
    method_summary_path = output_dir / "ordering_summary_by_method.csv"
    sequence_summary_path = output_dir / "ordering_summary_by_sequence.csv"

    write_csv(results_path, results_rows)
    write_csv(contrast_results_path, results_rows)
    write_csv(method_summary_path, summary_by_method)
    write_csv(sequence_summary_path, summary_by_sequence)

    visual_paths = save_visual_reports(
        output_dir=output_dir,
        results_rows=results_rows,
        summary_by_method=summary_by_method,
        summary_by_sequence=summary_by_sequence,
        sequences=sequences,
    )

    print_method_summary(summary_by_method)
    print_sequence_difficulty_summary(summary_by_sequence)

    root = project_root()
    print(f"Results CSV: {results_path.relative_to(root)}")
    print(f"Results with contrast CSV: {contrast_results_path.relative_to(root)}")
    print(f"Method summary CSV: {method_summary_path.relative_to(root)}")
    print(f"Sequence summary CSV: {sequence_summary_path.relative_to(root)}")
    print("Visual reports:")
    for visual_path in visual_paths:
        print(f"  {visual_path.relative_to(root)}")


def save_results_dataframe(path: str | Path, rows: list[MetricRow]) -> None:
    """Persist evaluation rows to CSV."""
    write_csv(Path(path), rows)


def shuffled_similarity_inputs(
    similarity_matrix: np.ndarray,
    seed: int,
) -> tuple[list[int], np.ndarray]:
    """Return shuffled original indices plus the similarity matrix in shuffled order."""
    validate_similarity_matrix(similarity_matrix)

    shuffled_order = random_permutation(similarity_matrix.shape[0], seed=seed)
    permutation = np.array(shuffled_order)
    shuffled_matrix = similarity_matrix[np.ix_(permutation, permutation)]
    return shuffled_order, shuffled_matrix


def build_method_summary(
    method_metric_groups: dict[str, dict[str, list[float]]],
) -> list[MetricRow]:
    """Aggregate metrics across sequences for each method."""
    rows: list[MetricRow] = []
    for method_name in sorted(method_metric_groups):
        row: MetricRow = {"method": method_name}
        for metric_name in METRIC_COLUMNS:
            values = method_metric_groups[method_name].get(metric_name, [])
            row[metric_name] = mean(values) if values else float("nan")
        rows.append(row)
    return rows


def build_sequence_summary(
    sequence_metric_groups: dict[str, dict[str, list[float]]],
) -> list[MetricRow]:
    """Aggregate metrics into one row per sequence."""
    rows: list[MetricRow] = []
    for sequence_key in sorted(sequence_metric_groups):
        sequence_id, category, difficulty, sequence_type, contrast = sequence_key.split("|", maxsplit=4)
        row: MetricRow = {
            "sequence_id": sequence_id,
            "category": category,
            "difficulty": difficulty,
            "sequence_type": sequence_type,
            "temporal_contrast_score": float(contrast),
        }
        for metric_name, values in sorted(sequence_metric_groups[sequence_key].items()):
            row[metric_name] = mean(values) if values else float("nan")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[MetricRow]) -> None:
    """Write rows to CSV using a stable field order."""
    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_visual_reports(
    output_dir: Path,
    results_rows: list[MetricRow],
    summary_by_method: list[MetricRow],
    summary_by_sequence: list[MetricRow],
    sequences: list[Sequence] | None = None,
) -> list[Path]:
    """Create chart-based summaries of benchmark performance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    method_chart = output_dir / "ordering_method_performance.png"
    contrast_chart = output_dir / "ordering_contrast_vs_pairwise.png"
    heatmap_chart = output_dir / "ordering_sequence_pairwise_heatmap.png"

    _plot_method_comparison(summary_by_method, method_chart)
    _plot_contrast_scatter(results_rows, contrast_chart)
    _plot_sequence_heatmap(summary_by_sequence, heatmap_chart)

    visual_paths = [method_chart, contrast_chart, heatmap_chart]
    if sequences:
        visual_paths.extend(
            save_reordering_storyboards(
                output_dir=output_dir,
                sequences=sequences,
                results_rows=results_rows,
            )
        )

    return visual_paths


def save_reordering_storyboards(
    output_dir: Path,
    sequences: list[Sequence],
    results_rows: list[MetricRow],
) -> list[Path]:
    """Save per-sequence visuals of original, shuffled, and predicted reorder rows."""
    storyboard_dir = output_dir / "storyboards"
    storyboard_dir.mkdir(parents=True, exist_ok=True)

    rows_by_sequence: dict[str, dict[str, MetricRow]] = defaultdict(dict)
    for row in results_rows:
        rows_by_sequence[str(row["sequence_id"])][str(row["method"])] = row

    saved_paths: list[Path] = []
    ordered_methods = [
        "random",
        "greedy_nearest_neighbor",
        "continuity",
        "continuity_plus_direction",
        "continuity_plus_endpoint",
        "continuity_plus_reverse_disambiguation",
    ]

    for sequence in sequences:
        method_rows = rows_by_sequence.get(sequence.sequence_id)
        if not method_rows:
            continue

        available_methods = [method for method in ordered_methods if method in method_rows]
        if not available_methods:
            continue

        shuffle_order = _parse_index_list(str(method_rows[available_methods[0]]["shuffle_order"]))
        n_frames = sequence.num_frames
        if len(shuffle_order) != n_frames:
            continue

        frame_paths = sequence.frames
        shuffled_frames = [frame_paths[idx] for idx in shuffle_order]

        n_rows = 2 + len(available_methods)
        fig, axes = plt.subplots(
            n_rows,
            n_frames,
            figsize=(max(10.0, n_frames * 2.1), max(4.8, n_rows * 1.8)),
        )
        if n_rows == 1:
            axes = np.array([axes])
        if n_frames == 1:
            axes = axes.reshape(n_rows, 1)

        title = (
            f"{sequence.sequence_id} | {sequence.caption}\n"
            f"Category: {sequence.category}  Difficulty: {sequence.difficulty}"
        )
        fig.suptitle(title, fontsize=10)

        _plot_frame_row(
            axes=axes[0],
            frame_paths=frame_paths,
            labels=[f"#{idx + 1}" for idx in range(n_frames)],
            row_title="Original",
        )
        _plot_frame_row(
            axes=axes[1],
            frame_paths=shuffled_frames,
            labels=[f"orig #{idx + 1}" for idx in shuffle_order],
            row_title="Shuffled Input",
        )

        for method_row_index, method in enumerate(available_methods, start=2):
            row = method_rows[method]
            predicted_shuffled_order = _parse_index_list(str(row["predicted_shuffled_order"]))
            if len(predicted_shuffled_order) != n_frames:
                predicted_frames = shuffled_frames
                labels = ["invalid" for _ in range(n_frames)]
            else:
                predicted_frames = [shuffled_frames[idx] for idx in predicted_shuffled_order]
                labels = [f"orig #{shuffle_order[idx] + 1}" for idx in predicted_shuffled_order]

            pairwise = float(row.get("pairwise_order_accuracy", float("nan")))
            exact = float(row.get("exact_match_accuracy", float("nan")))
            row_title = f"Predicted ({method})\npairwise={pairwise:.2f}, exact={exact:.2f}"
            _plot_frame_row(
                axes=axes[method_row_index],
                frame_paths=predicted_frames,
                labels=labels,
                row_title=row_title,
            )

        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        output_path = storyboard_dir / f"{sequence.sequence_id}_pipeline.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def _plot_frame_row(
    axes: np.ndarray,
    frame_paths: list[Path],
    labels: list[str],
    row_title: str,
) -> None:
    for axis, frame_path, label in zip(axes, frame_paths, labels):
        image = mpimg.imread(str(frame_path))
        axis.imshow(image)
        axis.set_title(label, fontsize=8, pad=3)
        axis.axis("off")
    axes[0].set_ylabel(row_title, fontsize=9, rotation=90, labelpad=12, va="center")


def _parse_index_list(indices_text: str) -> list[int]:
    if not indices_text.strip():
        return []
    return [int(part) for part in indices_text.split()]


def _plot_method_comparison(summary_by_method: list[MetricRow], output_path: Path) -> None:
    methods = [str(row["method"]) for row in summary_by_method]
    exact = [float(row["exact_match_accuracy"]) for row in summary_by_method]
    pairwise = [float(row["pairwise_order_accuracy"]) for row in summary_by_method]
    normalized = [float(row["normalized_kendall_agreement"]) for row in summary_by_method]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(x - width, exact, width=width, label="Exact Match")
    ax.bar(x, pairwise, width=width, label="Pairwise")
    ax.bar(x + width, normalized, width=width, label="Normalized Kendall")

    ax.set_xticks(x, labels=methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("ChronoLogic Method Performance")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_contrast_scatter(results_rows: list[MetricRow], output_path: Path) -> None:
    rows_by_method: dict[str, list[MetricRow]] = defaultdict(list)
    for row in results_rows:
        rows_by_method[str(row["method"])].append(row)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method, rows in sorted(rows_by_method.items()):
        contrasts = [float(row["temporal_contrast_score"]) for row in rows]
        pairwise = [float(row["pairwise_order_accuracy"]) for row in rows]
        ax.scatter(contrasts, pairwise, s=58, alpha=0.78, label=method)

    ax.set_xlabel("Temporal Contrast Score")
    ax.set_ylabel("Pairwise Order Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Contrast vs Reconstruction Accuracy")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_sequence_heatmap(summary_by_sequence: list[MetricRow], output_path: Path) -> None:
    sequence_ids = [str(row["sequence_id"]) for row in summary_by_sequence]
    method_names = _collect_method_names(summary_by_sequence)

    matrix = np.zeros((len(sequence_ids), len(method_names)), dtype=np.float32)
    for row_idx, row in enumerate(summary_by_sequence):
        for col_idx, method in enumerate(method_names):
            metric_key = f"{method}__pairwise_order_accuracy"
            matrix[row_idx, col_idx] = float(row.get(metric_key, np.nan))

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(method_names)), labels=method_names)
    ax.set_yticks(np.arange(len(sequence_ids)), labels=sequence_ids)
    ax.set_xlabel("Method")
    ax.set_ylabel("Sequence")
    ax.set_title("Pairwise Accuracy Heatmap")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pairwise Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _collect_method_names(summary_by_sequence: list[MetricRow]) -> list[str]:
    method_names: set[str] = set()
    for row in summary_by_sequence:
        for key in row:
            if key.endswith("__pairwise_order_accuracy"):
                method_names.add(key.split("__", maxsplit=1)[0])
    return sorted(method_names)


def print_method_summary(summary_by_method: list[MetricRow]) -> None:
    """Print aggregate metrics by method."""
    print("Average performance by method:")
    for row in summary_by_method:
        print(
            "  "
            f"{row['method']}: "
            f"exact={float(row['exact_match_accuracy']):.3f}, "
            f"pairwise={float(row['pairwise_order_accuracy']):.3f}, "
            f"kendall_tau={float(row['kendall_tau']):.3f}, "
            f"normalized_inversion={float(row['normalized_kendall_agreement']):.3f}"
        )


def print_sequence_difficulty_summary(summary_by_sequence: list[MetricRow]) -> None:
    """Print a compact sequence-level summary for manual inspection."""
    print("\nSequence difficulty summary:")
    for row in summary_by_sequence:
        greedy_pairwise = row.get("greedy_nearest_neighbor__pairwise_order_accuracy", float("nan"))
        random_pairwise = row.get("random__pairwise_order_accuracy", float("nan"))
        continuity_pairwise = row.get("continuity__pairwise_order_accuracy", float("nan"))
        continuity_direction_pairwise = row.get(
            "continuity_plus_direction__pairwise_order_accuracy",
            float("nan"),
        )
        continuity_endpoint_pairwise = row.get(
            "continuity_plus_endpoint__pairwise_order_accuracy",
            float("nan"),
        )
        continuity_reverse_pairwise = row.get(
            "continuity_plus_reverse_disambiguation__pairwise_order_accuracy",
            float("nan"),
        )
        print(
            "  "
            f"{row['sequence_id']} ({row['difficulty']}, contrast={float(row['temporal_contrast_score']):.3f}): "
            f"greedy_pairwise={float(greedy_pairwise):.3f}, "
            f"continuity_pairwise={float(continuity_pairwise):.3f}, "
            f"continuity_direction_pairwise={float(continuity_direction_pairwise):.3f}, "
            f"continuity_endpoint_pairwise={float(continuity_endpoint_pairwise):.3f}, "
            f"continuity_reverse_pairwise={float(continuity_reverse_pairwise):.3f}, "
            f"random_pairwise={float(random_pairwise):.3f}"
        )


def _random_baseline_prediction(
    similarity_matrix: np.ndarray,
    seed: int,
) -> tuple[list[int], float | None]:
    return random_permutation(similarity_matrix.shape[0], seed=seed), None


def _greedy_baseline_prediction(similarity_matrix: np.ndarray) -> tuple[list[int], float]:
    return best_greedy_path(similarity_matrix)


def _continuity_prediction(similarity_matrix: np.ndarray) -> tuple[list[int], float]:
    return best_continuity_path(similarity_matrix)


def _continuity_plus_direction_prediction(
    similarity_matrix: np.ndarray,
    frame_embeddings: np.ndarray,
    directional_evidence: object,
) -> tuple[list[int], float]:
    return best_continuity_path(
        similarity_matrix,
        direction_weight=CONTINUITY_DIRECTION_WEIGHT,
        frame_embeddings=frame_embeddings,
        directional_evidence=directional_evidence,
    )


def _continuity_plus_endpoint_prediction(
    similarity_matrix: np.ndarray,
    frame_embeddings: np.ndarray,
    directional_evidence: object,
) -> tuple[list[int], float]:
    return best_continuity_path(
        similarity_matrix,
        endpoint_weight=CONTINUITY_ENDPOINT_WEIGHT,
        frame_embeddings=frame_embeddings,
        directional_evidence=directional_evidence,
    )


def _continuity_plus_reverse_disambiguation_prediction(
    similarity_matrix: np.ndarray,
    frame_embeddings: np.ndarray,
    directional_evidence: object,
    base_prediction: tuple[list[int], float] | None = None,
) -> tuple[list[int], float]:
    if base_prediction is None:
        base_prediction = _continuity_prediction(similarity_matrix)

    base_path, base_score = base_prediction
    disambiguated_path, support_score = disambiguate_reversal(
        base_path,
        directional_evidence=directional_evidence,
        frame_embeddings=frame_embeddings,
        direction_weight=REVERSE_DISAMBIGUATION_DIRECTION_WEIGHT,
        endpoint_weight=REVERSE_DISAMBIGUATION_ENDPOINT_WEIGHT,
    )
    if disambiguated_path == base_path:
        return base_path, base_score

    flipped_score = _score_disambiguated_path(similarity_matrix, disambiguated_path)
    return disambiguated_path, flipped_score + support_score


def _score_disambiguated_path(similarity_matrix: np.ndarray, path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    return float(
        np.sum([similarity_matrix[current, nxt] for current, nxt in zip(path, path[1:])])
    )


def _serialize_int_list(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


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
    "save_reordering_storyboards",
    "save_visual_reports",
    "save_results_dataframe",
    "shuffled_similarity_inputs",
    "write_csv",
]
