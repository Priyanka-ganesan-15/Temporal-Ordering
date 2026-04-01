"""Similarity matrix analysis for temporal ordering."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from temporal_ordering.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRETRAINED,
    project_root,
    resolve_project_path,
)
from temporal_ordering.data_loader import load_sequences
from temporal_ordering.embedding import OpenCLIPEmbedder
from temporal_ordering.models import Sequence
from temporal_ordering.ordering.nearest_neighbor import (
    greedy_nearest_neighbor_ordering,
    greedy_path_from_start,
    path_adjacency_score,
    validate_similarity_matrix,
)
from temporal_ordering.ordering.random_baseline import random_ordering_baseline
from temporal_ordering.utils import get_sequence_or_raise


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix of shape (n, n)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def temporal_structure_score(
    similarity_matrix: np.ndarray,
    near_gap: int = 1,
    far_gap: int = 4,
) -> tuple[float, float, float]:
    """Return near mean, far mean, and contrast (near - far)."""
    n_frames = similarity_matrix.shape[0]
    idx = np.arange(n_frames)
    dist = np.abs(idx[:, None] - idx[None, :])

    near_mask = (dist > 0) & (dist <= near_gap)
    far_mask = dist >= far_gap

    near_mean = float(similarity_matrix[near_mask].mean()) if np.any(near_mask) else float("nan")
    far_mean = float(similarity_matrix[far_mask].mean()) if np.any(far_mask) else float("nan")
    return near_mean, far_mean, near_mean - far_mean


def save_heatmap(similarity_matrix: np.ndarray, sequence: Sequence, output_path: Path) -> None:
    """Save cosine similarity heatmap for one sequence."""
    n = similarity_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(similarity_matrix, vmin=-1.0, vmax=1.0, cmap="viridis")

    ticks = np.arange(n)
    labels = [str(i + 1) for i in range(n)]
    ax.set_xticks(ticks, labels=labels)
    ax.set_yticks(ticks, labels=labels)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.set_title(f"{sequence.sequence_id} cosine similarity")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cosine similarity")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def sequence_embeddings(
    sequence: Sequence,
    embedder: OpenCLIPEmbedder,
    cache_dir: Path,
    batch_size: int,
    force_recompute: bool,
) -> np.ndarray:
    """Load cached sequence embeddings or compute and cache them."""
    cache_path = cache_dir / f"{sequence.sequence_id}.npy"
    if cache_path.exists() and not force_recompute:
        return np.load(cache_path)
    return embedder.embed_paths(
        sequence.frames,
        batch_size=batch_size,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )


def parse_similarity_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-sequence similarity heatmaps")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence", default=None, help="Optional single sequence_id")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--near-gap", type=int, default=1)
    parser.add_argument("--far-gap", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    return parser.parse_args()


def run_similarity_cli(args: argparse.Namespace) -> None:
    sequences = load_sequences(args.manifest)
    if args.sequence:
        sequences = [get_sequence_or_raise(sequences, args.sequence)]

    embedder = OpenCLIPEmbedder(model_name=args.model, pretrained=args.pretrained)
    cache_dir = resolve_project_path(args.cache_dir)
    output_dir = resolve_project_path(args.output_dir)
    root = project_root()

    summary_rows: list[dict[str, str | float]] = []
    for sequence in sequences:
        embeddings = sequence_embeddings(
            sequence,
            embedder,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            force_recompute=args.force_embeddings,
        )
        similarity = cosine_similarity_matrix(embeddings)
        heatmap_path = output_dir / f"{sequence.sequence_id}_heatmap.png"
        save_heatmap(similarity, sequence, heatmap_path)

        near_mean, far_mean, contrast = temporal_structure_score(
            similarity,
            near_gap=args.near_gap,
            far_gap=args.far_gap,
        )
        summary_rows.append(
            {
                "sequence_id": sequence.sequence_id,
                "difficulty": sequence.difficulty,
                "sequence_type": sequence.sequence_type,
                "near_mean": near_mean,
                "far_mean": far_mean,
                "contrast": contrast,
                "heatmap": str(heatmap_path.relative_to(root)),
            }
        )

    summary_rows.sort(key=lambda row: float(row["contrast"]), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "sequence_id",
                "difficulty",
                "sequence_type",
                "near_mean",
                "far_mean",
                "contrast",
                "heatmap",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Saved heatmaps and summary:")
    for row in summary_rows:
        print(
            f"  {row['sequence_id']}: contrast={float(row['contrast']):.4f} "
            f"(near={float(row['near_mean']):.4f}, far={float(row['far_mean']):.4f})"
        )
    print(f"Summary CSV: {summary_csv.relative_to(root)}")


def main() -> None:
    run_similarity_cli(parse_similarity_args())


if __name__ == "__main__":
    main()
