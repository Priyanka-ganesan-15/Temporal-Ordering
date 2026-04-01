"""Endpoint distinctiveness diagnostics for temporal ordering."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EndpointRow = dict[str, str | int | float]


def compute_endpoint_distinctiveness(
    sequence_id: str,
    embeddings: np.ndarray,
    similarity_matrix: np.ndarray,
) -> list[EndpointRow]:
    """Compute endpoint-related per-frame distinctiveness statistics."""
    n_frames = embeddings.shape[0]
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)

    rows: list[EndpointRow] = []
    for frame_idx in range(n_frames):
        others = [i for i in range(n_frames) if i != frame_idx]
        mean_similarity = float(np.mean(similarity_matrix[frame_idx, others])) if others else 1.0
        rows.append(
            {
                "sequence_id": sequence_id,
                "frame_index": frame_idx,
                "mean_similarity_to_others": mean_similarity,
                "distance_to_centroid": float(distances_to_centroid[frame_idx]),
                "is_true_endpoint": int(frame_idx in (0, n_frames - 1)),
            }
        )
    return rows


def write_endpoint_rows(path: Path, rows: list[EndpointRow]) -> None:
    """Write endpoint distinctiveness rows to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "sequence_id",
                "frame_index",
                "mean_similarity_to_others",
                "distance_to_centroid",
                "is_true_endpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_endpoint_distinctiveness(
    sequence_id: str,
    rows: list[EndpointRow],
    output_path: Path,
) -> None:
    """Plot endpoint distinctiveness bar charts for one sequence."""
    frame_indices = [int(row["frame_index"]) for row in rows]
    mean_similarities = [float(row["mean_similarity_to_others"]) for row in rows]
    centroid_distances = [float(row["distance_to_centroid"]) for row in rows]
    endpoint_flags = [int(row["is_true_endpoint"]) for row in rows]

    colors = ["#2a9d8f" if flag == 1 else "#457b9d" for flag in endpoint_flags]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    axes[0].bar(frame_indices, mean_similarities, color=colors, edgecolor="#333333", linewidth=0.4)
    axes[0].set_ylabel("Mean similarity to others")
    axes[0].set_title("Frame Distinctiveness by Similarity")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(frame_indices, centroid_distances, color=colors, edgecolor="#333333", linewidth=0.4)
    axes[1].set_ylabel("Distance to centroid")
    axes[1].set_xlabel("Frame index")
    axes[1].set_title("Frame Distinctiveness by Centroid Distance")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Endpoint Distinctiveness: {sequence_id}")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "compute_endpoint_distinctiveness",
    "plot_endpoint_distinctiveness",
    "write_endpoint_rows",
]
