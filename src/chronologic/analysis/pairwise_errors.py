"""Pairwise ordering error diagnostics and heatmaps."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_pairwise_error_matrix(predicted_order: list[int], true_order: list[int]) -> np.ndarray:
    """Return matrix where upper triangle is 1 if pair order is correct else 0."""
    if len(predicted_order) != len(true_order):
        raise ValueError("predicted_order and true_order must have the same length")

    n_frames = len(true_order)
    predicted_positions = {frame: idx for idx, frame in enumerate(predicted_order)}

    matrix = np.full((n_frames, n_frames), np.nan, dtype=np.float64)
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            matrix[i, j] = 1.0 if predicted_positions[i] < predicted_positions[j] else 0.0
    return matrix


def pairwise_error_rows(
    sequence_id: str,
    method: str,
    predicted_order: list[int],
    true_order: list[int],
) -> list[dict[str, str | int | float]]:
    """Emit pairwise correctness rows for CSV export."""
    matrix = compute_pairwise_error_matrix(predicted_order, true_order)
    rows: list[dict[str, str | int | float]] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "method": method,
                    "frame_i": i,
                    "frame_j": j,
                    "is_correct": int(matrix[i, j]),
                }
            )
    return rows


def write_pairwise_error_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write pairwise correctness rows to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["sequence_id", "method", "frame_i", "frame_j", "is_correct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_pairwise_error_matrix(
    sequence_id: str,
    method: str,
    pairwise_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Render pairwise ordering correctness heatmap."""
    n_frames = pairwise_matrix.shape[0]
    lower = np.tril_indices(n_frames)
    display_matrix = pairwise_matrix.copy()
    display_matrix[lower] = 0.5

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.imshow(display_matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_title(f"Pairwise Ordering Correctness\n{sequence_id} | {method}")
    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")
    ax.set_xticks(np.arange(n_frames))
    ax.set_yticks(np.arange(n_frames))
    ax.grid(alpha=0.15)

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            value = int(pairwise_matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="#111111")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("1 = correct, 0 = incorrect")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "compute_pairwise_error_matrix",
    "pairwise_error_rows",
    "plot_pairwise_error_matrix",
    "write_pairwise_error_rows",
]
