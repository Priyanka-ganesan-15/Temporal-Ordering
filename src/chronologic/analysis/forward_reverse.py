"""Forward-vs-reverse score diagnostics for continuity-based ordering."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chronologic.ordering.continuity import score_permutation_with_continuity

ForwardReverseRow = dict[str, str | float]


def compute_forward_reverse_scores(
    sequence_id: str,
    similarity_matrix: np.ndarray,
    adjacency_weight: float = 1.0,
    continuity_weight: float = 1.0,
) -> ForwardReverseRow:
    """Compute continuity objective for true order and reversed order."""
    n_frames = similarity_matrix.shape[0]
    true_order = list(range(n_frames))
    reverse_order = list(reversed(true_order))

    forward_score = score_permutation_with_continuity(
        similarity_matrix,
        true_order,
        adjacency_weight=adjacency_weight,
        continuity_weight=continuity_weight,
    )
    reverse_score = score_permutation_with_continuity(
        similarity_matrix,
        reverse_order,
        adjacency_weight=adjacency_weight,
        continuity_weight=continuity_weight,
    )
    score_gap = forward_score - reverse_score

    winner = "forward" if score_gap > 0.0 else "reverse" if score_gap < 0.0 else "tie"
    return {
        "sequence_id": sequence_id,
        "forward_score": float(forward_score),
        "reverse_score": float(reverse_score),
        "forward_minus_reverse": float(score_gap),
        "winner": winner,
    }


def write_forward_reverse_scores(path: Path, rows: list[ForwardReverseRow]) -> None:
    """Write forward/reverse comparison rows to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "sequence_id",
                "forward_score",
                "reverse_score",
                "forward_minus_reverse",
                "winner",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_forward_reverse_gap(rows: list[ForwardReverseRow], output_path: Path) -> None:
    """Render bar chart of forward-minus-reverse score gaps per sequence."""
    if not rows:
        raise ValueError("rows must not be empty")

    sequence_ids = [str(row["sequence_id"]) for row in rows]
    gaps = np.array([float(row["forward_minus_reverse"]) for row in rows], dtype=np.float64)
    colors = ["#2a9d8f" if gap >= 0.0 else "#e76f51" for gap in gaps]

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 1.1), 4.8))
    ax.bar(sequence_ids, gaps, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axhline(0.0, color="#1f1f1f", linewidth=1.0)
    ax.set_ylabel("Forward - Reverse Continuity Score")
    ax.set_xlabel("Sequence")
    ax.set_title("Forward vs Reverse Continuity Score Gap")
    ax.grid(axis="y", alpha=0.25)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "compute_forward_reverse_scores",
    "plot_forward_reverse_gap",
    "write_forward_reverse_scores",
]
