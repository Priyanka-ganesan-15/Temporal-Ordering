"""Alignment diagnostics between predicted and ground-truth frame positions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prediction_alignment_points(predicted_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return x=true index and y=predicted position for each frame index."""
    n_frames = len(predicted_order)
    true_indices = np.arange(n_frames, dtype=np.int32)
    predicted_positions = np.zeros(n_frames, dtype=np.int32)
    for predicted_position, frame_index in enumerate(predicted_order):
        predicted_positions[frame_index] = predicted_position
    return true_indices, predicted_positions


def plot_order_alignment(
    sequence_id: str,
    method_predictions: dict[str, list[int]],
    output_path: Path,
) -> None:
    """Plot predicted position vs true position with diagonal and anti-diagonal."""
    if not method_predictions:
        raise ValueError("method_predictions must not be empty")

    methods = list(method_predictions.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(max(5.0 * n_methods, 7.5), 4.8),
        squeeze=False,
    )

    for axis, method in zip(axes[0], methods):
        predicted = method_predictions[method]
        true_idx, pred_pos = prediction_alignment_points(predicted)
        n_frames = len(predicted)

        axis.scatter(true_idx, pred_pos, s=60, color="#1f77b4", alpha=0.85)
        axis.plot([0, n_frames - 1], [0, n_frames - 1], "--", color="#2a9d8f", label="diagonal")
        axis.plot(
            [0, n_frames - 1],
            [n_frames - 1, 0],
            "--",
            color="#e76f51",
            label="anti-diagonal",
        )
        axis.set_title(method)
        axis.set_xlabel("True frame index")
        axis.set_ylabel("Predicted position")
        axis.set_xlim(-0.4, n_frames - 0.6)
        axis.set_ylim(-0.4, n_frames - 0.6)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Predicted vs True Position Alignment: {sequence_id}")
    fig.tight_layout(rect=[0, 0.0, 1, 0.9])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_order_alignment",
    "prediction_alignment_points",
]
