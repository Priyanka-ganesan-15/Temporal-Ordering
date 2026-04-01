"""Trajectory and profile diagnostics for sequence embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_adjacency_similarity_profile(
    embeddings: np.ndarray,
    order: list[int],
) -> np.ndarray:
    """Compute cosine similarity for consecutive frames in the provided order."""
    if len(order) < 2:
        return np.array([], dtype=np.float64)

    ordered = _normalize_rows(np.asarray(embeddings[order], dtype=np.float64))
    similarities = np.sum(ordered[:-1] * ordered[1:], axis=1)
    return similarities.astype(np.float64)


def compute_second_order_jump_profile(
    embeddings: np.ndarray,
    order: list[int],
) -> np.ndarray:
    """Compute norm of consecutive delta differences along the order."""
    if len(order) < 3:
        return np.array([], dtype=np.float64)

    ordered = np.asarray(embeddings[order], dtype=np.float64)
    deltas = np.diff(ordered, axis=0)
    if deltas.shape[0] < 2:
        return np.array([], dtype=np.float64)
    second_order = np.diff(deltas, axis=0)
    return np.linalg.norm(second_order, axis=1)


def pca_project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using PCA via SVD."""
    array = np.asarray(embeddings, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)

    if centered.shape[1] == 1:
        return np.concatenate([centered, np.zeros_like(centered)], axis=1)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = np.concatenate([projected, np.zeros((projected.shape[0], 1))], axis=1)
    return projected


def plot_sequence_profiles(
    sequence_id: str,
    adjacency_profile: np.ndarray,
    second_order_profile: np.ndarray,
    output_path: Path,
) -> None:
    """Plot adjacency cosine profile and second-order jump profile."""
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4))

    axes[0].plot(np.arange(adjacency_profile.size), adjacency_profile, marker="o", color="#264653")
    axes[0].set_title("Adjacency Similarity Profile")
    axes[0].set_xlabel("Transition index t")
    axes[0].set_ylabel("cos(frame_t, frame_t+1)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        np.arange(second_order_profile.size),
        second_order_profile,
        marker="o",
        color="#e76f51",
    )
    axes[1].set_title("Second-Order Jump Profile")
    axes[1].set_xlabel("Transition index t")
    axes[1].set_ylabel("||delta_{t+1} - delta_t||")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Embedding Profiles: {sequence_id}")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_embedding_trajectories(
    sequence_id: str,
    projected_2d: np.ndarray,
    true_order: list[int],
    method_predictions: dict[str, list[int]],
    output_path: Path,
) -> None:
    """Plot 2D embedding trajectory in true order and predicted orders."""
    methods = ["true"] + list(method_predictions.keys())
    n_panels = len(methods)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(5.0 * n_panels, 8.5), 4.8),
        squeeze=False,
    )

    for axis, method in zip(axes[0], methods):
        order = true_order if method == "true" else method_predictions[method]
        ordered_points = projected_2d[order]
        axis.plot(ordered_points[:, 0], ordered_points[:, 1], "-o", linewidth=1.6, markersize=4)
        for idx, frame_index in enumerate(order):
            axis.text(
                ordered_points[idx, 0],
                ordered_points[idx, 1],
                str(frame_index),
                fontsize=8,
                ha="left",
                va="bottom",
            )
        axis.set_title(method)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.25)

    fig.suptitle(f"Embedding Trajectories: {sequence_id}")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-8)
    return vectors / safe_norms


__all__ = [
    "compute_adjacency_similarity_profile",
    "compute_second_order_jump_profile",
    "pca_project_2d",
    "plot_embedding_trajectories",
    "plot_sequence_profiles",
]
