"""Text-direction utilities for temporal ordering diagnostics and scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from temporal_ordering.embedding import OpenCLIPEmbedder

TextEmbedder = Callable[[list[str]], np.ndarray]


def build_temporal_prompts(caption: str) -> dict[str, str]:
    """Build start/middle/end prompt variants for a sequence caption."""
    caption_text = caption.strip()
    if not caption_text:
        raise ValueError("caption must not be empty")

    return {
        "start": f"start of {caption_text}",
        "middle": f"middle of {caption_text}",
        "end": f"end of {caption_text}",
    }


def embed_temporal_prompts(
    caption: str,
    *,
    embed_texts: TextEmbedder | None = None,
    embedder: OpenCLIPEmbedder | None = None,
) -> dict[str, np.ndarray]:
    """Embed temporal prompts using an existing text-embedding provider.

    Exactly one provider must be supplied:
    - embed_texts: callable that accepts list[str] and returns (n_prompts, dim)
    - embedder: existing OpenCLIPEmbedder instance exposing embed_texts
    """
    provider = _resolve_text_provider(embed_texts=embed_texts, embedder=embedder)
    prompts = build_temporal_prompts(caption)
    ordered_keys = ["start", "middle", "end"]

    embeddings = np.asarray(provider([prompts[key] for key in ordered_keys]), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ordered_keys):
        raise ValueError("text embedder must return shape (3, embedding_dim)")

    return {key: embeddings[idx] for idx, key in enumerate(ordered_keys)}


def compute_frame_text_similarity(
    frame_embeddings: np.ndarray,
    text_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute cosine similarity between each frame and each temporal text vector."""
    frame_array = np.asarray(frame_embeddings, dtype=np.float64)
    if frame_array.ndim != 2 or frame_array.shape[0] == 0:
        raise ValueError("frame_embeddings must have shape (n_frames, embedding_dim)")

    required_keys = ("start", "middle", "end")
    missing = [key for key in required_keys if key not in text_embeddings]
    if missing:
        raise ValueError(f"text_embeddings missing keys: {missing}")

    normalized_frames = _normalize_rows(frame_array)
    similarities: dict[str, np.ndarray] = {}
    for key in required_keys:
        text_vector = np.asarray(text_embeddings[key], dtype=np.float64)
        if text_vector.ndim != 1:
            raise ValueError(f"text_embeddings['{key}'] must be a 1D vector")
        if text_vector.shape[0] != normalized_frames.shape[1]:
            raise ValueError(
                f"text_embeddings['{key}'] dim {text_vector.shape[0]} does not match "
                f"frame embedding dim {normalized_frames.shape[1]}"
            )

        normalized_text = _normalize_vector(text_vector)
        similarities[key] = (normalized_frames @ normalized_text).astype(np.float32)

    return similarities


def temporal_direction_score(
    path: list[int],
    frame_to_prompt_similarity: dict[str, np.ndarray],
) -> float:
    """Score a path using position-aware start/middle/end prompt alignment.

    The scoring is intentionally simple and ablation-friendly:
    - early positions get higher start weight
    - middle positions get higher middle weight
    - late positions get higher end weight
    """
    if not path:
        raise ValueError("path must not be empty")

    required_keys = ("start", "middle", "end")
    missing = [key for key in required_keys if key not in frame_to_prompt_similarity]
    if missing:
        raise ValueError(f"frame_to_prompt_similarity missing keys: {missing}")

    n_frames = len(path)
    for key in required_keys:
        values = np.asarray(frame_to_prompt_similarity[key], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"frame_to_prompt_similarity['{key}'] must be a 1D array")
        if values.shape[0] < n_frames:
            raise ValueError(
                f"frame_to_prompt_similarity['{key}'] length {values.shape[0]} is "
                f"smaller than required path index space {n_frames}"
            )

    score_total = 0.0
    for pos, frame_idx in enumerate(path):
        if frame_idx < 0 or frame_idx >= n_frames:
            raise ValueError(f"path contains out-of-range frame index: {frame_idx}")

        if n_frames == 1:
            position = 0.5
        else:
            position = pos / (n_frames - 1)

        start_weight = 1.0 - position
        end_weight = position
        middle_weight = max(0.0, 1.0 - 2.0 * abs(position - 0.5))

        combined = (
            start_weight * float(frame_to_prompt_similarity["start"][frame_idx])
            + middle_weight * float(frame_to_prompt_similarity["middle"][frame_idx])
            + end_weight * float(frame_to_prompt_similarity["end"][frame_idx])
        )
        weight_sum = start_weight + middle_weight + end_weight
        score_total += combined / max(weight_sum, 1e-8)

    return score_total / n_frames


def _resolve_text_provider(
    *,
    embed_texts: TextEmbedder | None,
    embedder: OpenCLIPEmbedder | None,
) -> TextEmbedder:
    if embed_texts is not None and embedder is not None:
        raise ValueError("Provide only one of embed_texts or embedder")
    if embed_texts is not None:
        return embed_texts
    if embedder is not None:
        return embedder.embed_texts
    raise ValueError("Either embed_texts or embedder must be provided")


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


__all__ = [
    "build_temporal_prompts",
    "compute_frame_text_similarity",
    "embed_temporal_prompts",
    "temporal_direction_score",
]
