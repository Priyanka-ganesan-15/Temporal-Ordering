"""Centralized project defaults and path helpers."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MANIFEST = "Data/manifests/sequences.json"
DEFAULT_CACHE_DIR = "Data/embeddings/openclip"
DEFAULT_OUTPUT_DIR = "Data/analysis/similarity"
DEFAULT_ORDERING_OUTPUT_DIR = "Data/analysis/ordering"

DEFAULT_MODEL = "RN50"
DEFAULT_PRETRAINED = "openai"
DEFAULT_BATCH_SIZE = 16
DEFAULT_RANDOM_SEED = 42


def project_root() -> Path:
    """Return project root based on package location."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_like: str | Path) -> Path:
    """Resolve relative paths against project root, preserve absolute paths."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path
