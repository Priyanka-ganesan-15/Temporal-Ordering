"""Backward-compatible embedder entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_ordering.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_PRETRAINED,
)
from temporal_ordering.embedding import OpenCLIPEmbedder, main, parse_embedder_args, run_embedder_cli
from temporal_ordering.utils import flatten_frames, get_sequence_or_raise

# Backward-compatible helper alias.
get_sequence = get_sequence_or_raise

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MANIFEST",
    "DEFAULT_MODEL",
    "DEFAULT_PRETRAINED",
    "OpenCLIPEmbedder",
    "flatten_frames",
    "get_sequence",
    "parse_embedder_args",
    "run_embedder_cli",
]


if __name__ == "__main__":
    main()
