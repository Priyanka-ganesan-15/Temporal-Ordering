"""Backward-compatible similarity analysis entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_ordering.similarity import (
    cosine_similarity_matrix,
    main,
    parse_similarity_args,
    run_similarity_cli,
    save_heatmap,
    sequence_embeddings,
    temporal_structure_score,
)

__all__ = [
    "cosine_similarity_matrix",
    "main",
    "parse_similarity_args",
    "run_similarity_cli",
    "save_heatmap",
    "sequence_embeddings",
    "temporal_structure_score",
]


if __name__ == "__main__":
    main()
