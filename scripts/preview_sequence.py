"""Backward-compatible preview entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_ordering.preview import choose_sequence, main, parse_preview_args, plot_row, preview, run_preview_cli

__all__ = [
    "choose_sequence",
    "main",
    "parse_preview_args",
    "plot_row",
    "preview",
    "run_preview_cli",
]


if __name__ == "__main__":
    main()
