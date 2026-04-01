"""Backward-compatible loader entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

# Support running without editable install.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_ordering.config import DEFAULT_MANIFEST
from temporal_ordering.data_loader import load_sequences, print_summary
from temporal_ordering.models import Sequence


__all__ = ["Sequence", "load_sequences", "print_summary"]


def main() -> None:
    manifest = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MANIFEST
    try:
        sequences = load_sequences(manifest)
        print_summary(sequences)
        print("All sequences valid. Sample:")
        print(f"  sequence_id : {sequences[0].sequence_id}")
        print(f"  caption     : {sequences[0].caption}")
        print(f"  frames[0]   : {sequences[0].frames[0]}")
        print(f"  frames[-1]  : {sequences[0].frames[-1]}")
    except Exception as exc:  # Keep broad for compatibility with old script behavior.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
