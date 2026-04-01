"""Load and validate sequences from manifest files."""

from __future__ import annotations

import json
from pathlib import Path

from temporal_ordering.config import DEFAULT_MANIFEST, resolve_project_path
from temporal_ordering.exceptions import ManifestValidationError
from temporal_ordering.models import Sequence


def load_sequences(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    base_dir: str | Path | None = None,
) -> list[Sequence]:
    """Load and validate all sequences from a JSON manifest."""
    manifest = resolve_project_path(manifest_path).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    root = Path(base_dir).resolve() if base_dir else manifest.parent.parent.parent

    with open(manifest, encoding="utf-8") as f:
        raw = json.load(f)

    sequences: list[Sequence] = []
    errors: list[str] = []

    for entry in raw:
        seq_id = entry.get("sequence_id", "<unknown>")
        declared = entry.get("num_frames", -1)
        frame_paths_raw = entry.get("frames", [])

        if declared != len(frame_paths_raw):
            errors.append(
                f"[{seq_id}] num_frames={declared} but {len(frame_paths_raw)} paths listed"
            )

        resolved_frames: list[Path] = []
        for rel_path in frame_paths_raw:
            abs_path = root / rel_path
            if not abs_path.exists():
                errors.append(f"[{seq_id}] Missing frame: {abs_path}")
            resolved_frames.append(abs_path)

        sequences.append(
            Sequence(
                sequence_id=seq_id,
                category=entry.get("category", ""),
                caption=entry.get("caption", ""),
                difficulty=entry.get("difficulty", ""),
                sequence_type=entry.get("sequence_type", ""),
                num_frames=declared,
                frames=resolved_frames,
            )
        )

    if errors:
        msg = "\n".join(["Validation errors found:"] + [f"  - {e}" for e in errors])
        raise ManifestValidationError(msg)

    return sequences


def print_summary(sequences: list[Sequence]) -> None:
    """Pretty-print a compact sequence table."""
    print(f"\nLoaded {len(sequences)} sequence(s)\n")
    print(f"{'ID':<15} {'Category':<18} {'Difficulty':<10} {'Type':<14} {'Frames'}")
    print("-" * 70)
    for seq in sequences:
        print(
            f"{seq.sequence_id:<15} {seq.category:<18} "
            f"{seq.difficulty:<10} {seq.sequence_type:<14} {seq.num_frames}"
        )
    print()
