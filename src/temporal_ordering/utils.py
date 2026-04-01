"""Shared utility helpers."""

from __future__ import annotations

from pathlib import Path

from temporal_ordering.exceptions import SequenceNotFoundError
from temporal_ordering.models import Sequence


def flatten_frames(sequences: list[Sequence]) -> list[Path]:
    """Return all frame paths in sequence order."""
    return [frame for seq in sequences for frame in seq.frames]


def get_sequence_or_raise(sequences: list[Sequence], sequence_id: str) -> Sequence:
    """Fetch one sequence by id or raise a descriptive error."""
    for seq in sequences:
        if seq.sequence_id == sequence_id:
            return seq
    available = ", ".join(s.sequence_id for s in sequences)
    raise SequenceNotFoundError(
        f"Unknown sequence_id '{sequence_id}'. Available: {available}"
    )
