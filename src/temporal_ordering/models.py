"""Domain models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sequence:
    sequence_id: str
    category: str
    caption: str
    difficulty: str
    sequence_type: str
    num_frames: int
    frames: list[Path]
