from pathlib import Path
import sys
import importlib

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))

load_sequences = importlib.import_module("temporal_ordering.data_loader").load_sequences
ManifestValidationError = importlib.import_module(
  "temporal_ordering.exceptions"
).ManifestValidationError


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_sequences_success(tmp_path: Path) -> None:
    data_dir = tmp_path / "Data"
    seq_dir = data_dir / "Sequence_1"
    seq_dir.mkdir(parents=True)
    for idx in range(1, 3):
        (seq_dir / f"{idx:03d}.png").write_bytes(b"png")

    manifest = data_dir / "manifests" / "sequences.json"
    _write_file(
        manifest,
        """
[
  {
    "sequence_id": "sequence_1",
    "category": "x",
    "caption": "y",
    "difficulty": "easy",
    "sequence_type": "procedural",
    "num_frames": 2,
    "frames": ["Data/Sequence_1/001.png", "Data/Sequence_1/002.png"]
  }
]
""".strip(),
    )

    sequences = load_sequences(manifest)
    assert len(sequences) == 1
    assert sequences[0].sequence_id == "sequence_1"
    assert sequences[0].num_frames == 2


def test_load_sequences_invalid_manifest_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "Data" / "manifests" / "sequences.json"
    _write_file(
        manifest,
        """
[
  {
    "sequence_id": "sequence_1",
    "num_frames": 2,
    "frames": ["Data/Sequence_1/001.png"]
  }
]
""".strip(),
    )

    with pytest.raises(ManifestValidationError):
        load_sequences(manifest)
