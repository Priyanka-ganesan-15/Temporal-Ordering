"""Sequence preview utilities and CLI."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from temporal_ordering.config import DEFAULT_MANIFEST, resolve_project_path
from temporal_ordering.data_loader import load_sequences
from temporal_ordering.models import Sequence
from temporal_ordering.utils import get_sequence_or_raise


def plot_row(axes, frames: list[Path], labels: list[str], row_title: str) -> None:
    """Plot one frame row on provided axes."""
    for ax, frame_path, label in zip(axes, frames, labels):
        image = mpimg.imread(str(frame_path))
        ax.imshow(image)
        ax.set_title(label, fontsize=8, pad=3)
        ax.axis("off")
    axes[0].set_ylabel(row_title, fontsize=10, rotation=90, labelpad=8, va="center")


def preview(sequence: Sequence) -> None:
    """Display ordered and shuffled views for one sequence."""
    n = sequence.num_frames
    ordered = sequence.frames

    shuffled = ordered.copy()
    while shuffled == ordered:
        random.shuffle(shuffled)

    fig = plt.figure(figsize=(n * 2.8, 6.5))
    fig.suptitle(
        f"{sequence.sequence_id}  |  {sequence.caption}\n"
        f"Category: {sequence.category}   "
        f"Difficulty: {sequence.difficulty}   Type: {sequence.sequence_type}",
        fontsize=11,
    )

    grid = gridspec.GridSpec(2, n, figure=fig, hspace=0.4, wspace=0.05)
    ordered_axes = [fig.add_subplot(grid[0, i]) for i in range(n)]
    shuffled_axes = [fig.add_subplot(grid[1, i]) for i in range(n)]

    plot_row(ordered_axes, ordered, [f"#{i+1}" for i in range(n)], "Ordered")
    shuffled_positions = [ordered.index(frame) + 1 for frame in shuffled]
    plot_row(shuffled_axes, shuffled, [f"(orig #{p})" for p in shuffled_positions], "Shuffled")

    plt.tight_layout()
    plt.show()


def parse_preview_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a sequence in order and shuffled")
    parser.add_argument("--sequence", "-s", default=None, help="sequence_id to preview")
    parser.add_argument("--manifest", "-m", default=DEFAULT_MANIFEST, help="path to sequences.json")
    parser.add_argument("--seed", type=int, default=None, help="random seed for shuffle")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive picker when --sequence is omitted",
    )
    return parser.parse_args()


def choose_sequence(sequences: list[Sequence], sequence_id: str | None, no_interactive: bool) -> Sequence:
    """Choose sequence by id or interactive prompt."""
    if sequence_id:
        return get_sequence_or_raise(sequences, sequence_id)

    if no_interactive:
        raise ValueError("Pass --sequence when --no-interactive is used")

    print("\nAvailable sequences:")
    for idx, seq in enumerate(sequences, start=1):
        print(f"  [{idx}] {seq.sequence_id} - {seq.caption} ({seq.difficulty}, {seq.sequence_type})")

    while True:
        choice = input("\nEnter sequence number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(sequences):
            return sequences[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(sequences)}")


def run_preview_cli(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    manifest = resolve_project_path(args.manifest)
    sequences = load_sequences(manifest)
    sequence = choose_sequence(sequences, args.sequence, args.no_interactive)

    print(f"\nPreviewing '{sequence.sequence_id}' ({sequence.num_frames} frames)...")
    preview(sequence)


def main() -> None:
    run_preview_cli(parse_preview_args())


if __name__ == "__main__":
    main()
