"""Comprehensive experiment runner for ChronoLogic temporal-ordering research.

Experiment 1 — All Methods Comparison
    Runs every implemented method (random, greedy_nn, continuity, spectral,
    insertion_sort) across all 7 sequences with the canonical shuffle seed
    (42 + sequence_index).  Writes per-sequence rows and a per-method summary.

Experiment 2 — Multi-Seed Robustness
    Evaluates greedy_nn and continuity under 15 different base seeds to
    measure how stable each method is across different frame orderings at
    input.

Experiment 3 — Continuity Weight Sweep
    Sweeps adjacency_weight in {0.5, 0.75, 1.0, 1.25, 1.5} ×
    continuity_weight in {0.5, 0.75, 1.0, 1.25, 1.5} (25 combinations) on
    all sequences.  Useful for finding the optimal weight regime.

Outputs
-------
All CSV files are written to  Data/analysis/experiments/.

Usage
-----
    python scripts/run_experiments.py                  # run all experiments
    python scripts/run_experiments.py --exp 1          # single experiment
    python scripts/run_experiments.py --exp 1 2        # subset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronologic.evaluation.metrics import evaluate_ordering_prediction
from chronologic.ordering.continuity import best_continuity_path
from chronologic.ordering.insertion_sort import best_insertion_sort_ordering
from chronologic.ordering.nearest_neighbor import (
    adjacency_path_score,
    best_greedy_path,
)
from chronologic.ordering.random_baseline import random_permutation
from chronologic.ordering.spectral import spectral_fiedler_ordering
from temporal_ordering.similarity import cosine_similarity_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_PATH = PROJECT_ROOT / "Data" / "manifests" / "sequences.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "Data" / "embeddings" / "openclip"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "analysis" / "experiments"

MULTI_SEED_BASE_SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

WEIGHT_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]

METRIC_KEYS = [
    "exact_match_accuracy",
    "pairwise_order_accuracy",
    "normalized_kendall_agreement",
    "kendall_tau",
    "inversion_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def load_embeddings(sequence_id: str) -> np.ndarray:
    """Return cached (n_frames, embed_dim) embedding array."""
    path = EMBEDDINGS_DIR / f"{sequence_id}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Run the embedder first: python embedder.py"
        )
    return np.load(path)


def shuffled_ground_truth(n: int, seed: int) -> list[int]:
    """Ground truth is the identity permutation; the *shuffled* input is what
    the methods receive.  We use the seed to reproduce the same shuffle that
    evaluate_ordering.py applies.

    Returns the *true* ordering (0..n-1) and the *shuffled* indices so we can
    reconstruct the similarity matrix in shuffled frame order.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n).tolist()
    return shuffled  # caller uses this as the index into embeddings


def build_similarity_matrix(embeddings: np.ndarray, shuffled_indices: list[int]) -> np.ndarray:
    """Build cosine similarity matrix for frames in shuffled order."""
    shuffled_embeddings = embeddings[shuffled_indices]
    return cosine_similarity_matrix(shuffled_embeddings)


def true_order_in_shuffled_space(shuffled_indices: list[int]) -> list[int]:
    """For evaluation: what is the correct ordering of the *shuffled* frames?

    If shuffled_indices = [3, 1, 0, 2], the correct order in shuffled space is
    [2, 1, 3, 0] — i.e., the position of original frame k in the shuffled list.
    """
    n = len(shuffled_indices)
    ground_truth = [0] * n
    for shuffled_pos, orig_frame in enumerate(shuffled_indices):
        ground_truth[orig_frame] = shuffled_pos
    return ground_truth


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path.relative_to(PROJECT_ROOT)}")


def summarise_by_method(rows: list[dict], method_col: str = "method") -> list[dict]:
    """Aggregate metric means across all sequences for each method."""
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row[method_col]].append(row)

    summary = []
    for method, method_rows in sorted(buckets.items()):
        entry: dict[str, Any] = {method_col: method, "n_sequences": len(method_rows)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in method_rows if key in r]
            entry[f"mean_{key}"] = round(float(np.mean(vals)), 4) if vals else None
            entry[f"std_{key}"] = round(float(np.std(vals)), 4) if vals else None
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Experiment 1: All methods comparison
# ---------------------------------------------------------------------------

def run_experiment_1(sequences: list[dict]) -> None:
    print("\n=== Experiment 1: All Methods Comparison ===")

    rows: list[dict] = []

    for seq_idx, seq in enumerate(sequences):
        seq_id = seq["sequence_id"]
        seed = 42 + seq_idx
        print(f"  [{seq_id}] seed={seed}", end="")

        embeddings = load_embeddings(seq_id)
        n = embeddings.shape[0]
        shuffled = shuffled_ground_truth(n, seed)
        sim = build_similarity_matrix(embeddings, shuffled)
        ground_truth = true_order_in_shuffled_space(shuffled)

        methods: dict[str, tuple[list[int], float]] = {
            "random": (random_permutation(n, seed), 0.0),
            "greedy_nn": best_greedy_path(sim),
            "continuity": best_continuity_path(sim),
            "spectral": spectral_fiedler_ordering(sim),
            "insertion_sort": best_insertion_sort_ordering(sim),
        }
        # Fix random path_score
        rnd_ordering, _ = methods["random"]
        methods["random"] = (rnd_ordering, adjacency_path_score(rnd_ordering, sim))

        print(f"  methods={list(methods.keys())}")

        for method_name, (pred, _score) in methods.items():
            metrics = evaluate_ordering_prediction(pred, ground_truth)
            row: dict[str, Any] = {
                "sequence_id": seq_id,
                "category": seq.get("category", ""),
                "method": method_name,
                "seed": seed,
            }
            row.update(metrics)
            rows.append(row)

    fieldnames = ["sequence_id", "category", "method", "seed"] + METRIC_KEYS
    write_csv(OUTPUT_DIR / "all_methods_results.csv", rows, fieldnames)

    summary = summarise_by_method(rows)
    summary_fields = ["method", "n_sequences"] + [
        f"{prefix}_{key}" for key in METRIC_KEYS for prefix in ("mean", "std")
    ]
    write_csv(OUTPUT_DIR / "all_methods_summary.csv", summary, summary_fields)


# ---------------------------------------------------------------------------
# Experiment 2: Multi-seed robustness
# ---------------------------------------------------------------------------

def run_experiment_2(sequences: list[dict]) -> None:
    print("\n=== Experiment 2: Multi-Seed Robustness ===")

    rows: list[dict] = []

    for base_seed in MULTI_SEED_BASE_SEEDS:
        print(f"  base_seed={base_seed}")
        for seq_idx, seq in enumerate(sequences):
            seq_id = seq["sequence_id"]
            seed = base_seed + seq_idx

            embeddings = load_embeddings(seq_id)
            n = embeddings.shape[0]
            shuffled = shuffled_ground_truth(n, seed)
            sim = build_similarity_matrix(embeddings, shuffled)
            ground_truth = true_order_in_shuffled_space(shuffled)

            methods: dict[str, tuple[list[int], float]] = {
                "greedy_nn": best_greedy_path(sim),
                "continuity": best_continuity_path(sim),
            }

            for method_name, (pred, _score) in methods.items():
                metrics = evaluate_ordering_prediction(pred, ground_truth)
                row: dict[str, Any] = {
                    "sequence_id": seq_id,
                    "category": seq.get("category", ""),
                    "method": method_name,
                    "base_seed": base_seed,
                    "seed": seed,
                }
                row.update(metrics)
                rows.append(row)

    fieldnames = ["sequence_id", "category", "method", "base_seed", "seed"] + METRIC_KEYS
    write_csv(OUTPUT_DIR / "multi_seed_results.csv", rows, fieldnames)

    # Also write a per-method, per-seed summary (averaged across sequences)
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["base_seed"])
        buckets[key].append(row)

    summary_rows = []
    for (method, base_seed), group in sorted(buckets.items()):
        entry: dict[str, Any] = {
            "method": method,
            "base_seed": base_seed,
            "n_sequences": len(group),
        }
        for key in METRIC_KEYS:
            vals = [r[key] for r in group if key in r]
            entry[f"mean_{key}"] = round(float(np.mean(vals)), 4) if vals else None
        summary_rows.append(entry)

    summary_fields = ["method", "base_seed", "n_sequences"] + [f"mean_{k}" for k in METRIC_KEYS]
    write_csv(OUTPUT_DIR / "multi_seed_summary.csv", summary_rows, summary_fields)


# ---------------------------------------------------------------------------
# Experiment 3: Continuity weight sweep
# ---------------------------------------------------------------------------

def run_experiment_3(sequences: list[dict]) -> None:
    print("\n=== Experiment 3: Continuity Weight Sweep ===")

    rows: list[dict] = []
    total = len(WEIGHT_VALUES) ** 2 * len(sequences)
    done = 0

    for adj_w in WEIGHT_VALUES:
        for cont_w in WEIGHT_VALUES:
            for seq_idx, seq in enumerate(sequences):
                seq_id = seq["sequence_id"]
                seed = 42 + seq_idx

                embeddings = load_embeddings(seq_id)
                n = embeddings.shape[0]
                shuffled = shuffled_ground_truth(n, seed)
                sim = build_similarity_matrix(embeddings, shuffled)
                ground_truth = true_order_in_shuffled_space(shuffled)

                pred, _score = best_continuity_path(
                    sim,
                    adjacency_weight=adj_w,
                    continuity_weight=cont_w,
                )
                metrics = evaluate_ordering_prediction(pred, ground_truth)

                row: dict[str, Any] = {
                    "sequence_id": seq_id,
                    "category": seq.get("category", ""),
                    "adjacency_weight": adj_w,
                    "continuity_weight": cont_w,
                    "seed": seed,
                }
                row.update(metrics)
                rows.append(row)

                done += 1

            print(
                f"  adj_w={adj_w:.2f} cont_w={cont_w:.2f}  [{done}/{total}]",
                end="\r",
            )

    print()  # newline after \r progress

    fieldnames = [
        "sequence_id",
        "category",
        "adjacency_weight",
        "continuity_weight",
        "seed",
    ] + METRIC_KEYS
    write_csv(OUTPUT_DIR / "weight_sweep_results.csv", rows, fieldnames)

    # Aggregate over sequences for each weight combo
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["adjacency_weight"], row["continuity_weight"])
        buckets[key].append(row)

    summary_rows = []
    for (adj_w, cont_w), group in sorted(buckets.items()):
        entry: dict[str, Any] = {
            "adjacency_weight": adj_w,
            "continuity_weight": cont_w,
            "n_sequences": len(group),
        }
        for key in METRIC_KEYS:
            vals = [r[key] for r in group if key in r]
            entry[f"mean_{key}"] = round(float(np.mean(vals)), 4) if vals else None
        summary_rows.append(entry)

    summary_fields = ["adjacency_weight", "continuity_weight", "n_sequences"] + [
        f"mean_{k}" for k in METRIC_KEYS
    ]
    write_csv(OUTPUT_DIR / "weight_sweep_summary.csv", summary_rows, summary_fields)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ChronoLogic ordering experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        metavar="N",
        help="Which experiments to run (1, 2, 3). Defaults to all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = set(args.exp) if args.exp else {1, 2, 3}

    sequences = load_manifest()
    print(f"Loaded {len(sequences)} sequences from manifest.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if 1 in experiments:
        run_experiment_1(sequences)
    if 2 in experiments:
        run_experiment_2(sequences)
    if 3 in experiments:
        run_experiment_3(sequences)

    print("\nAll requested experiments complete.")
    print(f"Results written to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
