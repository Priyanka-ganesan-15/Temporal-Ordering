"""Error taxonomy diagnostics for ordering predictions."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from chronologic.evaluation.metrics import count_inversions, pairwise_order_accuracy

TaxonomyRow = dict[str, str | float]


def classify_prediction_error(predicted_order: list[int], true_order: list[int]) -> str:
    """Classify one prediction into a small error taxonomy."""
    if predicted_order == true_order:
        return "exact"
    if predicted_order == list(reversed(true_order)):
        return "reversed"

    if _is_local_swap(predicted_order, true_order):
        return "local_swap"

    if _is_endpoint_error(predicted_order, true_order):
        return "endpoint_error"

    return "scrambled"


def summarize_error_taxonomy(rows: list[TaxonomyRow]) -> list[TaxonomyRow]:
    """Aggregate taxonomy counts and fractions by method."""
    totals: dict[str, int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        method = str(row["method"])
        taxonomy = str(row["taxonomy"])
        totals[method] += 1
        counts[(method, taxonomy)] += 1

    summary_rows: list[TaxonomyRow] = []
    taxonomies = ["exact", "reversed", "local_swap", "endpoint_error", "scrambled"]
    for method in sorted(totals):
        total = totals[method]
        for taxonomy in taxonomies:
            count = counts.get((method, taxonomy), 0)
            fraction = (count / total) if total > 0 else 0.0
            summary_rows.append(
                {
                    "method": method,
                    "taxonomy": taxonomy,
                    "count": float(count),
                    "fraction": float(fraction),
                }
            )
    return summary_rows


def write_error_taxonomy_summary(path: Path, rows: list[TaxonomyRow]) -> None:
    """Write taxonomy rows to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_error_taxonomy_summary(summary_rows: list[TaxonomyRow], output_path: Path) -> None:
    """Plot stacked bar chart of taxonomy fractions by method."""
    methods = sorted({str(row["method"]) for row in summary_rows})
    taxonomies = ["exact", "reversed", "local_swap", "endpoint_error", "scrambled"]
    colors = {
        "exact": "#2a9d8f",
        "reversed": "#e76f51",
        "local_swap": "#f4a261",
        "endpoint_error": "#457b9d",
        "scrambled": "#6d597a",
    }

    fractions_by_method = {
        method: {taxonomy: 0.0 for taxonomy in taxonomies} for method in methods
    }
    for row in summary_rows:
        fractions_by_method[str(row["method"])][str(row["taxonomy"])] = float(row["fraction"])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bottom = [0.0 for _ in methods]
    for taxonomy in taxonomies:
        values = [fractions_by_method[method][taxonomy] for method in methods]
        ax.bar(methods, values, bottom=bottom, color=colors[taxonomy], label=taxonomy)
        bottom = [current + value for current, value in zip(bottom, values)]

    ax.set_ylabel("Fraction of sequences")
    ax.set_xlabel("Method")
    ax.set_title("Error Taxonomy Summary by Method")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _is_local_swap(predicted_order: list[int], true_order: list[int]) -> bool:
    n_frames = len(true_order)
    predicted_positions = {frame: idx for idx, frame in enumerate(predicted_order)}
    displacements = [abs(predicted_positions[frame] - frame) for frame in true_order]

    max_disp = max(displacements)
    inv_count = count_inversions(predicted_order, true_order)
    return max_disp <= 1 and inv_count <= 2


def _is_endpoint_error(predicted_order: list[int], true_order: list[int]) -> bool:
    if len(true_order) < 4:
        return False

    endpoint_set = {true_order[0], true_order[-1]}
    predicted_front_back = {predicted_order[0], predicted_order[-1]}
    if endpoint_set == predicted_front_back:
        return False

    interior_true = true_order[1:-1]
    interior_pred = [frame for frame in predicted_order if frame in interior_true]
    if interior_pred != interior_true:
        return False

    return pairwise_order_accuracy(predicted_order, true_order) >= 0.7


__all__ = [
    "classify_prediction_error",
    "plot_error_taxonomy_summary",
    "summarize_error_taxonomy",
    "write_error_taxonomy_summary",
]
