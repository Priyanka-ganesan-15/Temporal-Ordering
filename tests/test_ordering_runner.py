from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chronologic.evaluation.runner import (
    evaluate_method_on_sequence,
    run_full_evaluation,
    save_reordering_storyboards,
    save_visual_reports,
)
from temporal_ordering.models import Sequence


def test_evaluate_method_on_sequence_returns_method_rows() -> None:
    sequence = Sequence(
        sequence_id="sequence_a",
        category="demo",
        caption="first",
        difficulty="easy",
        sequence_type="procedural",
        num_frames=4,
        frames=[Path(f"frame_{idx}.png") for idx in range(4)],
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.55, 0.83],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rows = evaluate_method_on_sequence(
        sequence,
        embedding_provider=lambda _: embeddings,
        seed=7,
    )

    assert len(rows) == 6
    assert {row["method"] for row in rows} == {
        "random",
        "greedy_nearest_neighbor",
        "continuity",
        "continuity_plus_direction",
        "continuity_plus_endpoint",
        "continuity_plus_reverse_disambiguation",
    }
    for row in rows:
        assert row["predicted_order"]
        assert row["ground_truth_order"] == "0 1 2 3"


def test_run_full_evaluation_returns_sequence_and_method_summaries() -> None:
    sequences = [
        Sequence(
            sequence_id="sequence_a",
            category="demo",
            caption="first",
            difficulty="easy",
            sequence_type="procedural",
            num_frames=4,
            frames=[Path(f"frame_{idx}.png") for idx in range(4)],
        )
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.55, 0.83],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    results_rows, summary_by_method, summary_by_sequence = run_full_evaluation(
        sequences,
        embedding_provider=lambda _: embeddings,
        seed=7,
    )

    assert len(results_rows) == 6
    assert len(summary_by_method) == 6
    assert len(summary_by_sequence) == 1
    assert "temporal_contrast_score" in summary_by_sequence[0]


def test_save_visual_reports_creates_png_outputs(tmp_path: Path) -> None:
    results_rows = [
        {
            "method": "random",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 0.50,
        },
        {
            "method": "greedy_nearest_neighbor",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 0.75,
        },
        {
            "method": "continuity",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 1.00,
        },
        {
            "method": "continuity_plus_direction",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 1.00,
        },
        {
            "method": "continuity_plus_endpoint",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 1.00,
        },
        {
            "method": "continuity_plus_reverse_disambiguation",
            "temporal_contrast_score": 0.10,
            "pairwise_order_accuracy": 1.00,
        },
    ]
    summary_by_method = [
        {
            "method": "random",
            "exact_match_accuracy": 0.0,
            "pairwise_order_accuracy": 0.5,
            "normalized_kendall_agreement": 0.5,
        },
        {
            "method": "greedy_nearest_neighbor",
            "exact_match_accuracy": 0.5,
            "pairwise_order_accuracy": 0.75,
            "normalized_kendall_agreement": 0.75,
        },
        {
            "method": "continuity",
            "exact_match_accuracy": 1.0,
            "pairwise_order_accuracy": 1.0,
            "normalized_kendall_agreement": 1.0,
        },
        {
            "method": "continuity_plus_direction",
            "exact_match_accuracy": 1.0,
            "pairwise_order_accuracy": 1.0,
            "normalized_kendall_agreement": 1.0,
        },
        {
            "method": "continuity_plus_endpoint",
            "exact_match_accuracy": 1.0,
            "pairwise_order_accuracy": 1.0,
            "normalized_kendall_agreement": 1.0,
        },
        {
            "method": "continuity_plus_reverse_disambiguation",
            "exact_match_accuracy": 1.0,
            "pairwise_order_accuracy": 1.0,
            "normalized_kendall_agreement": 1.0,
        },
    ]
    summary_by_sequence = [
        {
            "sequence_id": "sequence_1",
            "random__pairwise_order_accuracy": 0.50,
            "greedy_nearest_neighbor__pairwise_order_accuracy": 0.75,
            "continuity__pairwise_order_accuracy": 1.00,
            "continuity_plus_direction__pairwise_order_accuracy": 1.00,
            "continuity_plus_endpoint__pairwise_order_accuracy": 1.00,
            "continuity_plus_reverse_disambiguation__pairwise_order_accuracy": 1.00,
        }
    ]

    paths = save_visual_reports(
        output_dir=tmp_path,
        results_rows=results_rows,
        summary_by_method=summary_by_method,
        summary_by_sequence=summary_by_sequence,
    )

    assert len(paths) == 3
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"


def test_save_reordering_storyboards_creates_sequence_pngs(tmp_path: Path) -> None:
    frame_paths: list[Path] = []
    for idx in range(4):
        frame_path = tmp_path / f"frame_{idx}.png"
        pixel = np.ones((8, 8, 3), dtype=np.float32) * (idx + 1) / 4.0
        plt.imsave(frame_path, pixel)
        frame_paths.append(frame_path)

    sequence = Sequence(
        sequence_id="sequence_x",
        category="demo",
        caption="caption",
        difficulty="easy",
        sequence_type="procedural",
        num_frames=4,
        frames=frame_paths,
    )

    results_rows = [
        {
            "sequence_id": "sequence_x",
            "method": "random",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 0 3 2",
            "pairwise_order_accuracy": 0.5,
            "exact_match_accuracy": 0.0,
        },
        {
            "sequence_id": "sequence_x",
            "method": "greedy_nearest_neighbor",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 3 0 2",
            "pairwise_order_accuracy": 0.75,
            "exact_match_accuracy": 0.0,
        },
        {
            "sequence_id": "sequence_x",
            "method": "continuity",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 2 0 3",
            "pairwise_order_accuracy": 1.0,
            "exact_match_accuracy": 1.0,
        },
        {
            "sequence_id": "sequence_x",
            "method": "continuity_plus_direction",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 2 0 3",
            "pairwise_order_accuracy": 1.0,
            "exact_match_accuracy": 1.0,
        },
        {
            "sequence_id": "sequence_x",
            "method": "continuity_plus_endpoint",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 2 0 3",
            "pairwise_order_accuracy": 1.0,
            "exact_match_accuracy": 1.0,
        },
        {
            "sequence_id": "sequence_x",
            "method": "continuity_plus_reverse_disambiguation",
            "shuffle_order": "2 0 3 1",
            "predicted_shuffled_order": "1 2 0 3",
            "pairwise_order_accuracy": 1.0,
            "exact_match_accuracy": 1.0,
        },
    ]

    paths = save_reordering_storyboards(
        output_dir=tmp_path,
        sequences=[sequence],
        results_rows=results_rows,
    )

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"