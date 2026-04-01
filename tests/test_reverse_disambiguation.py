from chronologic.ordering.reverse_disambiguation import (
    choose_oriented_path,
    compare_forward_reverse_scores,
)


def test_compare_forward_reverse_scores_reports_gap() -> None:
    path = [0, 1, 2]

    def base_score_fn(candidate: list[int]) -> float:
        # Favor smaller first index.
        return float(-candidate[0])

    metrics = compare_forward_reverse_scores(path, base_score_fn)

    assert metrics["forward_score"] == 0.0
    assert metrics["reverse_score"] == -2.0
    assert metrics["score_gap"] == 2.0


def test_choose_oriented_path_uses_base_when_not_ambiguous() -> None:
    path = [0, 1, 2]

    def base_score_fn(candidate: list[int]) -> float:
        return float(candidate[0])

    selected, metrics = choose_oriented_path(path, base_score_fn, epsilon=1e-6)

    assert selected == [2, 1, 0]
    assert metrics["is_ambiguous"] == 0.0
    assert metrics["used_direction_tiebreak"] == 0.0


def test_choose_oriented_path_uses_direction_tiebreak_when_ambiguous() -> None:
    path = [0, 1, 2]

    def base_score_fn(_: list[int]) -> float:
        return 1.0

    def direction_score_fn(candidate: list[int]) -> float:
        return float(-candidate[0])

    selected, metrics = choose_oriented_path(
        path,
        base_score_fn,
        direction_score_fn=direction_score_fn,
        epsilon=1e-6,
    )

    assert selected == [0, 1, 2]
    assert metrics["is_ambiguous"] == 1.0
    assert metrics["used_direction_tiebreak"] == 1.0
    assert metrics["forward_direction_score"] > metrics["reverse_direction_score"]


def test_choose_oriented_path_defaults_forward_on_ambiguous_without_direction() -> None:
    path = [0, 1, 2]

    selected, metrics = choose_oriented_path(path, lambda _: 0.0, epsilon=1e-6)

    assert selected == [0, 1, 2]
    assert metrics["is_ambiguous"] == 1.0
    assert metrics["used_direction_tiebreak"] == 0.0
