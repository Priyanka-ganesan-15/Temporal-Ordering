from chronologic.ordering.random_baseline import random_permutation


def test_random_permutation_returns_full_index_set() -> None:
    permutation = random_permutation(5)

    assert len(permutation) == 5
    assert sorted(permutation) == [0, 1, 2, 3, 4]


def test_random_permutation_is_seeded() -> None:
    assert random_permutation(8, seed=13) == random_permutation(8, seed=13)


def test_random_permutation_rejects_non_positive_n() -> None:
    try:
        random_permutation(0)
    except ValueError as exc:
        assert str(exc) == "n must be greater than 0"
    else:
        raise AssertionError("Expected ValueError for n <= 0")