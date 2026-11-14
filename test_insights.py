"""Tests use hand-made vectors, so no model download is needed."""
import numpy as np
import pytest

from insights import find_patterns


def vectors():
    # Two tight groups pointing in different directions, plus one outlier.
    return np.array([
        [1.0, 0.05, 0.0],
        [0.95, 0.1, 0.0],
        [0.0, 1.0, 0.05],
        [0.05, 0.9, 0.0],
        [0.1, 0.95, 0.05],
        [0.0, 0.0, 1.0],
    ])


QUOTES = ["a1", "a2", "b1", "b2", "b3", "alone"]


def test_groups_similar_quotes_and_leaves_outlier():
    patterns, unmatched = find_patterns(QUOTES, vectors(), threshold=0.5)
    assert [sorted(p.quotes) for p in patterns] == [["b1", "b2", "b3"], ["a1", "a2"]]
    assert unmatched == ["alone"]
    assert all(p.cohesion > 0.9 for p in patterns)
    assert all(p.representative in p.quotes for p in patterns)


def test_result_does_not_depend_on_input_order():
    order = [5, 3, 0, 4, 1, 2]
    shuffled, _ = find_patterns([QUOTES[i] for i in order], vectors()[order], threshold=0.5)
    original, _ = find_patterns(QUOTES, vectors(), threshold=0.5)
    as_sets = lambda ps: {frozenset(p.quotes) for p in ps}
    assert as_sets(shuffled) == as_sets(original)


def test_high_threshold_groups_nothing():
    patterns, unmatched = find_patterns(["x", "y"], np.array([[1.0, 0.0], [0.0, 1.0]]), threshold=0.9)
    assert patterns == [] and unmatched == ["x", "y"]


def test_fewer_than_two_quotes():
    assert find_patterns([], np.empty((0, 3))) == ([], [])
    assert find_patterns(["only"], np.array([[1.0, 0.0]])) == ([], ["only"])


@pytest.mark.parametrize("bad", [0, 1, -0.2, 1.5])
def test_rejects_bad_threshold(bad):
    with pytest.raises(ValueError):
        find_patterns(QUOTES, vectors(), threshold=bad)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        find_patterns(["a", "b", "c"], vectors())
