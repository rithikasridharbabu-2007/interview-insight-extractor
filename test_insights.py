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


# --- theme labels -----------------------------------------------------------

from insights import label_themes

BILLING = ["The pricing page confused me", "I could not find the pricing page", "Pricing page was unclear"]
SPEED = ["The app is slow to load", "Slow to load every morning", "Reports load so slow"]


def test_labels_use_words_that_set_a_group_apart():
    billing, speed = label_themes([BILLING, SPEED], others=["I want dark mode"])
    assert "pricing page" in billing and "slow" in speed and "load" in speed
    assert "slow" not in billing and "pricing" not in speed
    assert "dark" not in billing + speed


def test_labels_are_short_and_skip_stop_words():
    for label in label_themes([BILLING, SPEED]):
        words = label.replace(",", "").split()
        assert 1 <= len(words) <= 5
        assert not {"the", "to", "i", "is", "was"} & set(words)


def test_label_does_not_repeat_a_word():
    (label,) = label_themes([["free plan limits", "the free plan", "plan limits on free"]], others=["search is fast"])
    words = label.replace(",", "").split()
    assert len(words) == len(set(words))


def test_label_is_empty_when_there_are_no_usable_words():
    assert label_themes([["a1", "a2"]], others=["b1"]) == [""]
    assert label_themes([]) == []


def test_find_patterns_fills_in_labels():
    quotes = BILLING[:2] + SPEED[:2]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    patterns, _ = find_patterns(quotes, embeddings, threshold=0.5)
    assert ["pricing" in p.label for p in patterns] == [True, False]
    assert all(p.label for p in patterns)
