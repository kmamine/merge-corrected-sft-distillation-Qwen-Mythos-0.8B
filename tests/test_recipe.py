"""Unit tests for distill.recipe (the merge-vs-continue + best-candidate policies; offline)."""
import math

from distill.recipe import decide, pick_best


class TestDecide:
    def test_merge_strictly_better(self):
        assert decide(0.40, 0.45) == ("merge", 0.45)

    def test_sft_strictly_better(self):
        assert decide(0.50, 0.40) == ("sft", 0.50)

    def test_tie_goes_to_merge(self):
        assert decide(0.42, 0.42) == ("merge", 0.42)

    def test_missing_sft_takes_merge(self):
        assert decide(None, 0.30) == ("merge", 0.30)
        assert decide(float("nan"), 0.30) == ("merge", 0.30)

    def test_missing_merge_takes_sft(self):
        assert decide(0.30, None) == ("sft", 0.30)

    def test_both_missing_continues_sft(self):
        choice, score = decide(None, None)
        assert choice == "sft" and math.isnan(score)


class TestPickBest:
    def test_highest_aggregate_wins(self):
        name, score = pick_best({"sft": 0.40, "linear": 0.45, "ties": 0.42})
        assert name == "linear" and score == 0.45

    def test_tie_prefers_a_merge_over_sft(self):
        name, _ = pick_best({"sft": 0.45, "linear": 0.45})
        assert name == "linear"

    def test_nan_or_none_treated_as_worst(self):
        name, _ = pick_best({"sft": 0.40, "slerp": float("nan"), "dare_linear": None})
        assert name == "sft"
