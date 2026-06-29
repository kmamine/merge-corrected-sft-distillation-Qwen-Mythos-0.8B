"""Unit tests for distill.recipe.decide (the merge-vs-continue policy; offline)."""
import math

from distill.recipe import decide


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
