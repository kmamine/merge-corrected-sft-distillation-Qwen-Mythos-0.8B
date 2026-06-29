"""Unit tests for the pure helpers in aero_cpt.utils (SQuAD metrics, formatting).

These exercise the deterministic, dependency-light logic — no GPU, no network.
They also serve as the import smoke test for the `aero_cpt` package layout.
"""
from aero_cpt.utils import (ThroughputMeter, exact_match, f1, human,
                       normalize_answer, score_qa)


class TestNormalizeAnswer:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_answer("The Boeing 747!") == "boeing 747"

    def test_removes_articles(self):
        assert normalize_answer("a an the engine") == "engine"

    def test_collapses_whitespace(self):
        assert normalize_answer("  loss   of   control  ") == "loss of control"


class TestExactMatch:
    def test_match_ignores_articles_and_punctuation(self):
        assert exact_match("the Engine.", "engine") == 1.0

    def test_mismatch(self):
        assert exact_match("engine fire", "bird strike") == 0.0


class TestF1:
    def test_identical_is_one(self):
        assert f1("loss of control", "loss of control") == 1.0

    def test_no_overlap_is_zero(self):
        assert f1("engine fire", "bird strike") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        # pred has 1 of 2 gold tokens right -> precision 1/1, recall 1/2 -> F1 = 2/3
        score = f1("control", "loss control")
        assert 0.0 < score < 1.0
        assert round(score, 4) == round(2 / 3, 4)


class TestScoreQa:
    def test_returns_percent_em_f1_and_count(self):
        preds = ["the engine", "bird strike"]
        golds = ["engine", "engine fire"]
        out = score_qa(preds, golds)
        assert out["n"] == 2
        assert out["em"] == 50.0          # first exact after normalization, second not
        assert 0.0 <= out["f1"] <= 100.0


class TestHuman:
    def test_small_numbers_have_no_unit(self):
        assert human(999) == "999"

    def test_thousands_and_millions(self):
        assert human(1500) == "1.5K"
        assert human(2_000_000) == "2.0M"


class TestThroughputMeter:
    def test_rate_accumulates_tokens(self):
        m = ThroughputMeter()
        m.update(100)
        m.update(50)
        assert m.tokens == 150
        assert m.rate >= 0.0
