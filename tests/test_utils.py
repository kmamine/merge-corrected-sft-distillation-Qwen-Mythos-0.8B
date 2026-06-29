"""Unit tests for distill.utils (formatting + throughput; offline)."""
from distill.utils import ThroughputMeter, human


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
