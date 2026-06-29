"""Unit tests for distill.tracking pure helpers (offline; no mlflow server)."""
import math

from distill.tracking import DEFAULT_URI, clean_metrics, resolve_tracking_uri


class TestResolveTrackingUri:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert resolve_tracking_uri() == DEFAULT_URI

    def test_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://host:5000/")
        assert resolve_tracking_uri() == "http://host:5000"   # trailing slash stripped

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
        assert resolve_tracking_uri("http://explicit:9999") == "http://explicit:9999"


class TestCleanMetrics:
    def test_keeps_finite_numbers(self):
        assert clean_metrics({"a": 1, "b": 2.5, "c": "3"}) == {"a": 1.0, "b": 2.5, "c": 3.0}

    def test_drops_nan_inf_and_nonnumeric(self):
        out = clean_metrics({"nan": float("nan"), "inf": float("inf"),
                             "txt": "abc", "none": None, "ok": 0.5})
        assert out == {"ok": 0.5}
        assert all(math.isfinite(v) for v in out.values())
