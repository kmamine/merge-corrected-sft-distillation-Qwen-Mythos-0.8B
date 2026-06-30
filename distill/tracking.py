"""MLflow logging helpers - thin, best-effort wrappers over the `mlflow` library.

The tracking *server* runs separately (default http://localhost:5000). These
helpers configure the client and wrap a run so any tracking hiccup (server down,
network blip) prints a warning and degrades to a no-op rather than killing a
training/eval run.

    with run("mythos-distill", run_name="epoch1-sft", params={"alpha": 1.0}) as r:
        log_metrics(r, {"gsm8k": 0.31, "mmlu": 0.44}, step=1)
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Optional

DEFAULT_URI = "http://localhost:5000"


def resolve_tracking_uri(explicit: Optional[str] = None) -> str:
    """explicit > $MLFLOW_TRACKING_URI > default localhost:5000 (trailing / stripped)."""
    uri = explicit or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI
    return uri.rstrip("/")


def clean_metrics(metrics: dict) -> dict:
    """Keep only finite numeric metrics (mlflow rejects NaN/inf and non-numbers)."""
    out = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k] = f
    return out


@contextmanager
def run(experiment, run_name=None, tracking_uri=None, tags=None, params=None, nested=False):
    """Best-effort mlflow run context. Yields the live `mlflow` module, or None if disabled.

    Callers should guard on the yielded value (or just use log_metrics/log_params,
    which no-op on None).
    """
    mlflow = None
    try:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
        if experiment:
            mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        if params:
            log_params(mlflow, params)
    except Exception as e:  # noqa: BLE001 - tracking must never break the pipeline
        print(f"[mlflow] disabled ({type(e).__name__}: {e}); continuing without tracking.")
        yield None
        return
    try:
        yield mlflow
    finally:
        try:
            mlflow.end_run()
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] end_run failed ({type(e).__name__}: {e}).")


def log_params(mlflow, params: dict):
    if mlflow is None:
        return
    try:
        mlflow.log_params({k: ("None" if v is None else v) for k, v in params.items()})
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] log_params failed ({type(e).__name__}: {e}).")


def log_metrics(mlflow, metrics: dict, step: Optional[int] = None):
    if mlflow is None:
        return
    clean = clean_metrics(metrics)
    if not clean:
        return
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] log_metrics failed ({type(e).__name__}: {e}).")
