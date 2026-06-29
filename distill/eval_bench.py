"""Benchmark evaluation via EleutherAI lm-evaluation-harness.

`run_benchmarks` evaluates a HF model dir/id on GSM8K / MMLU / ARC-Challenge and
returns a normalized `{task: score}` in [0,1] plus the aggregate (mean). Models are
evaluated in plain completion mode (no chat template) so the base, instruct, SFT,
and merged checkpoints are compared apples-to-apples on the standard harness tasks.

`primary_metric` and `aggregate` are pure and unit-tested; `run_benchmarks` is the
lm-eval call (GPU, heavy).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

# Preferred metric per harness task, by prefix; first match (ignoring the ",filter" suffix) wins.
_METRIC_PREFERENCE = ("exact_match", "acc_norm", "acc")


def primary_metric(task_result: dict) -> Optional[float]:
    """Pick the headline scalar from one lm-eval task-result dict.

    Keys look like 'acc,none' / 'acc_norm,none' / 'exact_match,strict-match'. We take
    the first key whose metric name (before the comma) matches our preference order.
    """
    for pref in _METRIC_PREFERENCE:
        for key, val in task_result.items():
            if not isinstance(key, str) or "_stderr" in key:
                continue
            name = key.split(",", 1)[0]
            if name == pref and isinstance(val, (int, float)) and not math.isnan(val):
                return float(val)
    return None


def aggregate(scores: Dict[str, Optional[float]]) -> float:
    """Mean of the finite per-task scores (NaN/None dropped); NaN if none usable."""
    vals = [v for v in scores.values()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def extract_scores(results: dict, tasks: List[str]) -> Dict[str, Optional[float]]:
    """From lm-eval's `results['results']`, pull the primary metric for each requested task."""
    res = results.get("results", results)
    out: Dict[str, Optional[float]] = {}
    for t in tasks:
        out[t] = primary_metric(res[t]) if t in res else None
    return out


def run_benchmarks(model_path, tasks, limit=None, batch_size=8, device="cuda:0",
                   dtype="bfloat16", num_fewshot=None) -> Dict[str, Optional[float]]:
    """Run lm-eval on `tasks`; return {task: primary_metric} (+ '_aggregate')."""
    import lm_eval

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    lim = None if (limit is None or limit <= 0) else limit
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={dtype}",
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=lim,
    )
    scores = extract_scores(results, tasks)
    scores["_aggregate"] = aggregate({k: v for k, v in scores.items() if k != "_aggregate"})
    return scores


def main():
    """Standalone full eval of one or more checkpoints: python -m distill.eval_bench ..."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="HF ids or local dirs")
    ap.add_argument("--tasks", default="gsm8k,mmlu,arc_challenge")
    ap.add_argument("--limit", type=int, default=0, help="<=0 = full")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    width = max(len(m) for m in args.models)
    print(f"{'model'.ljust(width)}  " + "  ".join(f"{t:>12}" for t in tasks + ["aggregate"]))
    for m in args.models:
        s = run_benchmarks(m, tasks, limit=args.limit, batch_size=args.batch_size, device=args.device)
        cells = "  ".join(f"{s.get(t):>12.4f}" if isinstance(s.get(t), (int, float)) else f"{'n/a':>12}"
                          for t in tasks + ["_aggregate"])
        print(f"{m.ljust(width)}  {cells}")


if __name__ == "__main__":
    main()
