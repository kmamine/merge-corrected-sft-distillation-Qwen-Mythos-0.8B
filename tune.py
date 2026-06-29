"""Autonomous hyperparameter loop (autoresearch-style keep/discard) over our recipe.

Borrows the "propose a change → run cheap → score → keep-or-discard → repeat" loop
philosophy (à la karpathy/autoresearch) but drives our own train_distill recipe — no
external dependency. Each trial is a CHEAP proxy run (1 epoch, a data subset, fast
benchmarks, no final eval); we keep the config with the best in-loop aggregate. The
winning learning rate is then used for one full recipe run + publish.

The over-memorization in the first full run (train loss → 0.02, 99% token accuracy)
points at the learning rate, so that's the swept knob here.

    python tune.py            # runs the sweep, prints the winner
"""
from __future__ import annotations

import os
import re
import subprocess

from distill import tracking

# cheap proxy so each trial is ~20 min instead of ~3.5 h.
# output_dir=outputs/tune keeps trials from clobbering the full run's checkpoints.
PROXY = ["num_epochs=1", "max_samples=8000", "eval_limit=64",
         "output_dir=outputs/tune", "mlflow_experiment=mythos-distill-tune"]
# coordinate search over the dominant knob; baseline first
TRIALS = [
    {"learning_rate": "1e-5"},   # baseline (the over-memorizing run)
    {"learning_rate": "5e-6"},
    {"learning_rate": "2e-6"},
    {"learning_rate": "1e-6"},
]
EXPERIMENT = "mythos-distill-tune"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def run_trial(i, overrides):
    os.makedirs("results/tune", exist_ok=True)
    log_path = f"results/tune/trial{i}.log"
    sets = PROXY + [f"{k}={v}" for k, v in overrides.items()]
    cmd = ["python", "train_distill.py", "--config", "configs/sft.yaml",
           "--no_final", "--results", f"results/tune/trial{i}.md", "--set", *sets]
    print(f"[tune] trial {i}: {overrides} -> {log_path}")
    with open(log_path, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ})
    txt = open(log_path, errors="ignore").read()
    agg = float(m.group(1)) if (m := re.search(r"best checkpoint:.*agg=([0-9.]+)", txt)) else float("nan")
    losses = [float(x) for x in re.findall(r"train_loss'?:\s*'?([0-9.eE+-]+)", txt)]
    final_loss = losses[-1] if losses else float("nan")
    return agg, final_loss


def main():
    best = None
    board = []
    for i, overrides in enumerate(TRIALS):
        agg, final_loss = run_trial(i, overrides)
        keep = best is None or agg > best["agg"]
        verdict = "KEEP" if keep else "discard"
        if keep:
            best = {"trial": i, "overrides": overrides, "agg": agg}
        board.append((i, overrides, agg, final_loss, verdict))
        print(f"[tune] trial {i} {overrides}: agg={agg:.4f} final_loss={final_loss:.4f} -> {verdict}")
        with tracking.run(EXPERIMENT, run_name=f"trial{i}-lr{overrides.get('learning_rate')}",
                          tracking_uri=TRACKING_URI, params=overrides) as r:
            tracking.log_metrics(r, {"proxy_aggregate": agg, "final_train_loss": final_loss}, step=i)

    print("\n=== tuning leaderboard (proxy: 1 epoch, 8k samples, eval_limit 64) ===")
    print(f"{'trial':>5}  {'overrides':<28} {'agg':>8} {'final_loss':>11}  verdict")
    for i, ov, agg, fl, v in board:
        print(f"{i:>5}  {str(ov):<28} {agg:>8.4f} {fl:>11.4f}  {v}")
    print(f"\n[tune] WINNER: trial {best['trial']} {best['overrides']} (agg={best['agg']:.4f})")
    print(f"[tune] next: full run with {best['overrides']}")


if __name__ == "__main__":
    main()
