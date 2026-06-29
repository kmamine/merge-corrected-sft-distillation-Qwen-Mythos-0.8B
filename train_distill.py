"""Merge-corrected iterative SFT distillation — the orchestrator (multi-GPU).

Per epoch: SFT one epoch (multi-GPU, both cuda:0+cuda:1) -> build the
merge-corrected candidate (instruct + alpha*(sft - instruct), a soup back toward the
original instruct) -> benchmark both (limited) -> keep the better as the next epoch's
start (merge wins ties = a guard against forgetting). Logs to MLflow; a final full
benchmark pass compares the original instruct / final-SFT / best.

    MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml

Single-process driver: it shells out to `sft_worker.py` via `accelerate launch` for
the multi-GPU SFT step (CUDA_VISIBLE_DEVICES=0,1), and runs merge (CPU) + benchmarks
(cuda:1, the free GPU) in-process — keeping the orchestration free of rank guards.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

from distill import tracking
from distill.config import load_config, parse_kv
from distill.eval_bench import run_benchmarks
from distill.merge import merge_models
from distill.recipe import decide

ACCEL_CONFIG = os.environ.get("ACCEL_CONFIG", "configs/accelerate_multi.yaml")
SFT_GPUS = os.environ.get("SFT_CUDA_VISIBLE_DEVICES", "0,1")
EVAL_DEVICE = os.environ.get("EVAL_DEVICE", "cuda:1")          # free GPU (cuda:0 is shared)


def sft_one_epoch(args, init_model, out_dir):
    cmd = ["accelerate", "launch", "--config_file", ACCEL_CONFIG, "sft_worker.py",
           "--config", args.config, "--init_model", init_model, "--out", out_dir]
    if args.overrides:
        cmd += ["--set", *args.overrides]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": SFT_GPUS}
    print(f"[recipe] SFT epoch -> {out_dir} (init={init_model}, gpus={SFT_GPUS})")
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def prune_checkpoints(created, keep, protected):
    """Keep at most `keep` checkpoint dirs (oldest deleted); never touch `protected`."""
    count = len([d for d in created if os.path.isdir(d)])
    for d in created:                                  # oldest-first
        if count <= keep:
            break
        if d in protected or not os.path.isdir(d):
            continue
        shutil.rmtree(d, ignore_errors=True)
        print(f"[recipe] pruned old checkpoint: {d}")
        count -= 1


def bench(model_path, cfg, limit, label):
    print(f"[recipe] eval {label}: {model_path} (limit={limit or 'full'})")
    scores = run_benchmarks(model_path, cfg.eval_tasks, limit=limit,
                            batch_size=cfg.eval_batch_size, device=EVAL_DEVICE)
    print(f"[recipe] {label}: {scores}")
    return scores


def log_eval(cfg, run_name, scores, params, step=None):
    with tracking.run(cfg.mlflow_experiment, run_name=run_name,
                      tracking_uri=cfg.mlflow_tracking_uri,
                      tags={k: str(v) for k, v in params.items()}, params=params) as r:
        tracking.log_metrics(r, scores, step=step)


def write_results(path, cfg, history, finals, tasks):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = tasks + ["_aggregate"]
    header = "| model | " + " | ".join(cols) + " |"
    sep = "| --- " * (len(cols) + 1) + "|"

    def row(name, scores):
        cells = [f"{scores.get(c):.4f}" if isinstance(scores.get(c), (int, float)) else "n/a" for c in cols]
        return f"| {name} | " + " | ".join(cells) + " |"

    lines = ["# Mythos distillation — benchmark results", "",
             f"- instruct `{cfg.instruct_model}` · dataset `{cfg.dataset}`",
             f"- E={cfg.num_epochs} · merge_alpha={cfg.merge_alpha} "
             f"(soup: instruct + α·(sft−instruct))", "",
             f"## Per-epoch (in-loop, limit={cfg.eval_limit})", "", header, sep]
    for h in history:
        lines.append(row(f"epoch{h['epoch']}-sft", h["sft"]))
        lines.append(row(f"epoch{h['epoch']}-merge", h["merge"]))
        lines.append(f"| → decision epoch{h['epoch']}: **{h['decision']}** | | | |")
    lines += ["", "## Final (full benchmarks)", "", header, sep]
    for name, scores in finals.items():
        lines.append(row(name, scores))
    lines += ["", "_Plain completion-mode lm-eval (no chat template), apples-to-apples._", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[recipe] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    ap.add_argument("--results", default="results/benchmarks.md")
    ap.add_argument("--final_limit", type=int, default=0, help="final per-task cap; <=0 = full")
    ap.add_argument("--keep_checkpoints", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_kv(args.overrides))
    tasks = [t.strip() for t in cfg.eval_tasks.split(",") if t.strip()]

    instruct = cfg.instruct_model
    live = instruct                                   # SFT starts from the instruct model
    best = {"score": float("-inf"), "path": None, "tag": None}
    history, created = [], []

    for k in range(1, cfg.num_epochs + 1):
        ep = os.path.join(cfg.output_dir, f"epoch{k}")
        out_sft = sft_one_epoch(args, live, os.path.join(ep, "sft"))
        out_merge = merge_models(instruct, out_sft, os.path.join(ep, "merge"), cfg.merge_alpha)
        created += [out_sft, out_merge]

        s_sft = bench(out_sft, cfg, cfg.eval_limit, f"epoch{k}-sft")
        s_merge = bench(out_merge, cfg, cfg.eval_limit, f"epoch{k}-merge")
        choice, _ = decide(s_sft["_aggregate"], s_merge["_aggregate"])
        live = out_merge if choice == "merge" else out_sft

        log_eval(cfg, f"epoch{k}-sft", s_sft, {"epoch": k, "side": "sft"}, step=k)
        log_eval(cfg, f"epoch{k}-merge", s_merge,
                 {"epoch": k, "side": "merge", "decision": choice}, step=k)
        history.append({"epoch": k, "sft": s_sft, "merge": s_merge, "decision": choice})
        print(f"[recipe] epoch {k}: decision={choice} -> live={live}")

        for cand, sc in [(out_sft, s_sft), (out_merge, s_merge)]:
            agg = sc["_aggregate"]
            if isinstance(agg, (int, float)) and agg > best["score"]:
                best = {"score": agg, "path": cand,
                        "tag": f"epoch{k}-{'sft' if cand == out_sft else 'merge'}"}
        prune_checkpoints(created, args.keep_checkpoints, protected={live, best["path"]})

    print(f"[recipe] best checkpoint: {best['path']} (agg={best['score']:.4f})")

    # ---- FULL final benchmarks: original instruct baseline vs final-SFT vs best ----
    final_sft = os.path.join(cfg.output_dir, f"epoch{cfg.num_epochs}", "sft")
    targets = {f"instruct ({instruct})": instruct,
               "final-SFT": final_sft, f"best ({best['tag']})": best["path"]}
    finals = {}
    for name, path in targets.items():
        if not path or (path != instruct and not os.path.isdir(path)):
            print(f"[recipe] skip final {name}: {path} not on disk")
            continue
        finals[name] = bench(path, cfg, args.final_limit, f"final::{name}")
        log_eval(cfg, f"final-{name}", finals[name], {"phase": "final", "model": name})

    write_results(args.results, cfg, history, finals, tasks)
    print(f"[recipe] DONE. Publish candidate: {best['path']}")


if __name__ == "__main__":
    main()
