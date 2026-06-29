"""Merge-corrected iterative SFT distillation — the orchestrator.

Per epoch: SFT one epoch (multi-GPU subprocess) -> build the merge-corrected
candidate -> benchmark both (limited) -> decide whether to recorrect with the merge
or continue plain SFT -> seed the next epoch with the winner. Everything is logged
to MLflow. After the loop, a FULL benchmark pass compares the original base/instruct
models, the final SFT trajectory, and the best merged checkpoint, written to
results/benchmarks.md.

    MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml

Single-process driver: it shells out to `sft_worker.py` via `accelerate launch` for
the multi-GPU SFT step (CUDA_VISIBLE_DEVICES=0,1), and runs merge (CPU) + benchmarks
(cuda:1, the free GPU) in-process.
"""
from __future__ import annotations

import argparse
import os
import subprocess

from distill import tracking
from distill.config import load_config, parse_kv
from distill.eval_bench import run_benchmarks
from distill.merge import merge_models
from distill.recipe import decide

ACCEL_CONFIG = os.environ.get("ACCEL_CONFIG", "configs/accelerate_multi.yaml")
SFT_GPUS = os.environ.get("SFT_CUDA_VISIBLE_DEVICES", "0,1")
EVAL_DEVICE = os.environ.get("EVAL_DEVICE", "cuda:1")


def sft_one_epoch(args, init_model, out_dir):
    cmd = ["accelerate", "launch", "--config_file", ACCEL_CONFIG, "sft_worker.py",
           "--config", args.config, "--init_model", init_model, "--out", out_dir]
    if args.overrides:
        cmd += ["--set", *args.overrides]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": SFT_GPUS}
    print(f"[recipe] SFT epoch -> {out_dir}  (init={init_model}, gpus={SFT_GPUS})")
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def bench(model_path, cfg, limit, label):
    print(f"[recipe] benchmarking {label}: {model_path}  (limit={limit or 'full'})")
    scores = run_benchmarks(model_path, cfg.eval_tasks, limit=limit,
                            batch_size=cfg.eval_batch_size, device=EVAL_DEVICE)
    print(f"[recipe] {label} scores: {scores}")
    return scores


def log_eval(cfg, run_name, scores, params, step=None):
    with tracking.run(cfg.mlflow_experiment, run_name=run_name,
                      tracking_uri=cfg.mlflow_tracking_uri,
                      tags={k: str(v) for k, v in params.items()}, params=params) as r:
        tracking.log_metrics(r, scores, step=step)


def write_results(path, cfg, history, finals, tasks):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = tasks + ["_aggregate"]
    lines = ["# Mythos distillation — benchmark results", "",
             f"- base: `{cfg.base_model}`  ·  instruct: `{cfg.instruct_model}`",
             f"- dataset: `{cfg.dataset}`  ·  epochs E={cfg.num_epochs}  ·  "
             f"merge_alpha={cfg.merge_alpha}  ·  chat_alpha={cfg.chat_alpha}", ""]

    def row(name, scores):
        cells = [f"{scores.get(c):.4f}" if isinstance(scores.get(c), (int, float)) else "n/a" for c in cols]
        return f"| {name} | " + " | ".join(cells) + " |"

    header = "| model | " + " | ".join(cols) + " |"
    sep = "| --- " * (len(cols) + 1) + "|"

    lines += [f"## Per-epoch (in-loop, limit={cfg.eval_limit})", "", header, sep]
    for h in history:
        lines.append(row(f"epoch{h['epoch']}-sft", h["sft"]))
        lines.append(row(f"epoch{h['epoch']}-merge", h["merge"]))
        lines.append(f"| → decision epoch{h['epoch']} | **{h['decision']}** |")
    lines += ["", "## Final (full benchmarks)", "", header, sep]
    for name, scores in finals.items():
        lines.append(row(name, scores))
    lines += ["", "_Plain completion-mode lm-eval (no chat template) for apples-to-apples "
              "across base / instruct / SFT / merged._", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[recipe] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    ap.add_argument("--results", default="results/benchmarks.md")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_kv(args.overrides))
    tasks = [t.strip() for t in cfg.eval_tasks.split(",") if t.strip()]

    base, instruct = cfg.base_model, cfg.instruct_model
    live = base
    best = {"score": float("-inf"), "path": None, "tag": None}
    history = []

    for k in range(1, cfg.num_epochs + 1):
        ep_dir = os.path.join(cfg.output_dir, f"epoch{k}")
        out_sft = sft_one_epoch(args, live, os.path.join(ep_dir, "sft"))
        out_merge = merge_models(base, instruct, out_sft, os.path.join(ep_dir, "merge"),
                                 cfg.merge_alpha, cfg.chat_alpha)

        s_sft = bench(out_sft, cfg, cfg.eval_limit, f"epoch{k}-sft")
        s_merge = bench(out_merge, cfg, cfg.eval_limit, f"epoch{k}-merge")
        choice, score = decide(s_sft["_aggregate"], s_merge["_aggregate"])
        live = out_merge if choice == "merge" else out_sft

        log_eval(cfg, f"epoch{k}-sft", s_sft,
                 {"epoch": k, "side": "sft", "merge_alpha": cfg.merge_alpha}, step=k)
        log_eval(cfg, f"epoch{k}-merge", s_merge,
                 {"epoch": k, "side": "merge", "decision": choice, "merge_alpha": cfg.merge_alpha}, step=k)
        history.append({"epoch": k, "sft": s_sft, "merge": s_merge, "decision": choice})
        print(f"[recipe] epoch {k}: decision={choice} (sft={s_sft['_aggregate']:.4f} "
              f"merge={s_merge['_aggregate']:.4f}) -> live={live}")

        for cand, tag in [(out_sft, f"epoch{k}-sft"), (out_merge, f"epoch{k}-merge")]:
            agg = (s_sft if cand == out_sft else s_merge)["_aggregate"]
            if isinstance(agg, (int, float)) and agg > best["score"]:
                best = {"score": agg, "path": cand, "tag": tag}

    print(f"[recipe] best in-loop checkpoint: {best['tag']} ({best['path']}, agg={best['score']:.4f})")

    # ---- FULL final benchmarks: original baselines vs final-SFT vs best merged ----
    final_targets = {
        f"base ({base})": base,
        f"instruct ({instruct})": instruct,
        "final-SFT": history[-1] and os.path.join(cfg.output_dir, f"epoch{cfg.num_epochs}", "sft"),
        f"best ({best['tag']})": best["path"],
    }
    finals = {}
    for name, path in final_targets.items():
        if not path:
            continue
        scores = bench(path, cfg, 0, f"final::{name}")          # limit=0 -> full
        finals[name] = scores
        log_eval(cfg, f"final-{name}", scores, {"phase": "final", "model": name})

    write_results(args.results, cfg, history, finals, tasks)
    print(f"[recipe] DONE. Publish candidate (best): {best['path']}")


if __name__ == "__main__":
    main()
