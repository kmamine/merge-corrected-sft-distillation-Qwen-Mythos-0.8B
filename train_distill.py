"""Merge-corrected iterative SFT distillation - the orchestrator (multi-GPU).

Per epoch: SFT one epoch (multi-GPU, both cuda:0+cuda:1) -> build one merge-corrected
candidate per method (linear / ties / dare_linear / slerp, all anchored on the original
instruct) -> benchmark the plain SFT and every merge candidate (limited) -> keep the
highest-scoring as the next epoch's start (ties favour a merge = a guard against
forgetting). Logs every candidate to MLflow; a final full benchmark pass compares the
original instruct / final-SFT / best.

    MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml

Single-process driver: it shells out to `sft_worker.py` via `accelerate launch` for
the multi-GPU SFT step (CUDA_VISIBLE_DEVICES=0,1), and runs merge (CPU) + benchmarks
(cuda:1, the free GPU) in-process - keeping the orchestration free of rank guards.
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
from distill.recipe import pick_best

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

    lines = ["# Mythos distillation - benchmark results", "",
             f"- instruct `{cfg.instruct_model}` · dataset `{cfg.dataset}`",
             f"- E={cfg.num_epochs} · merge methods compared per epoch: "
             f"`{cfg.merge_methods}` (alpha={cfg.merge_alpha}, density={cfg.ties_density}, "
             f"drop_p={cfg.dare_drop_p}, slerp_t={cfg.slerp_t})", "",
             f"## Per-epoch - SFT vs merge methods (in-loop, limit={cfg.eval_limit})", "", header, sep]
    for h in history:
        for name, scores in h["scores"].items():
            mark = " ⬅ winner" if name == h["winner"] else ""
            lines.append(row(f"epoch{h['epoch']}-{name}{mark}", scores))
    # the "road taken": which checkpoint seeded each epoch and which candidate won
    lines += ["", "## Training trajectory (the road taken)", ""]
    for h in history:
        start = h.get("start", "?")
        agg = h["scores"].get(h["winner"], {}).get("_aggregate")
        agg_s = f"{agg:.4f}" if isinstance(agg, (int, float)) else "n/a"
        lines.append(f"- epoch {h['epoch']}: start `{start}` → "
                     f"compare {{sft + {len(h['scores']) - 1} merges}} → "
                     f"**winner `{h['winner']}`** (agg {agg_s}) → seeds epoch {h['epoch'] + 1}")
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
    ap.add_argument("--no_final", action="store_true", help="skip the final full-benchmark pass (tuning)")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_kv(args.overrides))
    tasks = [t.strip() for t in cfg.eval_tasks.split(",") if t.strip()]

    instruct = cfg.instruct_model
    methods = [m.strip() for m in cfg.merge_methods.split(",") if m.strip()]
    live = instruct                                   # SFT starts from the instruct model
    best = {"score": float("-inf"), "path": None, "tag": None}
    history, created = [], []

    for k in range(1, cfg.num_epochs + 1):
        ep = os.path.join(cfg.output_dir, f"epoch{k}")
        start = os.path.basename(live) if live != instruct else "instruct"   # road taken
        out_sft = sft_one_epoch(args, live, os.path.join(ep, "sft"))
        created.append(out_sft)

        # candidate models: plain SFT + one per merge method (all vs the original instruct)
        cand_path = {"sft": out_sft}
        for m in methods:
            out_m = merge_models(instruct, out_sft, os.path.join(ep, f"merge_{m}"), method=m,
                                 alpha=cfg.merge_alpha, density=cfg.ties_density,
                                 drop_p=cfg.dare_drop_p, slerp_t=cfg.slerp_t,
                                 gamma=cfg.breadcrumbs_gamma, epsilon=cfg.della_epsilon,
                                 seed=cfg.seed)
            cand_path[m] = out_m
            created.append(out_m)

        scores = {name: bench(p, cfg, cfg.eval_limit, f"epoch{k}-{name}") for name, p in cand_path.items()}
        winner, _ = pick_best({name: s["_aggregate"] for name, s in scores.items()})
        live = cand_path[winner]

        for name, s in scores.items():
            log_eval(cfg, f"epoch{k}-{name}", s,
                     {"epoch": k, "candidate": name, "winner": name == winner}, step=k)
            agg = s["_aggregate"]
            if isinstance(agg, (int, float)) and agg > best["score"]:
                best = {"score": agg, "path": cand_path[name], "tag": f"epoch{k}-{name}"}
        history.append({"epoch": k, "start": start, "scores": scores, "winner": winner})
        print(f"[recipe] epoch {k}: start={start} winner={winner} -> live={live}")

        prune_checkpoints(created, args.keep_checkpoints, protected={live, best["path"]})

    print(f"[recipe] best checkpoint: {best['path']} (agg={best['score']:.4f})")

    if args.no_final:
        write_results(args.results, cfg, history, {}, tasks)
        print("[recipe] DONE (no final eval). Best:", best["path"])
        return

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
