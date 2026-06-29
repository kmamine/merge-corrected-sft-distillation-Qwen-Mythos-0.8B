"""Generate result figures from the committed artifacts.

Reads results/confirm_stderr.json (final eval ± stderr), results/benchmarks_mm.md
(per-epoch SFT-vs-7-merge aggregates), and results/loss_trajectory.json (SFT loss),
and writes PNGs to results/figures/. Run from repo root:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
TASKS = ["gsm8k", "mmlu", "arc_challenge"]


def fig_final_benchmarks():
    data = json.load(open("results/confirm_stderr.json"))
    labels = {"original (Qwen3.5-0.8B, pre-training)": "original\n(pre-training)",
              "run1-best epoch1/merge (lr1e-5, linear soup)": "SFT+merge\n(epoch1/merge)",
              "run2-best epoch3/sft (lr5e-6, multi-method)": "SFT\n(epoch3)"}
    groups = TASKS + ["aggregate"]
    models = list(labels)
    import math
    x = range(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        s = data[m]
        vals, errs = [], []
        for t in TASKS:
            vals.append(s[t]); errs.append(s.get(f"{t}_stderr", 0) or 0)
        agg = s["_aggregate"]
        agg_se = math.sqrt(sum((s.get(f"{t}_stderr", 0) or 0) ** 2 for t in TASKS)) / len(TASKS)
        vals.append(agg); errs.append(agg_se)
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, yerr=errs, capsize=3, label=labels[m])
    ax.set_xticks(list(x)); ax.set_xticklabels(["GSM8K", "MMLU", "ARC-C", "Aggregate"])
    ax.set_ylabel("score"); ax.set_ylim(0, 0.6)
    ax.set_title("Final benchmarks (full eval, ±1 SE) — original vs SFT vs SFT+merge")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_final_benchmarks.png", dpi=150); plt.close(fig)


def _parse_per_epoch():
    """Parse {candidate: {epoch: aggregate}} and winners from results/benchmarks_mm.md."""
    rows = re.findall(r"\|\s*epoch(\d+)-([a-z_]+)(\s*⬅ winner)?\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|",
                      open("results/benchmarks_mm.md").read())
    agg, winners = {}, {}
    for ep, name, win, a in rows:
        agg.setdefault(name, {})[int(ep)] = float(a)
        if win:
            winners[int(ep)] = name
    return agg, winners


def fig_method_comparison():
    agg, winners = _parse_per_epoch()
    epochs = sorted({e for d in agg.values() for e in d})
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in sorted(agg.items()):
        ys = [d.get(e) for e in epochs]
        style = dict(lw=3, color="black", marker="o", zorder=5) if name == "sft" else dict(lw=1.5, marker="o", alpha=0.8)
        ax.plot(epochs, ys, label=name, **style)
    for e, w in winners.items():
        ax.annotate(f"win: {w}", (e, agg[w][e]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="darkred")
    ax.set_xticks(epochs); ax.set_xlabel("epoch"); ax.set_ylabel("aggregate (in-loop, limit 32)")
    ax.set_title("Per-epoch: SFT vs 7 merge methods (winner seeds next epoch)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_method_comparison.png", dpi=150); plt.close(fig)


def fig_loss_collapse():
    losses = json.load(open("results/loss_trajectory.json"))["step_losses"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(losses) + 1), losses, lw=1.5, color="tab:red")
    ax.set_yscale("log"); ax.set_xlabel("logging step (every 10 optim steps)")
    ax.set_ylabel("SFT train loss (log)")
    n = len(losses)
    for b in (n // 3, 2 * n // 3):                      # epoch boundaries (3 SFT epochs)
        ax.axvline(b + 0.5, ls="--", color="gray", alpha=0.6)
    ax.set_title("SFT loss collapses to ~0.02 (over-memorization of the templated data)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_loss_collapse.png", dpi=150); plt.close(fig)


def fig_algorithm():
    """Flowchart of the merge-corrected iterative SFT recipe (PNG for the HF card, where
    Mermaid does not render; GitHub keeps the live Mermaid)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = [
        ("instruct = Qwen3.5-0.8B      live ← instruct", "#d7e8ff"),
        ("SFT(live) one epoch on mythos_25k  →  sft_k   (multi-GPU DDP)", "#e7f5e9"),
        ("merge candidate per method onto instruct   (δ = sft_k − instruct, excl. embed/lm_head)\n"
         "linear · ties · dare_linear · dare_ties · slerp · breadcrumbs · della", "#fff2e0"),
        ("benchmark sft_k AND every merge   ·   GSM8K / MMLU / ARC-C   (in-loop, limited)", "#fff2e0"),
        ("pick best = argmax aggregate  (ties → prefer a merge)\n"
         "live ← winner   ·   track global best   ·   prune ≤ 5 checkpoints", "#fff2e0"),
        ("k = E  →  FULL benchmarks:   original · final-SFT · best", "#e7f5e9"),
        ("publish best  →  Hugging Face   (model card + report)", "#d7e8ff"),
    ]
    n = len(boxes)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ys = [0.92 - i * 0.135 for i in range(n)]
    bw, bh, cx = 0.74, 0.052, 0.42
    for (text, color), y in zip(boxes, ys):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, y - bh), bw, 2 * bh,
                                    boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc=color, ec="#555", lw=1.2))
        ax.text(cx, y, text, ha="center", va="center", fontsize=8.5)
    for i in range(n - 1):
        ax.annotate("", xy=(cx, ys[i + 1] + bh), xytext=(cx, ys[i] - bh),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
    ax.text(cx + 0.02, (ys[4] + ys[5]) / 2, "k = E", ha="left", va="center", fontsize=8.5, color="#444")
    # loop-back: pick-best (idx 4) → SFT (idx 1)
    ax.add_patch(FancyArrowPatch((cx + bw / 2, ys[4]), (cx + bw / 2, ys[1]),
                                 connectionstyle="arc3,rad=-0.9", arrowstyle="-|>",
                                 color="#1a73e8", lw=1.7, mutation_scale=16))
    ax.text(0.95, (ys[1] + ys[4]) / 2, "next epoch\n(k < E)", ha="center", va="center",
            fontsize=8.5, color="#1a73e8")
    ax.set_title("Merge-corrected iterative SFT distillation", fontsize=13, weight="bold")
    fig.savefig(f"{OUT}/fig_algorithm.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_algorithm()
    fig_final_benchmarks()
    fig_method_comparison()
    fig_loss_collapse()
    print(f"wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
