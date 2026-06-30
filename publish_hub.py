"""Publish the chosen merged checkpoint to the Hugging Face Hub with a rich model card.

Uploads the model directory plus a generated README.md (model card) and the benchmark
results. The card carries the full telemetry: the per-epoch SFT-vs-SFT+merge benchmark
tables + decisions (from results/benchmarks.md), the per-epoch SFT training losses
(parsed from the run log), and a complete dump of everything logged to MLflow.

    python publish_hub.py --model_dir outputs/distill/epoch3/merge \
        --repo Amine-CV/Qwen3.5-0.8B-Mythos-Distill --results results/benchmarks.md \
        --train_log <run.log> --mlflow_experiment mythos-distill
"""
from __future__ import annotations

import argparse
import os
import re
import shutil


def mermaid_to_image(md: str, img="figures/fig_algorithm.png") -> str:
    """Replace ```mermaid fenced blocks with an image - the HF card renderer does not
    draw Mermaid (GitHub does), so the card embeds a pre-rendered PNG instead."""
    return re.sub(r"```mermaid\b.*?```", f"![Algorithm]({img})", md, flags=re.DOTALL)


def parse_train_losses(log_path):
    """Best-effort: per-epoch final train_loss + the loss trajectory from the run log."""
    finals, traj = [], []
    if not log_path or not os.path.exists(log_path):
        return finals, traj
    txt = open(log_path, errors="ignore").read()
    finals = [float(m) for m in re.findall(r"train_loss'?:\s*'?([0-9.eE+-]+)", txt)]
    traj = [float(m) for m in re.findall(r"(?<![\w'])'loss':\s*'?([0-9.eE+-]+)", txt)]
    return finals, traj


def losses_section(finals, traj) -> str:
    if not finals and not traj:
        return ""
    lines = ["## Training", ""]
    if finals:
        lines.append("| epoch | final train loss |")
        lines.append("| --- | --- |")
        for i, v in enumerate(finals, 1):
            lines.append(f"| {i} | {v:.4f} |")
        lines.append("")
    if traj:
        shown = ", ".join(f"{v:.3f}" for v in traj[:40])
        lines.append(f"SFT loss trajectory ({len(traj)} logged steps): {shown}"
                     + (" …" if len(traj) > 40 else ""))
        lines.append("")
    return "\n".join(lines)


def fetch_mlflow_section(tracking_uri, experiment) -> str:
    """Dump every run's params + metrics from the MLflow tracking server (best-effort)."""
    if not experiment:
        return ""
    try:
        import requests

        base = tracking_uri.rstrip("/") + "/api/2.0/mlflow"
        eid = requests.get(f"{base}/experiments/get-by-name",
                           params={"experiment_name": experiment}, timeout=5
                           ).json()["experiment"]["experiment_id"]
        runs = requests.post(f"{base}/runs/search",
                             json={"experiment_ids": [eid], "max_results": 200}, timeout=10
                             ).json().get("runs", [])
    except Exception as e:  # noqa: BLE001
        return f"## MLflow log\n\n_(could not reach tracking server: {e})_\n"
    if not runs:
        return ""

    def tag(r, key):
        return next((t["value"] for t in r["data"].get("tags", []) if t["key"] == key), None)

    def agg(r):
        return next((m["value"] for m in r["data"].get("metrics", []) if m["key"] == "_aggregate"), None)

    # ---- "road taken": per-epoch candidates, in order, winner marked ----
    epoch_runs = [r for r in runs if tag(r, "epoch") and tag(r, "candidate")]
    traj = []
    if epoch_runs:
        traj = ["## Training trajectory (the road taken)", ""]
        epochs = sorted({int(tag(r, "epoch")) for r in epoch_runs})
        for e in epochs:
            cands = [r for r in epoch_runs if int(tag(r, "epoch")) == e]
            cands.sort(key=lambda r: (agg(r) is not None, agg(r) or 0), reverse=True)
            win = next((tag(r, "candidate") for r in cands if tag(r, "winner") == "True"), "?")
            scored = ", ".join(f"{tag(r, 'candidate')} {agg(r):.4f}" for r in cands if agg(r) is not None)
            traj.append(f"- **epoch {e}**: {scored} → winner **{win}**")
        traj.append("")

    # ---- full dump: every run's metrics ----
    keys = []
    for r in runs:
        for m in r["data"].get("metrics", []):
            if m["key"] not in keys:
                keys.append(m["key"])
    lines = ["## MLflow log (all runs / metrics)", "",
             "| run | " + " | ".join(keys) + " |",
             "| --- " * (len(keys) + 1) + "|"]
    # oldest-first for readable epoch order
    for r in sorted(runs, key=lambda r: int(r["info"]["start_time"])):
        name = next((t["value"] for t in r["data"].get("tags", []) if t["key"] == "mlflow.runName"), "?")
        mv = {m["key"]: m["value"] for m in r["data"].get("metrics", [])}
        cells = [f"{mv[k]:.4f}" if k in mv else "" for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(traj + lines)


def build_model_card(repo, instruct, dataset, alpha, results_md, losses_md, mlflow_md) -> str:
    front = f"""---
license: apache-2.0
base_model:
- {instruct}
datasets:
- {dataset}
library_name: transformers
pipeline_tag: text-generation
tags:
- distillation
- reasoning
- merge-corrected-sft
- model-soup
- qwen3.5
---"""
    body = f"""
# {repo.split('/')[-1]}

Reasoning-distilled **Qwen3.5-0.8B** produced by **merge-corrected iterative SFT**: `{instruct}` is
supervised fine-tuned on [`{dataset}`](https://huggingface.co/datasets/{dataset}) (Claude-Mythos
distilled reasoning traces). After each epoch the SFT checkpoint is compared against a model-soup back
toward the original instruct (`merged = {instruct} + α·(SFT − {instruct})`, α={alpha}); the better of
*plain SFT* vs *SFT+merge* on GSM8K / MMLU / ARC-Challenge seeds the next epoch - using merging as a
correction against catastrophic forgetting of general capability.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", dtype="bfloat16", device_map="auto")
msgs = [{{"role": "user", "content": "Prove that there are infinitely many primes."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=512)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Benchmark results

{results_md}
{losses_md}
{mlflow_md}
## Intended use & limitations
Research demonstrator for distillation + model-merging recipes. A 0.8B model with synthetic
reasoning-style data; verify outputs for high-stakes use. Inherits the dataset's coding /
cybersecurity / math emphasis and any biases of the teacher.

## License
Model weights follow the Qwen3.5 license terms; the training data
[`{dataset}`](https://huggingface.co/datasets/{dataset}) is Apache-2.0.
"""
    return front + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="local checkpoint dir to publish")
    ap.add_argument("--repo", required=True, help="e.g. Amine-CV/Qwen3.5-0.8B-Mythos-Distill")
    ap.add_argument("--results", default="results/benchmarks.md")
    ap.add_argument("--train_log", default=None, help="run log to parse SFT losses from")
    ap.add_argument("--mlflow_experiment", default="mythos-distill")
    ap.add_argument("--mlflow_uri", default="http://localhost:5000")
    ap.add_argument("--instruct", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dataset", default="WithinUsAI/claude_mythos_distilled_25k")
    ap.add_argument("--alpha", default="0.5")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    results_md = ""
    if os.path.exists(args.results):
        results_md = mermaid_to_image(open(args.results).read())   # HF card: Mermaid -> PNG
        shutil.copy(args.results, os.path.join(args.model_dir, "benchmarks.md"))

    finals, traj = parse_train_losses(args.train_log)
    losses_md = losses_section(finals, traj)
    mlflow_md = fetch_mlflow_section(args.mlflow_uri, args.mlflow_experiment)

    card = build_model_card(args.repo, args.instruct, args.dataset, args.alpha,
                            results_md, losses_md, mlflow_md)
    with open(os.path.join(args.model_dir, "README.md"), "w") as f:
        f.write(card)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"[publish] uploading {args.model_dir} -> https://huggingface.co/{args.repo}")
    api.upload_folder(folder_path=args.model_dir, repo_id=args.repo, repo_type="model")
    print(f"[publish] done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
