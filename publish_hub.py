"""Publish the chosen merged checkpoint to the Hugging Face Hub with a model card.

Uploads the model directory plus a generated README.md (model card) and the
benchmark results. Run after the sweep, once the checkpoint to publish is chosen:

    python publish_hub.py --model_dir outputs/distill/epoch3/merge \
        --repo Amine-CV/Qwen3.5-0.8B-Mythos-Distill --results results/benchmarks.md
"""
from __future__ import annotations

import argparse
import os
import shutil


def build_model_card(repo, instruct, dataset, alpha, results_md: str) -> str:
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
*plain SFT* vs *SFT+merge* on GSM8K / MMLU / ARC-Challenge seeds the next epoch — using merging as a
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
    ap.add_argument("--instruct", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dataset", default="WithinUsAI/claude_mythos_distilled_25k")
    ap.add_argument("--alpha", default="0.5")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    results_md = ""
    if os.path.exists(args.results):
        with open(args.results) as f:
            results_md = f.read()
        shutil.copy(args.results, os.path.join(args.model_dir, "benchmarks.md"))

    card = build_model_card(args.repo, args.instruct, args.dataset, args.alpha, results_md)
    with open(os.path.join(args.model_dir, "README.md"), "w") as f:
        f.write(card)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"[publish] uploading {args.model_dir} -> https://huggingface.co/{args.repo}")
    api.upload_folder(folder_path=args.model_dir, repo_id=args.repo, repo_type="model")
    print(f"[publish] done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
