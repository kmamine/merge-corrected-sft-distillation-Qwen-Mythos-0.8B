"""Stage 3a — domain evaluation: extractive QA EM/F1 on held-out aviation questions.

Compares any number of models (pass several to --models). Typical use is the
stock instruct model vs the merged model, to read off the domain gain.

Run:
    python eval_qa.py --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged \
                      --qa_path data/qa_eval.jsonl --max_samples 500
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aero_cpt.data import build_qa_inputs
from aero_cpt.utils import score_qa


def load_qa(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:n] if n and n > 0 else rows


@torch.no_grad()
def run_model(model_path, rows, max_new_tokens, batch_size, device):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                         # correct for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()

    prompts = [build_qa_inputs(tok, r["context"], r["question"]) for r in rows]
    preds = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new_tokens = gen[j, enc["input_ids"].shape[1]:]
            preds.append(tok.decode(new_tokens, skip_special_tokens=True).strip())

    del model
    torch.cuda.empty_cache()
    golds = [r["answer"] for r in rows]
    return score_qa(preds, golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--qa_path", default="data/qa_eval.jsonl")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_qa(args.qa_path, args.max_samples)
    print(f"[eval-qa] {len(rows)} questions\n")

    results = {}
    for m in args.models:
        print(f"[eval-qa] running {m} ...")
        results[m] = run_model(m, rows, args.max_new_tokens, args.batch_size, device)

    width = max(len(m) for m in results)
    print("\n" + "=" * (width + 24))
    print(f"{'model'.ljust(width)}    {'EM':>7}  {'F1':>7}")
    print("-" * (width + 24))
    for m, r in results.items():
        print(f"{m.ljust(width)}    {r['em']:>7}  {r['f1']:>7}")
    print("=" * (width + 24))


if __name__ == "__main__":
    main()
