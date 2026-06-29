"""Stage 3b — retention check: perplexity on general text (wikitext).

A cheap proxy for catastrophic forgetting. Compare the stock instruct model
against the merged model: if the merge worked, general perplexity stays close
to the instruct baseline while QA EM/F1 (eval_qa.py) goes up. That gap is the
"dissociation" result.

Run:
    python eval_ppl.py --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
"""
from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def perplexity(model_path, dataset, name, split, max_tokens, block_size, device):
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()

    ds = load_dataset(dataset, name, split=split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:max_tokens]
    n_blocks = max(1, len(ids) // block_size)

    nll_sum, tok_count = 0.0, 0
    for i in range(n_blocks):
        block = ids[i * block_size:(i + 1) * block_size].unsqueeze(0).to(device)
        out = model(block, labels=block)
        # out.loss is mean NLL over (block_size-1) targets
        n = block.shape[1] - 1
        nll_sum += out.loss.item() * n
        tok_count += n

    del model
    torch.cuda.empty_cache()
    return math.exp(nll_sum / tok_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--name", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_tokens", type=int, default=100_000)
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for m in args.models:
        print(f"[eval-ppl] {m} ...")
        results[m] = perplexity(m, args.dataset, args.name, args.split,
                                args.max_tokens, args.block_size, device)

    width = max(len(m) for m in results)
    print("\n" + "=" * (width + 18))
    print(f"{'model'.ljust(width)}    {'wikitext ppl':>12}")
    print("-" * (width + 18))
    for m, ppl in results.items():
        print(f"{m.ljust(width)}    {ppl:>12.2f}")
    print("=" * (width + 18))


if __name__ == "__main__":
    main()
