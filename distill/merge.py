"""Model-soup merge toward the instruct model — the "merge correction".

    merged = instruct + alpha*(sft - instruct)

`alpha` < 1 interpolates the SFT'd model back toward the original instruct (the guard
against forgetting); `alpha` = 1 is the SFT model unchanged. Embedding / LM-head
tensors are kept from instruct (merging them hurts). If the SFT path is a LoRA
adapter it is folded into the instruct first. `merge_state_dicts` is the pure core
(unit-tested); `merge_models` does the I/O.
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXCLUDE_KEYS = ("embed_tokens", "lm_head", "wte", "wpe")


def excluded(key: str) -> bool:
    return any(tok in key for tok in EXCLUDE_KEYS)


def is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def merge_state_dicts(instruct_sd, sft_sd, alpha=0.5):
    """Pure soup: merged = instruct + alpha*(sft - instruct). Returns (sd, n_merged, n_kept).

    A key is merged only if present in both, not excluded, shape-matched, and floating
    point — otherwise the instruct tensor is kept as-is.
    """
    merged, n_merged, n_kept = {}, 0, 0
    for k, w in instruct_sd.items():
        if (k in sft_sd and not excluded(k)
                and w.shape == sft_sd[k].shape and torch.is_floating_point(w)):
            new = w.float() + alpha * (sft_sd[k].float() - w.float())
            merged[k] = new.to(w.dtype)
            n_merged += 1
        else:
            merged[k] = w
            n_kept += 1
    return merged, n_merged, n_kept


def materialise_full(instruct_id, adapter_dir, dtype, out):
    """Fold a LoRA adapter (trained on instruct) into full weights; return the dir."""
    from peft import PeftModel

    print(f"[merge] materialising full weights from adapter {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(instruct_id, dtype=dtype)
    full = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    full.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    del base, full
    return out


def merge_models(instruct_id, sft_path, out, alpha=0.5, dtype=torch.bfloat16):
    """Soup the SFT checkpoint back toward the instruct model; save to `out`. Returns `out`."""
    if is_adapter_dir(sft_path):
        sft_path = materialise_full(instruct_id, sft_path, dtype, out + "_sft_full")

    print(f"[merge] soup toward instruct (alpha={alpha}): {instruct_id} <- {sft_path}")
    instruct = AutoModelForCausalLM.from_pretrained(instruct_id, dtype=dtype)
    sft = AutoModelForCausalLM.from_pretrained(sft_path, dtype=dtype)
    merged_sd, n_merged, n_kept = merge_state_dicts(instruct.state_dict(), sft.state_dict(), alpha)
    print(f"[merge] merged {n_merged} tensors, kept {n_kept} from instruct")

    instruct.load_state_dict(merged_sd)
    os.makedirs(out, exist_ok=True)
    instruct.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(instruct_id).save_pretrained(out)
    del instruct, sft
    torch.cuda.empty_cache()
    print(f"[merge] wrote merged model -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--sft", required=True, help="full SFT model dir OR a LoRA adapter dir")
    ap.add_argument("--alpha", type=float, default=0.5, help="soup weight toward SFT (1.0 == SFT)")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()
    merge_models(args.instruct, args.sft, args.out, args.alpha)


if __name__ == "__main__":
    main()
