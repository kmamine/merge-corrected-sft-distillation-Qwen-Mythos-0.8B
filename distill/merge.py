"""Task-arithmetic / chat-vector merge — the "merge correction" in the recipe.

Pure-PyTorch and transparent. With the SFT task-vector `(sft - base)` and the
chat vector `(instruct - base)`:

    merged = base + chat_alpha*(instruct - base) + merge_alpha*(sft - base)

With chat_alpha=1.0 this is `instruct + merge_alpha*(sft - base)` — the distilled
SFT delta added on top of the instruct model, recovering general instruction-
following while keeping the distilled reasoning. Embedding / LM-head tensors are
excluded (they move a lot during SFT and merging them hurts; matches the
chat-vector literature). If the SFT path is a LoRA adapter it is first folded into
the base to materialise full weights.

`merge_state_dicts` is the pure core (unit-tested); `merge_models` does the I/O.
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


def merge_state_dicts(base_sd, inst_sd, sft_sd, merge_alpha=1.0, chat_alpha=1.0):
    """Pure tensor merge. Returns (merged_state_dict, n_merged, n_skipped).

    A key is merged only if it is present in all three dicts, not excluded, has
    matching shapes, and is floating point — otherwise the instruct tensor is kept.
    """
    merged, n_merged, n_skipped = {}, 0, 0
    for k, w_inst in inst_sd.items():
        if (k in base_sd and k in sft_sd
                and not excluded(k)
                and w_inst.shape == base_sd[k].shape == sft_sd[k].shape
                and torch.is_floating_point(w_inst)):
            sft_vec = sft_sd[k].float() - base_sd[k].float()
            chat_vec = w_inst.float() - base_sd[k].float()
            new = base_sd[k].float() + chat_alpha * chat_vec + merge_alpha * sft_vec
            merged[k] = new.to(w_inst.dtype)
            n_merged += 1
        else:
            merged[k] = w_inst
            n_skipped += 1
    return merged, n_merged, n_skipped


def materialise_full(base_id: str, adapter_dir: str, dtype, out: str) -> str:
    """Fold a LoRA adapter into the base, save full weights, return the dir."""
    from peft import PeftModel

    print(f"[merge] materialising full weights from adapter {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
    full = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    full.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    del base, full
    return out


def merge_models(base_id, instruct_id, sft_path, out, merge_alpha=1.0, chat_alpha=1.0,
                 dtype=torch.bfloat16):
    """Load base/instruct/sft, merge, and save to `out` with the instruct tokenizer.

    `sft_path` may be a full model dir or a LoRA adapter dir (folded in first).
    Returns `out`.
    """
    if is_adapter_dir(sft_path):
        sft_path = materialise_full(base_id, sft_path, dtype, os.path.join(out + "_sft_full"))

    print(f"[merge] loading base / instruct / sft (alpha={merge_alpha}, chat_alpha={chat_alpha}) ...")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
    instruct = AutoModelForCausalLM.from_pretrained(instruct_id, dtype=dtype)
    sft = AutoModelForCausalLM.from_pretrained(sft_path, dtype=dtype)

    merged_sd, n_merged, n_skipped = merge_state_dicts(
        base.state_dict(), instruct.state_dict(), sft.state_dict(), merge_alpha, chat_alpha)
    print(f"[merge] merged {n_merged} tensors, kept {n_skipped} from instruct")

    instruct.load_state_dict(merged_sd)
    os.makedirs(out, exist_ok=True)
    instruct.save_pretrained(out, safe_serialization=True)
    # keep the INSTRUCT tokenizer/chat template so the merged model stays chat-able
    AutoTokenizer.from_pretrained(instruct_id).save_pretrained(out)
    del base, instruct, sft
    torch.cuda.empty_cache()
    print(f"[merge] wrote merged model -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--sft", required=True, help="full SFT model dir OR a LoRA adapter dir")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale on the SFT task-vector")
    ap.add_argument("--chat_alpha", type=float, default=1.0, help="scale on the instruct vector")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()
    merge_models(args.base, args.instruct, args.sft, args.out, args.alpha, args.chat_alpha)


if __name__ == "__main__":
    main()
