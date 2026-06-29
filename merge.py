"""Stage 2 — merge the domain knowledge into the post-trained (instruct) model.

Pure-PyTorch implementation of the chat-vector / task-arithmetic merge so the
operation is fully transparent:

    merged = instruct + chat_alpha*(instruct - base)*0   ... see below

Concretely, with `domain = cpt - base` (the CPT task vector) and the standard
chat-vector recovery of instruction-following:

    merged = base + chat_alpha*(instruct - base) + alpha*(cpt - base)

With chat_alpha=1.0 (default) this is `instruct + alpha*(cpt - base)` — i.e. the
domain task-vector added on top of the instruct model. The embedding and LM-head
tensors are excluded from the merge (they move a lot during CPT and merging them
hurts; this matches the chat-vector papers).

If `--cpt` points at a LoRA adapter dir, it is first merged into the base to
materialise full CPT weights.

For sign-consensus / trimmed merges (TIES, DARE-TIES) use mergekit with the YAMLs
in configs/ instead — see README.

Run:
    python merge.py --base Qwen/Qwen2.5-0.5B \
                    --instruct Qwen/Qwen2.5-0.5B-Instruct \
                    --cpt outputs/cpt/adapter --alpha 1.0 --out outputs/merged
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXCLUDE_KEYS = ("embed_tokens", "lm_head", "wte", "wpe")


def is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def materialise_cpt(base_id: str, adapter_dir: str, dtype) -> str:
    """Merge a LoRA adapter into the base, save full weights, return the dir."""
    from peft import PeftModel

    print(f"[merge] materialising CPT model from adapter {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = os.path.join(os.path.dirname(adapter_dir.rstrip("/")) or ".", "cpt_full")
    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    del base, merged
    return out


def excluded(key: str) -> bool:
    return any(tok in key for tok in EXCLUDE_KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--cpt", required=True, help="full CPT model dir OR a LoRA adapter dir")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale on the domain task-vector")
    ap.add_argument("--chat_alpha", type=float, default=1.0,
                    help="scale on the instruct vector; 1.0 == add domain delta onto instruct")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()

    dtype = torch.bfloat16
    cpt_dir = materialise_cpt(args.base, args.cpt, dtype) if is_adapter_dir(args.cpt) else args.cpt

    print("[merge] loading base / instruct / cpt state dicts ...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    instruct = AutoModelForCausalLM.from_pretrained(args.instruct, torch_dtype=dtype)
    cpt = AutoModelForCausalLM.from_pretrained(cpt_dir, torch_dtype=dtype)

    base_sd = base.state_dict()
    inst_sd = instruct.state_dict()
    cpt_sd = cpt.state_dict()

    merged_sd = {}
    n_merged, n_skipped = 0, 0
    for k, w_inst in inst_sd.items():
        if (k in base_sd and k in cpt_sd
                and not excluded(k)
                and w_inst.shape == base_sd[k].shape == cpt_sd[k].shape
                and torch.is_floating_point(w_inst)):
            domain = cpt_sd[k].float() - base_sd[k].float()
            chat = w_inst.float() - base_sd[k].float()
            new = base_sd[k].float() + args.chat_alpha * chat + args.alpha * domain
            merged_sd[k] = new.to(w_inst.dtype)
            n_merged += 1
        else:
            merged_sd[k] = w_inst                    # keep instruct weights as-is
            n_skipped += 1

    print(f"[merge] merged {n_merged} tensors, kept {n_skipped} from instruct "
          f"(alpha={args.alpha}, chat_alpha={args.chat_alpha})")

    instruct.load_state_dict(merged_sd)
    os.makedirs(args.out, exist_ok=True)
    instruct.save_pretrained(args.out, safe_serialization=True)
    # keep the INSTRUCT tokenizer/chat-template so the merged model stays chat-able
    AutoTokenizer.from_pretrained(args.instruct).save_pretrained(args.out)
    print(f"[merge] wrote merged model -> {args.out}")


if __name__ == "__main__":
    main()
