"""Merge corrections toward the instruct model - several mergekit-style methods.

All methods combine the original instruct weights with the SFT checkpoint via the
task vector `delta = sft - instruct`, keeping embedding / LM-head tensors from
instruct (merging them hurts). With a single fine-tuned model the mergekit families
specialise to:

    linear       merged = instruct + alpha * delta                         (task arithmetic / soup)
    ties         merged = instruct + alpha * trim_topk(delta, density)      (TIES, single-vector)
    dare_linear  merged = instruct + alpha * dare(delta, drop_p)            (DARE drop + rescale)
    dare_ties    merged = instruct + alpha * trim_topk(dare(delta), density)
    slerp        merged = slerp(instruct, sft, slerp_t)                     (spherical interpolation)

`merge_state_dicts` is the pure core (unit-tested per method); `merge_models` is the
I/O wrapper. Running several methods per epoch lets the recipe compare them head-to-head.
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXCLUDE_KEYS = ("embed_tokens", "lm_head", "wte", "wpe")
METHODS = ("linear", "ties", "dare_linear", "dare_ties", "slerp", "breadcrumbs", "della")


def excluded(key: str) -> bool:
    return any(tok in key for tok in EXCLUDE_KEYS)


def is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def _trim_topk(delta, density):
    """Keep the top `density` fraction of entries by magnitude (per tensor); zero the rest."""
    if density >= 1.0:
        return delta
    n = delta.numel()
    k = max(1, int(n * density))
    thresh = torch.topk(delta.abs().flatten(), k, largest=True).values.min()
    return delta * (delta.abs() >= thresh)


def _dare(delta, drop_p, generator):
    """Randomly drop a fraction `drop_p` of entries and rescale survivors by 1/(1-drop_p)."""
    if drop_p <= 0.0:
        return delta
    keep = (torch.rand(delta.shape, generator=generator) >= drop_p).to(delta.dtype)
    return delta * keep / (1.0 - drop_p)


def _breadcrumbs(delta, density, gamma):
    """Keep a middle magnitude band of delta: drop the top `gamma` (outliers), then keep
    the top `density` of what remains; zero the rest (mergekit 'breadcrumbs')."""
    n = delta.numel()
    order = delta.abs().flatten().argsort(descending=True)
    n_drop_top = int(n * gamma)
    n_keep = max(1, int(n * density))
    keep_idx = order[n_drop_top:n_drop_top + n_keep]
    mask = torch.zeros(n, dtype=delta.dtype)
    mask[keep_idx] = 1.0
    return delta * mask.reshape(delta.shape)


def _della(delta, base_p, epsilon, generator):
    """Magnitude-ranked probabilistic drop (mergekit 'della'): smaller-magnitude entries get
    a higher drop probability (base_p + epsilon around the median rank), then rescale."""
    flat = delta.abs().flatten()
    ranks = flat.argsort().argsort().float() / max(1, flat.numel() - 1)   # 0=smallest .. 1=largest
    p = (base_p + epsilon * (0.5 - ranks.reshape(delta.shape))).clamp(0.0, 0.99)
    keep = (torch.rand(delta.shape, generator=generator) >= p).to(delta.dtype)
    return delta * keep / (1.0 - p)


def _slerp(w, s, t):
    """Per-tensor spherical interpolation between w (t=0) and s (t=1); lerp when collinear."""
    wf, sf = w.flatten().float(), s.flatten().float()
    wn, sn = wf / (wf.norm() + 1e-8), sf / (sf.norm() + 1e-8)
    dot = torch.dot(wn, sn).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        out = (1 - t) * wf + t * sf
    else:
        out = (torch.sin((1 - t) * omega) / so) * wf + (torch.sin(t * omega) / so) * sf
    return out.reshape(w.shape)


def _merge_tensor(w, s, method, alpha, density, drop_p, slerp_t, gamma, epsilon, generator):
    if method == "slerp":
        return _slerp(w, s, slerp_t)
    delta = s.float() - w.float()
    if method in ("dare_linear", "dare_ties"):
        delta = _dare(delta, drop_p, generator)
    if method in ("ties", "dare_ties"):
        delta = _trim_topk(delta, density)
    if method == "breadcrumbs":
        delta = _breadcrumbs(delta, density, gamma)
    if method == "della":
        delta = _della(delta, drop_p, epsilon, generator)
    return w.float() + alpha * delta


def merge_state_dicts(instruct_sd, sft_sd, method="linear", alpha=0.5, density=0.7,
                      drop_p=0.5, slerp_t=0.5, gamma=0.1, epsilon=0.1, seed=0):
    """Pure tensor merge by `method`. Returns (merged_state_dict, n_merged, n_kept).

    A key is merged only if present in both, not excluded, shape-matched, and floating
    point - otherwise the instruct tensor is kept. DARE randomness is seeded for repro.
    """
    if method not in METHODS:
        raise ValueError(f"unknown merge method {method!r}; choose from {METHODS}")
    generator = torch.Generator().manual_seed(seed)
    merged, n_merged, n_kept = {}, 0, 0
    for k, w in instruct_sd.items():
        if (k in sft_sd and not excluded(k)
                and w.shape == sft_sd[k].shape and torch.is_floating_point(w)):
            merged[k] = _merge_tensor(w, sft_sd[k], method, alpha, density, drop_p, slerp_t,
                                      gamma, epsilon, generator).to(w.dtype)
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


def merge_models(instruct_id, sft_path, out, method="linear", alpha=0.5, density=0.7,
                 drop_p=0.5, slerp_t=0.5, gamma=0.1, epsilon=0.1, seed=0, dtype=torch.bfloat16):
    """Merge the SFT checkpoint toward the instruct model by `method`; save to `out`."""
    if is_adapter_dir(sft_path):
        sft_path = materialise_full(instruct_id, sft_path, dtype, out + "_sft_full")

    print(f"[merge] {method} (alpha={alpha} density={density} drop_p={drop_p} t={slerp_t} "
          f"gamma={gamma} eps={epsilon}): {instruct_id} <- {sft_path}")
    instruct = AutoModelForCausalLM.from_pretrained(instruct_id, dtype=dtype)
    sft = AutoModelForCausalLM.from_pretrained(sft_path, dtype=dtype)
    merged_sd, n_merged, n_kept = merge_state_dicts(
        instruct.state_dict(), sft.state_dict(), method, alpha, density, drop_p, slerp_t,
        gamma, epsilon, seed)
    print(f"[merge] {method}: merged {n_merged} tensors, kept {n_kept} from instruct")

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
    ap.add_argument("--method", default="linear", choices=METHODS)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--drop_p", type=float, default=0.5)
    ap.add_argument("--slerp_t", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()
    merge_models(args.instruct, args.sft, args.out, args.method, args.alpha,
                 args.density, args.drop_p, args.slerp_t, args.gamma, args.epsilon)


if __name__ == "__main__":
    main()
