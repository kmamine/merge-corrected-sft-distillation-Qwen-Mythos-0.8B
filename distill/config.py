"""Typed config for the merge-corrected SFT distillation recipe.

Resolution order: dataclass defaults -> YAML file -> --set key=value overrides
(later wins). Values are YAML-parsed then coerced to each field's annotated type
(so e.g. --set learning_rate=2e-5 reaches the optimizer as a float, not a string).
Unknown YAML keys are ignored so one file can be shared across versions.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class SFTMergeConfig:
    # ---- model (SFT this instruct checkpoint; the merge soups back toward it) ----
    instruct_model: str = "Qwen/Qwen3.5-0.8B"       # SFT start + frozen merge anchor + baseline
    attn_implementation: str = "sdpa"               # sdpa | flash_attention_2 | eager
    gradient_checkpointing: bool = True             # on by default (cuda:0 is shared on this box)

    # ---- distillation data ----
    dataset: str = "WithinUsAI/claude_mythos_distilled_25k"
    dataset_split: str = "train"
    max_samples: int = -1                           # cap rows (debug); <=0 = all
    max_seq_len: int = 2048
    packing: bool = False

    # ---- LoRA (fallback for budget; full SFT is the default for a clean merge delta) ----
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # ---- SFT optimisation ----
    num_epochs: int = 3                             # E = number of merge-corrected rounds
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42

    # ---- merge correction(s): compared head-to-head per epoch (mergekit-style methods) ----
    merge_methods: str = "linear,ties,dare_linear,slerp"   # comma list; see distill/merge.py::METHODS
    merge_alpha: float = 0.5                        # scale on the task vector (linear/ties/dare); 1.0 == sft
    ties_density: float = 0.7                       # TIES: keep top fraction of |delta| by magnitude
    dare_drop_p: float = 0.5                        # DARE: fraction of delta entries dropped (then rescaled)
    slerp_t: float = 0.5                            # SLERP: interpolation factor instruct(0)->sft(1)

    # ---- benchmarks (lm-evaluation-harness) ----
    eval_tasks: str = "gsm8k,mmlu,arc_challenge"    # comma list
    eval_limit: int = 200                           # in-loop per-task cap; <=0 = full (final eval)
    eval_batch_size: int = 8

    # ---- io / logging ----
    output_dir: str = "outputs/distill"
    logging_steps: int = 10

    # ---- mlflow ----
    mlflow_experiment: Optional[str] = "mythos-distill"
    mlflow_tracking_uri: Optional[str] = "http://localhost:5000"
    run_name: str = "merge-corrected-sft"


def parse_kv(items):
    """Parse ['lr=1e-4', 'use_lora=false'] into a dict with YAML-typed values."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(val)
    return out


def _coerce(value, hint):
    """Coerce a parsed value to the dataclass field's annotated type.

    Fixes the PyYAML gotcha where `yaml.safe_load("2e-5")` returns the string
    "2e-5" (its float regex requires a dot). Idempotent for already-correct
    values; non-numeric strings for str fields pass through unchanged.
    """
    if typing.get_origin(hint) is typing.Union:        # Optional[X] / Union[..., None]
        if value is None:
            return None
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = non_none[0] if len(non_none) == 1 else None
    if hint is None or value is None:
        return value
    try:
        if hint is bool:
            return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}
        if hint is int:
            return int(value)
        if hint is float:
            return float(value)
        if hint is str:
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def load_config(path: Optional[str], overrides=None) -> SFTMergeConfig:
    data = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data.update(overrides)
    known = {f.name for f in fields(SFTMergeConfig)}
    clean = {k: v for k, v in data.items() if k in known}
    ignored = set(data) - known
    if ignored:
        print(f"[config] ignoring unknown keys: {sorted(ignored)}")
    hints = typing.get_type_hints(SFTMergeConfig)
    clean = {k: _coerce(v, hints.get(k)) for k, v in clean.items()}
    return SFTMergeConfig(**clean)
