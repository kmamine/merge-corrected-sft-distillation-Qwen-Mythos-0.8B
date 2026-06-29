"""Typed config for the continual-pretraining (CPT) stage.

Resolution order: dataclass defaults -> YAML file -> --set key=value overrides
(later wins). Unknown YAML keys are ignored so one file can be shared across versions.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class CPTConfig:
    # ---- model ----
    base_model: str = "Qwen/Qwen2.5-0.5B"          # the *base* (not instruct) checkpoint
    attn_implementation: str = "sdpa"               # sdpa | flash_attention_2 | eager
    gradient_checkpointing: bool = False            # tiny model; only enable if OOM

    # ---- LoRA (default path: fast, fits any GPU >=16GB) ----
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"         # PEFT shorthand; or comma list q_proj,v_proj,...

    # ---- data ----
    corpus_path: str = "data/cpt_corpus.jsonl"      # produced by prepare_data.py ({"text": ...})
    block_size: int = 1024
    num_proc: int = 4
    # general-text replay to bound catastrophic forgetting (set replay_ratio=0 to disable)
    replay_ratio: float = 0.05                       # replay blocks as a fraction of domain blocks
    replay_dataset: str = "Salesforce/wikitext"
    replay_name: Optional[str] = "wikitext-2-raw-v1"
    replay_split: str = "train"

    # ---- optimisation ----
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5                              # upper bound; time cap usually stops first
    max_train_minutes: float = 90.0                 # WALL-CLOCK CAP. <=0 disables.
    max_steps: int = -1                             # optional hard step cap. <=0 disables.
    learning_rate: float = 1e-4                     # LoRA. For full CPT use ~2e-5 (see configs/).
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    seed: int = 42

    # ---- io / logging ----
    output_dir: str = "outputs/cpt"                 # adapter -> {dir}/adapter, full model -> {dir}/final
    logging_steps: int = 10
    save_steps: int = 0                             # 0 = save only at the end
    report_to: Optional[str] = None                # wandb | tensorboard | None
    run_name: str = "aero-cpt"


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
    "2e-5" (its float regex requires a dot), so `--set learning_rate=2e-5` would
    otherwise reach AdamW as a string. Coercion is idempotent for already-correct
    values, and non-numeric strings for str fields pass through unchanged.
    """
    # Optional[X] / Union[..., None]: unwrap to the non-None member, keep None.
    if typing.get_origin(hint) is typing.Union:
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
        return value  # leave odd values for the dataclass / downstream to surface
    return value


def load_config(path: Optional[str], overrides=None) -> CPTConfig:
    data = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data.update(overrides)
    known = {f.name for f in fields(CPTConfig)}
    clean = {k: v for k, v in data.items() if k in known}
    ignored = set(data) - known
    if ignored:
        print(f"[config] ignoring unknown keys: {sorted(ignored)}")
    hints = typing.get_type_hints(CPTConfig)
    clean = {k: _coerce(v, hints.get(k)) for k, v in clean.items()}
    return CPTConfig(**clean)
