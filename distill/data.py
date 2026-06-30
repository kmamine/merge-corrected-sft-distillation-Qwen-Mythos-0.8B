"""Data layer for the Mythos reasoning-distillation SFT.

`WithinUsAI/claude_mythos_distilled_25k` ships single-turn chat rows
(`{"messages": [{"role":"user",...},{"role":"assistant",...}], ...}`), with the
distilled reasoning living in the assistant turn. We render each row to a single
`text` field via the student tokenizer's chat template; TRL's SFTTrainer then
tokenizes `text` and trains on it (full-sequence SFT - the long assistant turn,
which carries the reasoning, dominates the loss).
"""
from __future__ import annotations

from typing import List


def valid_messages(messages) -> bool:
    """A usable row has a non-empty user turn and a non-empty assistant turn."""
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    has_user = any(m.get("role") == "user" and (m.get("content") or "").strip()
                   for m in messages)
    has_asst = any(m.get("role") == "assistant" and (m.get("content") or "").strip()
                   for m in messages)
    return has_user and has_asst


def render_chat(messages: List[dict], tokenizer) -> str:
    """Render a full conversation to text via the chat template (no generation prompt).

    The complete assistant turn is included verbatim, so this is a training target,
    not a prompt - `add_generation_prompt=False` and thinking flags are irrelevant.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def load_distill_dataset(cfg, tokenizer):
    """Load the dataset and map rows to a `{'text': ...}` dataset for SFT."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, split=cfg.dataset_split)
    ds = ds.filter(lambda ex: valid_messages(ex.get("messages")))
    if cfg.max_samples and cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    cols = ds.column_names
    ds = ds.map(lambda ex: {"text": render_chat(ex["messages"], tokenizer)},
                remove_columns=cols)
    return ds
