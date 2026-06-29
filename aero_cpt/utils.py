"""Small shared helpers: reproducibility, parameter counting, throughput, QA metrics."""
from __future__ import annotations

import re
import string
import time
from collections import Counter

import numpy as np


def set_seed(seed: int) -> None:
    import random

    import torch  # local import: keeps the metric/throughput helpers usable without torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """(trainable, total). Call BEFORE FSDP sharding for meaningful numbers."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000.0
    return f"{n:.1f}P"


class ThroughputMeter:
    """Tracks tokens/second over the run (single-process count; multiply by world size)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.tokens = 0

    def update(self, n_tokens: int):
        self.tokens += int(n_tokens)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def rate(self) -> float:
        e = self.elapsed
        return self.tokens / e if e > 0 else 0.0


# ----------------------------------------------------------------------------
# SQuAD-style extractive-QA metrics (standard normalisation: lowercase, strip
# punctuation + articles, collapse whitespace). Implemented locally to avoid a
# dependency on `evaluate`.
# ----------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_qa(preds, golds):
    """preds/golds: parallel lists of strings -> dict with em and f1 (percent)."""
    assert len(preds) == len(golds)
    em = 100.0 * np.mean([exact_match(p, g) for p, g in zip(preds, golds)])
    f = 100.0 * np.mean([f1(p, g) for p, g in zip(preds, golds)])
    return {"em": round(em, 2), "f1": round(f, 2), "n": len(preds)}
