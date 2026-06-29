"""Small shared helpers: reproducibility, parameter counting, throughput."""
from __future__ import annotations

import time


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

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
