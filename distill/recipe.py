"""Merge-correction decision policy for the iterative distillation recipe.

After each SFT epoch we have two candidates — the plain SFT checkpoint and its
merge-corrected version (`instruct + alpha*(sft - base)`). `decide` picks which one
seeds the next epoch: merge wins ties (>=), so we only keep plain SFT when it is
strictly better, biasing toward the capability-preserving correction.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


def decide(score_sft: Optional[float], score_merge: Optional[float]) -> Tuple[str, float]:
    """Return ('merge'|'sft', winning_score).

    - both present: 'merge' if score_merge >= score_sft else 'sft' (merge wins ties);
    - one missing (eval failed): take the other;
    - both missing: continue plain SFT (decision='sft', score NaN).
    """
    sft_ok = isinstance(score_sft, (int, float)) and not (isinstance(score_sft, float) and math.isnan(score_sft))
    merge_ok = isinstance(score_merge, (int, float)) and not (isinstance(score_merge, float) and math.isnan(score_merge))
    if not sft_ok and not merge_ok:
        return "sft", float("nan")
    if not sft_ok:
        return "merge", float(score_merge)
    if not merge_ok:
        return "sft", float(score_sft)
    return ("merge", float(score_merge)) if score_merge >= score_sft else ("sft", float(score_sft))
