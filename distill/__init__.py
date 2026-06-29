"""Shared library for the Mythos reasoning-distillation pipeline (merge-corrected SFT).

Modules:
    config      — SFTMergeConfig dataclass + YAML/CLI override resolution
    data        — Mythos dataset loading + chat rendering for SFT
    merge       — task-arithmetic / chat-vector merge (the "merge correction")
    recipe      — the merge-vs-continue decision policy
    eval_bench  — lm-evaluation-harness wrapper (GSM8K / MMLU / ARC-Challenge)
    tracking    — best-effort MLflow logging helpers
    utils       — seeding, parameter counts, throughput
"""
