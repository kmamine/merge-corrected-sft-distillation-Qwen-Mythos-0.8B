# Mythos distillation — benchmark results

- instruct `Qwen/Qwen3.5-0.8B` · dataset `WithinUsAI/claude_mythos_distilled_25k`
- E=3 · merge_alpha=0.5 (soup: instruct + α·(sft−instruct))

## Per-epoch (in-loop, limit=64)

| model | gsm8k | mmlu | arc_challenge | _aggregate |
| --- | --- | --- | --- | --- |
| epoch1-sft | 0.3438 | 0.4959 | 0.3750 | 0.4049 |
| epoch1-merge | 0.5000 | 0.4997 | 0.3750 | 0.4582 |
| → decision epoch1: **merge** | | | |
| epoch2-sft | 0.3281 | 0.4942 | 0.3750 | 0.3991 |
| epoch2-merge | 0.4375 | 0.5008 | 0.3750 | 0.4378 |
| → decision epoch2: **merge** | | | |
| epoch3-sft | 0.3750 | 0.4910 | 0.4062 | 0.4241 |
| epoch3-merge | 0.3750 | 0.5000 | 0.3594 | 0.4115 |
| → decision epoch3: **sft** | | | |

## Final (full benchmarks)

| model | gsm8k | mmlu | arc_challenge | _aggregate |
| --- | --- | --- | --- | --- |
| instruct (Qwen/Qwen3.5-0.8B) | 0.3283 | 0.4813 | 0.3754 | 0.3950 |
| final-SFT | 0.3518 | 0.4759 | 0.3857 | 0.4044 |
| best (epoch1-merge) | 0.3632 | 0.4806 | 0.3703 | 0.4047 |

_Plain completion-mode lm-eval (no chat template), apples-to-apples._

