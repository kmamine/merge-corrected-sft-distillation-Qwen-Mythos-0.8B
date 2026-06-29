# Mythos distillation — benchmark results

- instruct `Qwen/Qwen3.5-0.8B` · dataset `WithinUsAI/claude_mythos_distilled_25k`
- E=3 · merge methods compared per epoch: `linear,ties,dare_linear,dare_ties,slerp,breadcrumbs,della` (alpha=0.5, density=0.7, drop_p=0.5, slerp_t=0.5)

## Per-epoch — SFT vs merge methods (in-loop, limit=32)

| model | gsm8k | mmlu | arc_challenge | _aggregate |
| --- | --- | --- | --- | --- |
| epoch1-sft | 0.3125 | 0.4885 | 0.3750 | 0.3920 |
| epoch1-linear | 0.4062 | 0.5005 | 0.3125 | 0.4064 |
| epoch1-ties ⬅ winner | 0.4375 | 0.5005 | 0.3125 | 0.4168 |
| epoch1-dare_linear | 0.4062 | 0.4945 | 0.3125 | 0.4044 |
| epoch1-dare_ties | 0.4375 | 0.4945 | 0.3125 | 0.4148 |
| epoch1-slerp | 0.3750 | 0.5016 | 0.2812 | 0.3860 |
| epoch1-breadcrumbs | 0.3750 | 0.5049 | 0.3125 | 0.3975 |
| epoch1-della | 0.4062 | 0.4956 | 0.2812 | 0.3944 |
| epoch2-sft ⬅ winner | 0.4062 | 0.4923 | 0.3750 | 0.4245 |
| epoch2-linear | 0.3750 | 0.5027 | 0.3125 | 0.3967 |
| epoch2-ties | 0.3750 | 0.5027 | 0.3125 | 0.3967 |
| epoch2-dare_linear | 0.3750 | 0.4973 | 0.2812 | 0.3845 |
| epoch2-dare_ties | 0.3750 | 0.4973 | 0.2812 | 0.3845 |
| epoch2-slerp | 0.3750 | 0.5005 | 0.3125 | 0.3960 |
| epoch2-breadcrumbs | 0.4062 | 0.5000 | 0.3125 | 0.4062 |
| epoch2-della | 0.3750 | 0.5005 | 0.3125 | 0.3960 |
| epoch3-sft ⬅ winner | 0.4062 | 0.4934 | 0.3750 | 0.4249 |
| epoch3-linear | 0.3750 | 0.4967 | 0.3125 | 0.3947 |
| epoch3-ties | 0.3750 | 0.4967 | 0.3125 | 0.3947 |
| epoch3-dare_linear | 0.3438 | 0.4978 | 0.3125 | 0.3847 |
| epoch3-dare_ties | 0.3438 | 0.4978 | 0.3125 | 0.3847 |
| epoch3-slerp | 0.3750 | 0.5005 | 0.3125 | 0.3960 |
| epoch3-breadcrumbs | 0.3438 | 0.5005 | 0.3125 | 0.3856 |
| epoch3-della | 0.3750 | 0.5000 | 0.3125 | 0.3958 |

## Final (full benchmarks)

| model | gsm8k | mmlu | arc_challenge | _aggregate |
| --- | --- | --- | --- | --- |
| instruct (Qwen/Qwen3.5-0.8B) | 0.3260 | 0.4813 | 0.3754 | 0.3943 |
| final-SFT | 0.3412 | 0.4727 | 0.3814 | 0.3984 |
| best (epoch3-sft) | 0.3419 | 0.4727 | 0.3814 | 0.3987 |

_Plain completion-mode lm-eval (no chat template), apples-to-apples._

