# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible pipeline that **distills Claude-Mythos reasoning data into Qwen3.5-0.8B** and
**measures whether merging helps**. The novel piece is the recipe: **merge-corrected iterative SFT**
— after each SFT epoch we compare the plain SFT checkpoint against its task-arithmetic merge onto the
instruct model, and seed the next epoch with whichever scores higher on benchmarks (merging used as a
*correction* against forgetting general capability). The deliverable is the **SFT vs SFT+merge**
comparison across GSM8K / MMLU / ARC-Challenge (with the original base + instruct as baselines),
tracked in MLflow and published to the Hub.

## Layout

`distill`-package project; run everything **from the repo root** (entrypoints do `from distill...`,
configs are `configs/...`, the SFT worker is launched relative to root).

```
train_distill.py sft_worker.py publish_hub.py                  # entrypoints (root)
distill/{__init__,config,data,merge,recipe,eval_bench,tracking,utils}.py   # library
configs/{sft,accelerate,accelerate_multi}.yaml                 # configs
tests/test_{config,utils,data,tracking,recipe,merge}.py        # offline unit tests
pyproject.toml                                                 # uv project + pytest config
```

`outputs/` (checkpoints) and `data/` are gitignored; `results/` (benchmark tables) is committed.

## The recipe (data flows one direction; the orchestrator drives the loop)

```
instruct = Qwen/Qwen3.5-0.8B ;  live = instruct        # no base model in this version
for epoch k in 1..E:
    sft_k   = SFT(live, mythos_25k, 1 epoch)            # sft_worker.py via accelerate (multi-GPU)
    merge_k = instruct + α·(sft_k − instruct)           # distill/merge.py::merge_models (soup, CPU)
    bench sft_k and merge_k (GSM8K/MMLU/ARC-C, limited) # distill/eval_bench.py (lm-eval, cuda:1)
    live = merge_k if score(merge_k) >= score(sft_k) else sft_k   # distill/recipe.py::decide
    log both to MLflow ; track best
final: FULL benchmarks on instruct, final-SFT, best → results/benchmarks.md
```

- **`train_distill.py`** is a single-process orchestrator: it shells out to `sft_worker.py` via
  `accelerate launch` for the multi-GPU SFT step, then runs merge (CPU) + benchmarks (cuda:1) and the
  decision in-process. This separation is deliberate — keep the heavy DDP SFT in the subprocess and
  the orchestration single-process.
- **`distill/eval_bench.py`** wraps lm-evaluation-harness; models are evaluated in **plain completion
  mode (no chat template)** so base / instruct / SFT / merged are compared apples-to-apples.
- **`distill/merge.py::merge_state_dicts`** is the pure task-arithmetic core (unit-tested);
  `merge_models` is the I/O wrapper.

## Environment & commands

Use the **`dist_train` conda env** (python 3.12; `uv` is the package manager). Activate first, run
from the repo root:

```bash
conda activate dist_train
uv pip install -r requirements.txt        # torch, transformers, trl, lm-eval, mlflow, ...
uv pip install --group dev                # pytest
```

**Tests (run first when changing pure logic — TDD; all offline, no GPU/network):**

```bash
python -m pytest                          # config, utils, data, tracking, recipe, merge
python -m pytest tests/test_recipe.py -q  # single file
```

**GPUs:** two on this box — `cuda:0` and `cuda:1`, but **`cuda:0` is shared with another process**.
The orchestrator runs **SFT on both GPUs** (`CUDA_VISIBLE_DEVICES=0,1` + `configs/accelerate_multi.yaml`)
and **benchmarks/merge on cuda:1** (the free one). MLflow server runs separately on
`http://localhost:5000` (env `mlflow_kma`); we log to it via the **mlflow library** (client installed
in `dist_train`).

**Run the recipe:**

```bash
# smoke (cheap end-to-end): 1 epoch, ~200 examples, limit 20
MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml \
  --set num_epochs=1 max_samples=200 eval_limit=20

# full recipe
MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml

# standalone full benchmark eval of any checkpoint(s)
CUDA_VISIBLE_DEVICES=1 python -m distill.eval_bench --models Qwen/Qwen3.5-0.8B outputs/distill/epoch3/merge

# publish the chosen checkpoint (after the sweep)
python publish_hub.py --model_dir outputs/distill/epoch3/merge --repo Amine-CV/Qwen3.5-0.8B-Mythos-Distill
```

**Config overrides** flow `dataclass defaults → YAML → --set key=value` (later wins); values are
YAML-parsed then **coerced to the dataclass field's annotated type** in `load_config`
(`distill/config.py::_coerce`) — so `--set learning_rate=2e-5` reaches the optimizer as a float, not
the string `"2e-5"`. Keep new `SFTMergeConfig` fields plain scalars / `Optional[scalar]`.

## Invariants to preserve when editing

- **No base model in this version.** We SFT the **instruct** (`Qwen3.5-0.8B`) directly; the merge is a
  soup back toward that same instruct (`merged = instruct + α·(sft − instruct)`, `merge_alpha<1` is the
  forgetting guard, `=1` is plain SFT). The instruct is also the only eval baseline.
- **Merge excludes `embed_tokens`/`lm_head`/`wte`/`wpe`** (`EXCLUDE_KEYS` in `distill/merge.py`); merging
  them degrades quality. Non-matching tensors are kept from instruct as-is.
- **Merged/SFT checkpoints keep the *instruct* tokenizer + chat template** (the base may ship none;
  `sft_worker.py` renders chat and saves with the instruct tokenizer).
- **`decide` biases to the merge on ties** (`>=`) — plain SFT is kept only when strictly better. This
  is the catastrophic-forgetting guard; don't flip it silently.
- **In-loop benchmarks are `eval_limit`-capped for speed; the FINAL eval is full** (`limit<=0`). Don't
  conflate the two — the per-epoch table is a fast proxy, the final table is the result.
- **Benchmarks run in completion mode (no chat template)** for apples-to-apples across all models.
- **MLflow + HF are best-effort** (`distill/tracking.py`, `publish_hub.py`): a tracking/upload hiccup
  must never lose a training run.

## Working practices (required in this repo)

- **Test-driven development.** Failing test first. The pure cores (`config` coercion, `recipe.decide`,
  `merge.merge_state_dicts`, `eval_bench.primary_metric/aggregate`, `data.render_chat`,
  `tracking` helpers) are unit-tested offline via a `FakeTokenizer`; keep new logic testable the same
  way — push GPU/network to the edges.
- **Scientific method.** Treat recipe/hyperparameter changes as experiments: state the hypothesis,
  change one variable (`merge_alpha`, E, lr, …), and report the SFT-vs-SFT+merge benchmark table vs the
  original-model baselines. Don't claim an improvement without before/after numbers.
- **Commits: do NOT add Claude as a co-author or include any Claude/AI attribution** in commit messages
  or PR bodies.
