# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small, reproducible pipeline that domain-adapts a general small LLM to aviation safety and **measures whether it worked**. It continually pretrains the *base* checkpoint on incident narratives, then merges the resulting domain delta into the *instruct* checkpoint via task arithmetic so instruction-following survives. The whole run is designed to finish in 1–2h on a single GPU. The deliverable is the *dissociation result*: domain QA EM/F1 rises while general perplexity stays flat. Full rationale lives in `README (2).md`.

## Layout

The repo is an `aero_cpt`-package project; all paths below are relative to the repo root, which must be the working directory when running anything (entrypoints do `from aero_cpt...`, scripts `cd` to root, configs are referenced as `configs/...`).

```
prepare_data.py train_cpt.py merge.py eval_qa.py eval_ppl.py   # stage entrypoints (root)
aero_cpt/{__init__,config,data,utils}.py                       # shared library package
configs/{cpt,accelerate,merge_ties,merge_dare_ties}.yaml       # configs
scripts/{run_all,train_cpt}.sh                                 # pipeline drivers
tests/{test_config,test_utils,test_data}.py                    # pure-unit tests
pyproject.toml                                                 # uv project + pytest config
```

`data/` (prepared corpus + QA splits) and `outputs/` (CPT adapter, merged model) are generated artifacts, gitignored.

## Pipeline (data flows in one direction; each stage is a standalone entrypoint)

```
prepare_data.py → train_cpt.py → merge.py → eval_qa.py + eval_ppl.py
   (stage 0)        (stage 1)     (stage 2)      (stage 3a / 3b)
```

- **Stage 0 — `prepare_data.py`**: loads `Timilehin674/Aviation_QA`, writes `data/cpt_corpus.jsonl` (dedup'd narratives = CPT corpus) and `data/qa_eval.jsonl` (held-out QA = eval set). One dataset feeds both training and evaluation.
- **Stage 1 — `train_cpt.py`**: continual-pretrains the **base** model (LoRA by default). A hand-rolled Accelerate training loop (not `Trainer`), bounded by a **wall-clock cap** (`max_train_minutes`) so the run lands in budget regardless of hardware. Saves a LoRA adapter to `outputs/cpt/adapter` or a full model to `outputs/cpt/final`.
- **Stage 2 — `merge.py`**: task-arithmetic / chat-vector merge onto the **instruct** model: `merged = base + chat_alpha·(instruct − base) + alpha·(cpt − base)`. A LoRA adapter `--cpt` is first folded into the base (`merge_and_unload` → `outputs/cpt/cpt_full`).
- **Stage 3 — `eval_qa.py`** (extractive QA EM/F1, the domain gain) and **`eval_ppl.py`** (wikitext perplexity, the retention/forgetting proxy). Run both on `INSTRUCT` vs `outputs/merged`; the *gap between them is the experiment*.

`aero_cpt/data.py` (loading + token packing + QA prompt formatting), `aero_cpt/config.py` (the `CPTConfig` dataclass), and `aero_cpt/utils.py` (seeding, throughput meter, SQuAD EM/F1 metrics) are the shared library; the five top-level scripts are thin stage drivers.

## Environment & commands

Use the **`dist_train` conda env** (python 3.12; `uv` is the package manager). Activate it before doing anything, and run from the repo root:

```bash
conda activate dist_train
uv pip install -r requirements.txt        # runtime stack (mirrors pyproject [project.dependencies])
uv pip install --group dev                # pytest (dev group)
```

**Tests (run these first when changing pure logic — TDD):**

```bash
python -m pytest                          # full suite; config (pytest.ini_options) lives in pyproject.toml
python -m pytest tests/test_utils.py -k f1 -q   # single test
```

`tests/test_config.py` and `tests/test_utils.py` are torch-free; `tests/test_data.py` skips itself if torch is missing (`pytest.importorskip`). All three are CPU-only/offline (a `FakeTokenizer` stands in for HF), so they need no GPU or network.

**GPUs:** this box has two — **`cuda:0` and `cuda:1`**, but **`cuda:0` is shared with another process**, so leave headroom there (or prefer `cuda:1` for single-GPU work). Select devices explicitly with `CUDA_VISIBLE_DEVICES`:

```bash
# single GPU (avoid contending on the shared cuda:0)
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate.yaml train_cpt.py --config configs/cpt.yaml
# both GPUs (DDP) — LoRA or small full CPT
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_multi.yaml train_cpt.py --config configs/cpt.yaml
# eval scripts pick a device via torch; pin with CUDA_VISIBLE_DEVICES the same way
CUDA_VISIBLE_DEVICES=1 python eval_qa.py --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
```

**Pipeline:**

```bash
# Full pipeline (prepare → CPT → merge → eval). Env overrides: BASE, INSTRUCT, MINUTES, ALPHA
bash scripts/run_all.sh
MINUTES=60 bash scripts/run_all.sh
BASE=meta-llama/Llama-3.2-1B INSTRUCT=meta-llama/Llama-3.2-1B-Instruct bash scripts/run_all.sh

# Individual stages
python prepare_data.py
accelerate launch --config_file configs/accelerate.yaml train_cpt.py --config configs/cpt.yaml
python merge.py --base Qwen/Qwen2.5-0.5B --instruct Qwen/Qwen2.5-0.5B-Instruct \
                --cpt outputs/cpt/adapter --alpha 1.0 --out outputs/merged
python eval_qa.py  --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
python eval_ppl.py --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
```

**Config overrides** flow `dataclass defaults → YAML → --set key=value` (later wins); values are YAML-parsed, then **coerced to the dataclass field's annotated type** in `load_config` (`aero_cpt/config.py::_coerce`). Unknown YAML keys are ignored (warned), so one config file works across versions. e.g. `--set max_train_minutes=60 use_lora=false learning_rate=2e-5` works as expected — including `learning_rate=2e-5`, which `yaml.safe_load` parses as the string `"2e-5"` (its float regex needs a dot) and `_coerce` casts back to a float. Keep new `CPTConfig` fields plain scalars / `Optional[scalar]` so coercion stays well-defined.

## Invariants to preserve when editing

These are the load-bearing design decisions; changing them silently breaks the result.

- **Base vs instruct must be the same family / identical architecture.** The merge subtracts three aligned state dicts tensor-by-tensor (`merge.py`). Dense pairs only — no MoE/multimodal.
- **Merge excludes `embed_tokens`/`lm_head`/`wte`/`wpe`** (`EXCLUDE_KEYS` in `merge.py`); these shift heavily during CPT and merging them degrades quality. Tensors that don't match on key/shape/float-ness are kept from instruct as-is.
- **The merged model keeps the *instruct* tokenizer + chat template** (saved explicitly in `merge.py`), so it stays conversational. CPT trains the *base*, which has no chat template.
- **CPT packing has no prompt masking** — `pack_texts` joins docs with EOS and every token is a label (`data.py`); this is plain causal-LM pretraining, distinct from the chat-templated QA prompts (`build_qa_inputs`) used only at eval.
- **`max_train_minutes` is the single wall-clock knob** that bounds the run; the loop checks elapsed time each optimizer step and saves+exits when hit. Keep it functional when touching the training loop.
- **`replay_ratio`** mixes general-text (wikitext) blocks into CPT to limit catastrophic forgetting — the counterweight to `alpha` at merge time. The forgetting/retention story depends on both.
- **Eval generation uses `padding_side="left"`** (`eval_qa.py`) — required for correct batched decoder generation; don't change to right padding.
- **EM/F1 use local SQuAD normalization** (`utils.py`) to avoid an `evaluate` dependency — keep it standard (lowercase, strip punctuation/articles, collapse whitespace).
- **`Aviation_QA` loading has a deliberate fallback** (`data.py::_load_raw_rows`): the HF auto-loader fails the SQuAD-nested cast, so it falls back to downloading + hand-parsing raw JSON/JSONL. Preserve the fallback.

## Working practices (required in this repo)

- **Test-driven development.** Write a failing test first, then the implementation. The `pytest` harness is set up (`tests/`, config in `pyproject.toml`); the existing tests cover the pure, deterministic units (`utils.py` EM/F1 + normalization, `config.py` override resolution, `data.py` packing/flattening) and stay offline via a `FakeTokenizer`. Keep new logic testable the same way — push GPU/network to the edges so the core stays unit-testable.
- **Scientific method for changes that affect the model.** Treat hyperparameter/merge/data changes as experiments: state the hypothesis, change one variable, and report the dissociation table (QA EM/F1 vs wikitext ppl) against the instruct baseline. `alpha`/`chat_alpha`/`replay_ratio` sweeps are the intended experimental axis. Don't claim an improvement without the before/after numbers.
- **Commits: do NOT add Claude as a co-author or include any Claude/AI attribution** in commit messages or PR bodies. (This repo is not yet a git repository.)
