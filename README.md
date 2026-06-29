# Aviation-Safety Domain Adaptation: Continual Pretraining + Model Merging

A small, reproducible pipeline that turns a general small LLM into an aviation-safety specialist, and — critically — **measures whether it worked**. It continually pretrains the *base* checkpoint on real incident narratives, then merges the resulting domain knowledge back into the *post-trained* (instruct) model so instruction-following survives. The whole training run is built to finish in **1–2 hours on a single GPU**.

The point isn't the method (CPT + merge is established). The point is the artifact: an open, end-to-end recipe on a **cleanly licensed public corpus**, with a domain eval and a forgetting check, on a domain where the only prior open models are encoder-only.

## The idea in one diagram

```
  base model ──(continual pretraining on aviation narratives)──► CPT model
      │                                                              │
      │  domain task-vector  Δ = (CPT − base)                        │
      ▼                                                              ▼
  instruct model ──────────────(merge: instruct + α·Δ)────────────► merged model
                                                                       │
                                              evaluate ◄───────────────┘
                                   domain QA (EM/F1)  ×  general retention (ppl)
```

## Data

[`Timilehin674/Aviation_QA`](https://huggingface.co/datasets/Timilehin674/Aviation_QA) — built from NTSB and NASA ASRS reports, ~13.5k unique safety narratives and ~350k QA pairs across 37 event types. Permissively licensed (verify MIT vs CC-BY-4.0 before redistributing weights). One dataset serves both stages:

- **narratives → CPT corpus** (the unlabeled domain text)
- **QA pairs → evaluation** (extractive EM/F1)

`prepare_data.py` handles the one wrinkle — the dataset ships as SQuAD-nested JSON and the HF auto-loader can fail the cast, so the loader falls back to downloading and parsing the raw file.

## Setup

```bash
conda activate dist_train          # python 3.12 + uv
uv pip install -r requirements.txt # runtime stack
uv pip install --group dev         # pytest, for running the tests
```

(Plain `pip install -r requirements.txt` works too.) `torch>=2.6`, `transformers>=4.50`, `accelerate>=1.0`, `datasets`, `peft`. The default merge is pure PyTorch — `mergekit` is only needed for the optional TIES / DARE-TIES path.

Run the unit tests with `python -m pytest` (config in `pyproject.toml`).

## Quickstart

End to end (prepare → CPT → merge → eval):

```bash
bash scripts/run_all.sh
```

For a strict 1-hour run, or a different model:

```bash
MINUTES=60 bash scripts/run_all.sh
BASE=meta-llama/Llama-3.2-1B INSTRUCT=meta-llama/Llama-3.2-1B-Instruct bash scripts/run_all.sh
```

Or run the stages individually:

```bash
python prepare_data.py
accelerate launch --config_file configs/accelerate.yaml train_cpt.py --config configs/cpt.yaml
python merge.py --base Qwen/Qwen2.5-0.5B --instruct Qwen/Qwen2.5-0.5B-Instruct \
                --cpt outputs/cpt/adapter --alpha 1.0 --out outputs/merged
python eval_qa.py  --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
python eval_ppl.py --models Qwen/Qwen2.5-0.5B-Instruct outputs/merged
```

## Hitting the 1–2 hour budget

Two mechanisms keep the run inside the budget on any hardware:

1. **LoRA continual pretraining (default).** Only adapter weights train, so memory is low (fits any GPU ≥16GB) and throughput is high. Set `use_lora: false` for full CPT if you have A100-class hardware and want a cleaner full-weight delta for merging (use `learning_rate: ~2e-5` for full CPT).
2. **A wall-clock cap.** `max_train_minutes` in `configs/cpt.yaml` stops training when the budget is hit, saves, and exits — so wall-clock is bounded regardless of GPU speed or corpus size.

This corpus is small on purpose (~5–10M tokens), so 1–2h means **several epochs**, not one rushed pass. To size a run by hand: measure tokens/sec from the first ~50 logged steps, then `tokens ≈ tok_s × seconds`. Rough single-GPU guide for the 0.5B model: an A100 covers the corpus several times over in 90 min; a 24GB consumer card 2–3×; a T4 about one pass (use LoRA). Multi-GPU multiplies this — use the FSDP2 accelerate config from the companion SFT repo and pass `--num_processes N` for full-CPT.

## Why a dense base+instruct pair (and not a multimodal/MoE model)

The merge subtracts weight vectors across three checkpoints, so it needs a **base and an instruct model from the same family with identical architecture**. Dense models like `Qwen/Qwen2.5-0.5B`(+`-Instruct`) or `meta-llama/Llama-3.2-1B`(+`-Instruct`) are ideal. Multimodal or MoE models add vision parameters and routing that complicate both fast CPT and clean merging, so they're a poor fit for a 1–2h showcase.

## How the merge works

`merge.py` is a transparent, pure-PyTorch implementation of the **chat-vector / task-arithmetic** merge:

```
merged = base + chat_alpha·(instruct − base) + alpha·(CPT − base)
```

With `chat_alpha = 1.0` (default) this reduces to `instruct + alpha·(CPT − base)`: the domain task-vector added on top of the instruct model. Embedding and LM-head tensors are **excluded** from the merge — they shift a lot during CPT and merging them degrades quality (consistent with the chat-vector literature). The merged model keeps the **instruct tokenizer and chat template**, so it stays conversational.

If `--cpt` points at a LoRA adapter, it's first folded into the base (`merge_and_unload`) to materialise full CPT weights at `outputs/cpt/cpt_full`.

Sweep `--alpha` (and optionally `--chat_alpha`) to trade domain gain against general retention — that sweep is the experiment.

**Stronger merges.** When CPT moves the weights far enough to cause destructive interference, switch from task-arithmetic to sign-consensus methods via mergekit:

```bash
pip install mergekit
mergekit-yaml configs/merge_ties.yaml      outputs/merged_ties  --cuda
mergekit-yaml configs/merge_dare_ties.yaml outputs/merged_dare  --cuda
```

(Edit the model paths in those YAMLs first; they expect the full CPT model at `outputs/cpt/cpt_full`.)

## Evaluation — the result that matters

The headline is a **dissociation table**, run on the stock instruct model vs. the merged model:

- `eval_qa.py` — extractive-QA **EM/F1** on held-out aviation questions (the *domain gain*).
- `eval_ppl.py` — **wikitext perplexity** as a cheap *retention / catastrophic-forgetting* proxy.

A successful merge shows domain EM/F1 rising over the instruct baseline **while** general perplexity stays close to it. If domain goes up but perplexity blows up, you've forgotten too much — lower `alpha`, add more replay (`replay_ratio`), or switch to DARE-TIES.

## Project structure

```
aero-cpt-merge/
├── prepare_data.py        # stage 0: Aviation_QA -> CPT corpus + QA eval splits
├── train_cpt.py           # stage 1: LoRA/full CPT of the base, wall-clock capped
├── merge.py               # stage 2: chat-vector / task-arithmetic merge onto instruct
├── eval_qa.py             # stage 3a: domain QA EM/F1
├── eval_ppl.py            # stage 3b: general-text perplexity (retention)
├── pyproject.toml         # uv project (deps) + pytest config
├── aero_cpt/
│   ├── __init__.py
│   ├── config.py          # CPT config dataclass + YAML/CLI overrides
│   ├── data.py            # robust dataset loading, token packing, replay, QA prompts
│   └── utils.py           # seed, param counts, throughput, SQuAD EM/F1
├── configs/
│   ├── cpt.yaml           # CPT hyperparameters + max_train_minutes
│   ├── accelerate.yaml    # single-GPU Accelerate config
│   ├── accelerate_multi.yaml  # two-GPU DDP (cuda:0 + cuda:1)
│   ├── merge_ties.yaml    # optional mergekit TIES
│   └── merge_dare_ties.yaml
├── scripts/
│   ├── run_all.sh         # full pipeline
│   └── train_cpt.sh       # CPT only
└── tests/                 # offline pure-unit tests (pytest)
    ├── test_config.py
    ├── test_utils.py
    └── test_data.py
```

## Troubleshooting

- **`load_dataset` fails on Aviation_QA** — expected; the loader auto-falls back to raw-file parsing. If even that fails, download the JSON from the dataset's Files tab and point the parser at it.
- **CUDA OOM** — keep `use_lora: true`, lower `per_device_batch_size`, or set `gradient_checkpointing: true`.
- **`from_pretrained` rejects `torch_dtype`** — only on bleeding-edge transformers where it was renamed; change `torch_dtype=` to `dtype=` in the four scripts.
- **`all-linear` LoRA target errors** — replace `lora_target_modules` with an explicit list: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- **Merged model won't chat** — confirm `merge.py` saved the instruct tokenizer (it does by default); the chat template comes from there.
- **Run finishes too fast / too slow** — adjust `max_train_minutes`; it's the single knob that bounds wall-clock.

## License & credits

Pipeline code: MIT. The Aviation_QA dataset and the underlying NTSB/ASRS reports retain their own licenses (verify before redistributing weights). Built with Hugging Face Transformers, Accelerate, Datasets, PEFT, and (optionally) Arcee mergekit.
