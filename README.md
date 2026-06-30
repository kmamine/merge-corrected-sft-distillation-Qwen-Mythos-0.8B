# Reasoning distillation into Qwen3.5-0.8B via merge-corrected SFT

Distill the [`WithinUsAI/claude_mythos_distilled_25k`](https://huggingface.co/datasets/WithinUsAI/claude_mythos_distilled_25k)
reasoning data into **Qwen3.5-0.8B**, and **measure whether merging helps**. The recipe interleaves
SFT with model merging: after each epoch, the plain SFT checkpoint is compared against its
task-arithmetic merge onto the instruct model, and the better one (on benchmarks) seeds the next
epoch - using merging as a *correction* against forgetting general capability.

## Model

**🤗 [`Amine-CV/Qwen3.5-0.8B-Mythos-Distill`](https://huggingface.co/Amine-CV/Qwen3.5-0.8B-Mythos-Distill)** -
the published checkpoint (the epoch-1 merge correction; GSM8K +3.7 pts ≈2σ over the base instruct,
MMLU retained). Full benchmark tables, training trajectory, and ±stderr results are in the model card
and [`results/REPORT.md`](results/REPORT.md).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Amine-CV/Qwen3.5-0.8B-Mythos-Distill")
model = AutoModelForCausalLM.from_pretrained("Amine-CV/Qwen3.5-0.8B-Mythos-Distill",
                                             dtype="bfloat16", device_map="auto")
msgs = [{"role": "user", "content": "Prove there are infinitely many primes."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=512)[0][ids.shape[1]:], skip_special_tokens=True))
```

## The idea

```
instruct = Qwen/Qwen3.5-0.8B ;  live = instruct
for epoch k in 1..E:
    sft_k   = SFT(live, mythos_25k, 1 epoch)
    merge_k = instruct + α·(sft_k − instruct)        # model soup back toward instruct, excl. embed/lm_head
    score sft_k and merge_k on GSM8K / MMLU / ARC-Challenge
    live    = merge_k if score(merge_k) >= score(sft_k) else sft_k     # "recorrect" vs "continue"
final: FULL benchmarks on the original instruct, final-SFT, and the best checkpoint
```

```mermaid
flowchart TD
    A["instruct = Qwen3.5-0.8B · live = instruct"] --> B{"for epoch k = 1..E"}
    B --> C["SFT(live) one epoch on mythos_25k<br/>multi-GPU DDP → sft_k"]
    C --> D["build a merge candidate per method onto instruct<br/>δ = sft_k − instruct (exclude embed / lm_head)<br/>linear · ties · dare_linear · dare_ties · slerp · breadcrumbs · della"]
    D --> E["benchmark sft_k AND every merge<br/>GSM8K / MMLU / ARC-C (in-loop, limited)"]
    E --> F["pick_best = argmax aggregate<br/>(ties → prefer a merge = forgetting guard)"]
    F --> G["live = winner · track global best · prune to ≤5 checkpoints"]
    G -->|"k < E"| B
    G -->|"k = E (done)"| H["FULL benchmarks:<br/>original · final-SFT · best"]
    H --> I["publish best → Hugging Face (model card + report)"]
```

The deliverable is the **SFT vs SFT+merge** comparison per epoch, plus a final table against the
original **instruct** baseline (`results/benchmarks.md`). Everything is tracked in **MLflow**; the
chosen checkpoint is published to the **Hugging Face Hub** with a model card.

## Results

Full write-up with error bars, significance, and threats-to-validity in [`results/REPORT.md`](results/REPORT.md).

**Final benchmarks (full eval, ±1 SE):** the merge checkpoint improves GSM8K (+3.7 pts, ≈2σ) while
retaining MMLU; aggregate gains are within noise on a single run.

![Final benchmarks](results/figures/fig_final_benchmarks.png)

**Per-epoch SFT vs 7 merge methods:** merges win at epoch 1 (TIES best) right after the instruct→SFT
shift; once stabilized, plain SFT overtakes - so *when* you merge matters more than *which* method.

![Per-epoch method comparison](results/figures/fig_method_comparison.png)

**Training dynamics:** SFT loss collapses to ~0.02 (≈99% token accuracy) - heavy memorization of the
templated data, which motivates the merge correction.

![SFT loss collapse](results/figures/fig_loss_collapse.png)

Regenerate with `python scripts/make_figures.py`.

## Setup

```bash
conda activate dist_train          # python 3.12 + uv
uv pip install -r requirements.txt # torch, transformers, trl, lm-eval, mlflow, ...
uv pip install --group dev         # pytest
python -m pytest                   # offline unit tests
```

## Run

```bash
# smoke (cheap end-to-end): 1 epoch, ~200 examples, in-loop bench limit 20
MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml \
  --set num_epochs=1 max_samples=200 eval_limit=20

# full recipe (E=3, 25k, in-loop limit 200; SFT on both GPUs, bench on cuda:1)
MLFLOW_TRACKING_URI=http://localhost:5000 python train_distill.py --config configs/sft.yaml

# full benchmark eval of any checkpoint(s), e.g. the original instruct vs a merged checkpoint
CUDA_VISIBLE_DEVICES=1 python -m distill.eval_bench --models Qwen/Qwen3.5-0.8B outputs/distill/epoch3/merge

# publish the chosen checkpoint with model card + results
python publish_hub.py --model_dir outputs/distill/epoch3/merge --repo Amine-CV/Qwen3.5-0.8B-Mythos-Distill
```

GPUs: SFT runs on both (`CUDA_VISIBLE_DEVICES=0,1` + `configs/accelerate_multi.yaml`); merge/eval pin
to `cuda:1` (cuda:0 is shared). MLflow server is expected at `http://localhost:5000`.

## Layout

```
mythos-distill/
├── train_distill.py       # orchestrator: per-epoch SFT → merge → bench → decide → MLflow
├── sft_worker.py          # one SFT epoch (TRL SFTTrainer; launched via accelerate, multi-GPU)
├── publish_hub.py         # push chosen checkpoint + model card + results to the Hub
├── pyproject.toml         # uv project (deps) + pytest config
├── distill/
│   ├── config.py          # SFTMergeConfig + YAML/CLI override resolution (type-coerced)
│   ├── data.py            # Mythos dataset loading + chat rendering for SFT
│   ├── merge.py           # task-arithmetic / chat-vector merge (pure core + I/O wrapper)
│   ├── recipe.py          # the merge-vs-continue decision policy
│   ├── eval_bench.py      # lm-evaluation-harness wrapper (GSM8K / MMLU / ARC-Challenge)
│   ├── tracking.py        # best-effort MLflow logging
│   └── utils.py           # seed, param counts, throughput
├── configs/
│   ├── sft.yaml           # recipe hyperparameters
│   ├── accelerate.yaml    # single-GPU
│   └── accelerate_multi.yaml  # two-GPU DDP (cuda:0 + cuda:1)
└── tests/                 # offline pure-unit tests (pytest)
```

## License

Pipeline code: MIT. Training data `WithinUsAI/claude_mythos_distilled_25k` is Apache-2.0; the Qwen3.5
weights follow their own license. Built with Hugging Face Transformers, TRL, Accelerate, Datasets,
PEFT, lm-evaluation-harness, and MLflow.
