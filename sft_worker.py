"""One epoch of SFT on the Mythos distillation data (TRL SFTTrainer).

Launched per epoch by `train_distill.py` (via `accelerate launch` for multi-GPU);
starts from `--init_model` (HF id or a local checkpoint), trains exactly one epoch,
and saves the result to `--out` with the instruct tokenizer. Kept as a separate
worker so the multi-GPU SFT step is isolated from the single-process orchestration
(merge + benchmark eval) that drives the recipe.

    accelerate launch --config_file configs/accelerate_multi.yaml sft_worker.py \
        --config configs/sft.yaml --init_model Qwen/Qwen3.5-0.8B-Base --out outputs/distill/epoch1/sft
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from distill.config import load_config, parse_kv
from distill.data import load_distill_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    ap.add_argument("--init_model", required=True, help="HF id or local dir to start SFT from")
    ap.add_argument("--out", required=True, help="where to save the 1-epoch SFT checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_kv(args.overrides))

    # The base model may ship no chat template; use the instruct tokenizer (same family/vocab)
    # to render chat and as the SFT processing class, and save it alongside the checkpoint.
    tok = AutoTokenizer.from_pretrained(cfg.instruct_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_distill_dataset(cfg, tok)

    model = AutoModelForCausalLM.from_pretrained(
        args.init_model, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False

    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig

        target = ("all-linear" if cfg.lora_target_modules.strip() == "all-linear"
                  else [m.strip() for m in cfg.lora_target_modules.split(",") if m.strip()])
        peft_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target, bias="none", task_type="CAUSAL_LM",
        )

    sft_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=1,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_length=cfg.max_seq_len,
        packing=cfg.packing,
        dataset_text_field="text",
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=ds,
        processing_class=tok, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    if trainer.accelerator.is_main_process:
        tok.save_pretrained(args.out)
        print(f"[sft] saved 1-epoch SFT checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
