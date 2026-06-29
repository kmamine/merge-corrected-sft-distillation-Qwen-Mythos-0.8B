"""Stage 1 — continual pretraining (CPT) of the BASE model on aviation narratives.

LoRA by default (fast, fits any GPU >=16GB); set use_lora=false for full CPT.
A wall-clock cap (max_train_minutes) makes the run land inside your time budget
regardless of hardware. Built on HF Accelerate so it runs single- or multi-GPU
(use the FSDP2 accelerate config for multi-GPU full CPT).

Launch:
    python train_cpt.py --config configs/cpt.yaml
    accelerate launch --config_file configs/accelerate.yaml train_cpt.py --config configs/cpt.yaml
    # override anything:  --set max_train_minutes=60 use_lora=false learning_rate=2e-5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed as accel_set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from aero_cpt.config import load_config, parse_kv
from aero_cpt.data import (PackedDataset, build_replay_blocks, collate_packed,
                      pack_texts)
from aero_cpt.utils import ThroughputMeter, count_parameters, human


def read_corpus(path):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2), fused=use_fused,
    )


def maybe_wrap_lora(model, cfg, accelerator):
    if not cfg.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    if cfg.lora_target_modules.strip() == "all-linear":
        target = "all-linear"
    else:
        target = [m.strip() for m in cfg.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    return model


def save_output(accelerator, model, tokenizer, cfg):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if cfg.use_lora:
        out = os.path.join(cfg.output_dir, "adapter")
        if accelerator.is_main_process:
            os.makedirs(out, exist_ok=True)
            unwrapped.save_pretrained(out)          # PEFT adapter (tiny)
            tokenizer.save_pretrained(out)
        tag = "LoRA adapter"
    else:
        out = os.path.join(cfg.output_dir, "final")
        state = accelerator.get_state_dict(model)    # full gather (works under FSDP too)
        if accelerator.is_main_process:
            os.makedirs(out, exist_ok=True)
            unwrapped.save_pretrained(
                out, state_dict=state, safe_serialization=True,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(out)
        tag = "full CPT model"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[save] {tag} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cpt.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_kv(args.overrides))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.report_to, project_dir=cfg.output_dir,
    )
    accel_set_seed(cfg.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data: pack domain corpus (+ optional general replay), done once on main first
    with accelerator.main_process_first():
        texts = read_corpus(cfg.corpus_path)
        domain_blocks = pack_texts(texts, tokenizer, cfg.block_size)
        n_replay = int(len(domain_blocks) * cfg.replay_ratio)
        replay_blocks = build_replay_blocks(
            tokenizer, cfg.block_size, n_replay,
            cfg.replay_dataset, cfg.replay_name, cfg.replay_split,
        ) if n_replay > 0 else []
    all_blocks = domain_blocks + replay_blocks
    import random as _random
    _random.Random(cfg.seed).shuffle(all_blocks)
    dataset = PackedDataset(all_blocks)

    loader = DataLoader(
        dataset, batch_size=cfg.per_device_batch_size, shuffle=True,
        collate_fn=collate_packed, drop_last=True,
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = maybe_wrap_lora(model, cfg, accelerator)
    trainable, total = count_parameters(model)

    optimizer = build_optimizer(model, cfg)

    # size the schedule from the (sharded) loader length
    loader = accelerator.prepare(loader)
    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    max_steps = steps_per_epoch * cfg.num_epochs
    if cfg.max_steps and cfg.max_steps > 0:
        max_steps = min(max_steps, cfg.max_steps)
    warmup_steps = int(max_steps * cfg.warmup_ratio)
    scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    if cfg.report_to:
        accelerator.init_trackers(cfg.run_name, config=vars(cfg))

    if accelerator.is_main_process:
        print("=" * 64)
        print(f"  base model     : {cfg.base_model}")
        print(f"  mode           : {'LoRA' if cfg.use_lora else 'full'} CPT")
        print(f"  world size     : {accelerator.num_processes}")
        print(f"  domain blocks  : {len(domain_blocks):,}  (+{len(replay_blocks):,} replay)")
        print(f"  block size     : {cfg.block_size}")
        print(f"  global batch   : {cfg.per_device_batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"  trainable/total: {human(trainable)} / {human(total)}")
        print(f"  max steps      : {max_steps}  (cap: {cfg.max_train_minutes} min)")
        print("=" * 64)

    # ---- training loop with wall-clock cap ----
    meter = ThroughputMeter()
    running = torch.zeros((), device=accelerator.device)
    global_step = 0
    t0 = time.time()
    stop = False

    for epoch in range(cfg.num_epochs):
        if stop:
            break
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.detach()
            meter.update(batch["attention_mask"].sum().item())

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.logging_steps == 0:
                    avg = accelerator.gather(running).mean().item() / (
                        cfg.logging_steps * cfg.gradient_accumulation_steps)
                    tok_s = meter.rate * accelerator.num_processes
                    lr = scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        mins = (time.time() - t0) / 60
                        print(f"step {global_step:>5} | loss {avg:6.4f} | "
                              f"ppl {math.exp(min(avg, 20)):8.2f} | {tok_s:8.0f} tok/s | "
                              f"lr {lr:.2e} | {mins:5.1f}m")
                    if cfg.report_to:
                        accelerator.log({"train/loss": avg, "train/tok_per_s": tok_s,
                                         "train/lr": lr}, step=global_step)
                    running.zero_()

                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    accelerator.save_state(os.path.join(cfg.output_dir, f"step_{global_step}"))

                # wall-clock + hard-step caps
                elapsed_min = (time.time() - t0) / 60
                if cfg.max_train_minutes > 0 and elapsed_min >= cfg.max_train_minutes:
                    if accelerator.is_main_process:
                        print(f"[stop] hit time cap ({cfg.max_train_minutes} min) at step {global_step}")
                    stop = True
                    break
                if global_step >= max_steps:
                    stop = True
                    break

    save_output(accelerator, model, tokenizer, cfg)
    if cfg.report_to:
        accelerator.end_training()


if __name__ == "__main__":
    main()
