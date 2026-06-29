#!/usr/bin/env bash
# End-to-end: prepare data -> CPT -> merge -> evaluate.
# Env overrides:  BASE, INSTRUCT, MINUTES, ALPHA, ACCEL_CONFIG
#   BASE=meta-llama/Llama-3.2-1B INSTRUCT=meta-llama/Llama-3.2-1B-Instruct bash scripts/run_all.sh
#   # both GPUs (cuda:0 is shared on this box):
#   CUDA_VISIBLE_DEVICES=0,1 ACCEL_CONFIG=configs/accelerate_multi.yaml bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-Qwen/Qwen2.5-0.5B}"
INSTRUCT="${INSTRUCT:-Qwen/Qwen2.5-0.5B-Instruct}"
MINUTES="${MINUTES:-90}"
ALPHA="${ALPHA:-1.0}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate.yaml}"   # configs/accelerate_multi.yaml for 2-GPU DDP

echo "== [0/3] prepare data =="
python prepare_data.py

echo "== [1/3] continual pretraining (base=$BASE, cap=${MINUTES}m, accel=$ACCEL_CONFIG) =="
accelerate launch --config_file "$ACCEL_CONFIG" train_cpt.py \
  --config configs/cpt.yaml \
  --set base_model="$BASE" max_train_minutes="$MINUTES"

echo "== [2/3] merge CPT delta onto instruct =="
python merge.py \
  --base "$BASE" --instruct "$INSTRUCT" \
  --cpt outputs/cpt/adapter --alpha "$ALPHA" --out outputs/merged

echo "== [3/3] evaluate (domain QA + general retention) =="
python eval_qa.py  --models "$INSTRUCT" outputs/merged --max_samples 500
python eval_ppl.py --models "$INSTRUCT" outputs/merged

echo "== done. merged model -> outputs/merged =="
