#!/usr/bin/env bash
# Launch just the CPT stage. Extra args forward to train_cpt.py.
#   bash scripts/train_cpt.sh --set max_train_minutes=60 use_lora=false learning_rate=2e-5
set -euo pipefail
cd "$(dirname "$0")/.."

ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

accelerate launch --config_file "${ACCEL_CONFIG}" --num_processes "${NUM_GPUS}" \
  train_cpt.py --config configs/cpt.yaml "$@"
