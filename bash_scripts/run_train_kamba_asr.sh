#!/usr/bin/env bash
set -euo pipefail

# Set HF_HOME, WANDB_API_KEY, and HF_API_KEY as appropriate for your environment.
python3 scripts/train_model.py \
  --config config_files/kamba_asr_w2v_bert_10h.yaml
