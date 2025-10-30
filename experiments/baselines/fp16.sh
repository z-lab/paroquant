#!/usr/bin/env bash

set -e

model_path="$1"
seqlen="$2"
project_dir=baselines/fp16

python scripts/eval_ppl.py \
    --model $model_path \
    --seed 0 \
    --seqlen $seqlen \
