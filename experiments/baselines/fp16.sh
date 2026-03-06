#!/usr/bin/env bash

set -e

model_path="$1"
seqlen="$2"
project_dir=baselines/fp16

python -m paroquant.cli.evaluate \
    --model $model_path \
    --seed 0 \
    --seqlen $seqlen \
