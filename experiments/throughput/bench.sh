#!/usr/bin/env bash

set -e

HF_MODEL=$1

# Benchmark batch size = 1 decoding throughput
python scripts/bench_model.py \
  --model "$HF_MODEL" \
  --prefill-len 256 \
  --decode-len 512
