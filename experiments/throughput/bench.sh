#!/usr/bin/env bash

set -e

HF_MODEL=$1

# Benchmark batch size = 1 decoding throughput
python scripts/interactive_gen.py \
  --model "$HF_MODEL" \
  --bench-model
