#!/usr/bin/env bash

set -e

HF_MODEL=$1

# Check required transformers version
REQUIRED_VERSION="4.45.2"
CURRENT_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
if [ "$CURRENT_VERSION" != "$REQUIRED_VERSION" ]; then
  echo "   Detected transformers version: ${CURRENT_VERSION}"
  echo "   Required version: ${REQUIRED_VERSION}"
  echo "   Please downgrade using:"
  echo "   pip install transformers==${REQUIRED_VERSION}"
  exit 1
fi

# Benchmark batch size = 1 decoding throughput
python scripts/interactive_gen.py \
  --hf-path "$HF_MODEL" \
  --bench-model
