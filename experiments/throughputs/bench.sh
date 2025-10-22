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


# Benchmark single-batch decoding throughput
python scripts/interactive_gen.py \
  --hf_path "$HF_MODEL" \
  --bench_model


# Optional benchmarks

# Example: benchmark a ParoQuant checkpoint with custom input
# python scripts/interactive_gen.py \
#   --hf_path Hschen335/Qwen3-1.7B-ParoQuant-4bit

# Example: benchmark an empty ParoQuant model (supports Qwen3, LLaMA-2, LLaMA-3)
# python scripts/interactive_gen.py \
#   --hf_path Qwen/Qwen3-14B \
#   --empty_model \
#   --bench_model

# Example: benchmark streaming generation (only for ParoQuant checkpoints)
# python scripts/interactive_gen.py \
#   --hf_path Hschen335/Qwen3-1.7B-ParoQuant-4bit \
#   --streaming

