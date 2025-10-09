#!/usr/bin/env bash
# https://github.com/OpenGVLab/OmniQuant
# Commit: 95fae1bd00eaa60d03a20eb39c76c4cb173f51f7

set -e

# Model path is one of models from
# https://huggingface.co/collections/relaxml/qtip-quantized-models-66fa253ad3186746f4b62803
model_path="$1"
seqlen="$2"
project_dir=baselines/qtip

if [[ ! -d ./baselines ]]; then
    mkdir -p ./baselines
fi

if [[ ! -d $project_dir ]]; then
    git clone https://github.com/Cornell-RelaxML/qtip $project_dir
fi

PYTHONPATH=$project_dir python $project_dir/eval/eval_ppl.py \
    --hf_path $model_path  \
    --seed 0 \
    --seqlen $seqlen