#!/usr/bin/env bash
# https://github.com/OpenGVLab/OmniQuant
# Commit: 95fae1bd00eaa60d03a20eb39c76c4cb173f51f7

# You need to modify the code to evaluate sequence length != 4096

set -e

model_path="$1"
model_name=$(echo $model_path | awk -F'/' '{print $2}')

bits="$2"

project_dir=baselines/OmniQuant

if [[ ! -d ./baselines ]]; then
    mkdir -p ./baselines
fi

if [[ ! -d $project_dir ]]; then
    git clone https://github.com/OpenGVLab/OmniQuant $project_dir
fi

(
    cd $project_dir && \
    PYTHONPATH=$project_dir python generate_act_scale_shift.py --model $model_path
)

(
    cd $project_dir && \
    PYTHONPATH=$project_dir python main.py \
        --model $model_path  \
        --epochs 20 \
        --output_dir ./log/$model_name-w${bits}a16 \
        --eval_ppl \
        --wbits ${bits} \
        --abits 16 \
        --group_size 128 \
        --lwc
)
