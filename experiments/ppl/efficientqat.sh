#!/usr/bin/env bash
# https://github.com/OpenGVLab/OmniQuant
# Commit: 95fae1bd00eaa60d03a20eb39c76c4cb173f51f7

set -e

model_path="$1"
model_name=$(echo $model_path | awk -F'/' '{print $2}')
model_name_wo_hf="${model_name%-hf}"

bits="$2"
seqlen="$3"
project_dir=baselines/EfficientQAT

if [[ ! -d ./baselines ]]; then
    mkdir -p ./baselines
fi

if [[ ! -d $project_dir ]]; then
    git clone https://github.com/OpenGVLab/EfficientQAT $project_dir
fi

PYTHONPATH=$project_dir python $project_dir/main_block_ap.py \
    --model $model_path  \
    --output_dir $project_dir/output/block_ap_log/$model_name \
    --wbits $bits \
    --group_size 128 \
    --quant_lr 1e-4 \
    --weight_lr 1e-5 \
    --real_quant \
    --eval_ppl \
    --ppl_seqlen $seqlen \
    --save_quant_dir $project_dir/output/block_ap_models/$model_name \
