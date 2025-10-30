#!/usr/bin/env bash
# https://github.com/facebookresearch/SpinQuant
# Commit: 8f47aa3f00e8662caf1a484153920a07e5281c3a

set -e

model_path="$1"
model_name=$(echo $model_path | awk -F'/' '{print $2}')
model_size=$(echo $model_name | awk -F'-' '{print $3}')
model_size=${model_size^^}

bits="$2"
seqlen="$3"

project_dir=baselines/SpinQuant

if [[ ! -d ./baselines ]]; then
    mkdir -p ./baselines
fi

if [[ ! -d $project_dir ]]; then
    git clone https://github.com/facebookresearch/SpinQuant $project_dir
fi

cache_dir=$project_dir/cache
rot_dir=$cache_dir/${model_size}_W${bits}_trained
output_dir=$cache_dir/${model_size}_W${bits}_output

PYTHONPATH=$project_dir torchrun --nnodes=1 --nproc_per_node=3 $project_dir/optimize_rotation.py \
    --input_model $model_path  \
    --output_rotation_path $rot_dir \
    --output_dir $output_dir \
    --logging_dir $output_dir \
    --model_max_length 2048 \
    --fp16 True \
    --bf16 False \
    --log_on_each_node False \
    --per_device_train_batch_size 8 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --save_safetensors False \
    --max_steps 100 \
    --w_bits $bits \
    --a_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --w_groupsize 128 \
    --k_groupsize 128 \
    --v_groupsize 128 \

PYTHONPATH=$project_dir torchrun --nnodes=1 --nproc_per_node=1 $project_dir/ptq.py \
    --input_model $model_path \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length $seqlen \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --w_bits ${bits} \
    --a_bits 16 \
    --k_bits 16 \
    --v_bits 16 \
    --w_clip \
    --a_asym \
    --k_asym \
    --v_asym \
    --w_groupsize 128 \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --rotate \
    --optimized_rotation_path $cache_dir/13b/R.bin \

