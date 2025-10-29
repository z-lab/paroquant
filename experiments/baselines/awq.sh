#!/usr/bin/env bash
# https://github.com/casper-hansen/AutoAWQ
# Commit: d6e797a42b9ef7778de8ee2352116e0f48a78d61
set -e

pip freeze | grep autoawq > /dev/null || pip install autoawq[kernels]

model_path="$1"
bits="$2"
seqlen="$3"
model_name=$(echo $model_path | awk -F'/' '{print $2}')

project_dir=baselines/autoawq

python experiments/ppl/autoawq_cli.py \
    --hf_model_path $model_path \
    --quant_name $model_name-w$bits-g128-quant \
    --local_save_path $project_dir/awq_cache/$model_name-w$bits-g128-quant \
    --zero_point \
    --q_group_size 128 \
    --w_bit $bits

python scripts/eval_ppl.py \
    --tokenizer_path $1 \
    --hf_path $project_dir/awq_cache/$model_name-w$bits-g128-quant \
    --seed 0 \
    --seqlen $seqlen \
