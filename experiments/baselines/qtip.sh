#!/usr/bin/env bash
set -e

model_path="$1"
seqlen="$2"
baseline_dir=paroquant-baselines
project_dir=$baseline_dir/qtip

if [[ ! -d $baseline_dir ]]; then
    git clone https://github.com/liang2kl/paroquant-baselines $baseline_dir
fi

export PYTHONPATH=$project_dir

(cd $project_dir && ./train.sh $model_path)
(cd $project_dir && ./train_e2e.sh $model_path)

# Convert to pseudo quant
# model_name=$(echo $model_path | awk -F'/' '{print $2}')
# (
#     cd $project_dir &&
#     python3 scripts/convert_to_pseudo_fp16.py \
#         --hf_path $project_dir/output/e2e_models/${model_name}_QTIP \
#         --output_path $project_dir/output/pseudo_models/$model_name
# )
