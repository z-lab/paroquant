#!/usr/bin/env bash
set -e

model_path="$1"
model_name=$(echo $model_path | awk -F'/' '{print $2}')
model_name_wo_hf="${model_name%-hf}"

bits="$2"
seqlen="$3"
baseline_dir=paroquant-baselines
project_dir=$baseline_dir/EfficientQAT

if [[ ! -d $baseline_dir ]]; then
    git clone https://github.com/liang2kl/paroquant-baselines $baseline_dir
fi

conda env list | grep efficientqat > /dev/null || conda create -n efficientqat python==3.11 -y

function run_conda_cmd() {
    conda run -n efficientqat --live-stream "$@"
}

export PYTHONPATH=$project_dir

run_conda_cmd pip install -r $project_dir/requirements.txt

run_conda_cmd python $project_dir/main_block_ap.py \
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

# Convert to pseudo quant
# run_conda_cmd python $project_dir/model_transfer/real_to_fake.py \
#     --model $project_dir/output/block_ap_models/$model_name \
#     --save_dir $project_dir/output/pseudo_models/$model_name \
#     --wbits $bits \
#     --group_size 128
