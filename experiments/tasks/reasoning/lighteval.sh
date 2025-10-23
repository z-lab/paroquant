#!/usr/bin/env bash
ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

set -e

model_path=$1
seed=$2
extra_args="${@:3}"

pip freeze | grep lighteval==0.8.1 > /dev/null || pip install lighteval==0.8.1
pip freeze | grep datasets==2.18.0 > /dev/null || pip install datasets==2.18.0
pip freeze | grep emoji > /dev/null || pip install emoji
pip freeze | grep autoawq > /dev/null || pip install autoawq[kernels]
pip freeze | grep transformers==4.55.2 > /dev/null || pip install transformers==4.55.2



datasets=( "GPQA-Diamond" "AIME-90" "GSM8K" "MMLU-PRO")


for dataset in "${datasets[@]}"; do 
    echo "Task: $dataset"
    python experiments/tasks/reasoning/lighteval_custom/inference.py \
        --model $model_path \
        --dataset $dataset \
        --seed $seed
done
