#!/usr/bin/env bash
ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

set -e

model_path=$1
seed=$2
extra_args="${@:3}"

function install_if_missing() {
    package=$1
    pip freeze | grep $package > /dev/null || pip install $package
}

install_if_missing lighteval==0.8.1
install_if_missing datasets==2.18.0
install_if_missing transformers==4.55.2
install_if_missing vllm==0.10.1
pip freeze | grep autoawq > /dev/null || pip install autoawq[kernels]

datasets=("GPQA-Diamond" "AIME-90" "GSM8K" "MMLU-PRO")

for dataset in "${datasets[@]}"; do 
    echo "Task: $dataset"
    python experiments/tasks/reasoning/lighteval_custom/inference.py \
        --model $model_path \
        --dataset $dataset \
        --seed $seed
done
