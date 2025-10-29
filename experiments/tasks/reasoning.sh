#!/usr/bin/env bash
ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

set -e

model_path=$1
seed=$2
datasets="${@:3}"

if [ -z "$datasets" ]; then
    datasets=("AIME-2024" "AIME-2025" "GPQA-Diamond" "MMLU-PRO")
else
    datasets=($datasets)
fi

function test_if_missing() {
    pip freeze | grep $1 > /dev/null || (echo "missing $1"; exit 1)
}

test_if_missing transformers==4.55.2
test_if_missing vllm==0.10.1
test_if_missing lighteval==0.8.1
test_if_missing datasets==3.6.0

for dataset in "${datasets[@]}"; do 
    echo "Task: $dataset"
    python experiments/tasks/reasoning/lighteval_custom/inference.py \
        --model $model_path \
        --dataset $dataset \
        --seed $seed
done
