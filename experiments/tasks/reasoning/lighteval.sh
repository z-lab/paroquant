#!/usr/bin/env bash

set -e

model=$1

pip freeze | grep lighteval==0.11.0 > /dev/null || pip install lighteval==0.11.0
pip freeze | grep emoji > /dev/null || pip install emoji
pip freeze | grep autoawq > /dev/null || pip install autoawq[kernels]
pip freeze | grep transformers==4.55.2 > /dev/null || pip install transformers==4.55.2

tasks=$2
extra_args="${@:3}"

echo "Task: $tasks"
lighteval accelerate \
    "model_name=$model,dtype=float16,batch_size=1,generation_parameters={temperature:0.6}" \
    "$tasks" \
    --custom-tasks experiments/tasks/reasoning/reasoning.py \
    --num-fewshot-seeds 0 \
    --output-dir lighteval_results \
    $extra_args