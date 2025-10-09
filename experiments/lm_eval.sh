#!/usr/bin/env bash

set -e

model=$1
task=$2
extra_args="${@:3}"

echo "Task: $task"
accelerate launch -m lm_eval\
    --model hf \
    --model_args pretrained=$model,dtype=float16 \
    --tasks $task \
    $extra_args
