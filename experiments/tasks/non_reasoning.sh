#!/usr/bin/env bash
model=$1
extra_args="${@:2}"
tasks=arc_challenge,arc_easy,boolq,hellaswag

./experiments/lm_eval.sh "$model,enable_thinking=False" $tasks \
    --output_path ./lm_eval_output \
    --log_samples \
    --batch_size 1 \
    $extra_args
