#!/usr/bin/env bash
model=$1
extra_args="${@:2}"
tasks=arc_challenge,arc_easy,boolq,hellaswag

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$model,enable_thinking=False,dtype=float16 \
    --tasks $tasks \
    --batch_size 32 \
    $extra_args
