#!/usr/bin/env bash

tasks="custom|gsm8k|0"
model=$1

./experiments/tasks/reasoning/lighteval.sh $model "$tasks"