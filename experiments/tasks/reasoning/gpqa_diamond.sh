#!/usr/bin/env bash

tasks="custom|gpqa:diamond|0"
model=$1

./experiments/tasks/reasoning/lighteval.sh $model "$tasks"