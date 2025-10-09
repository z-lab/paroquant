#!/usr/bin/env bash

tasks="custom|aime90|0"
model=$1

./experiments/tasks/reasoning/lighteval.sh $model "$tasks"