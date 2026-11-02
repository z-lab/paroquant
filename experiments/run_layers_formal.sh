#!/usr/bin/env bash
set -eo pipefail
card=${ASCEND_CARD:-6}
mkdir -p results/layers_formal
for size in 4 8; do
    for seed in 932 933 934; do
        name="${size}b_seed${seed}"
        source /usr/local/Ascend/cann/set_env.sh
        status=$(npu-smi info)
        printf '%s\n' "$status" > "results/layers_formal/${name}_hardware.txt"
        printf '%s\n' "$status" | rg -q "\\| ${card} +910B2C +\\| OK +\\|"
        printf '%s\n' "$status" | rg -q "No running processes found in NPU ${card} +\\|"
        ASCEND_RT_VISIBLE_DEVICES="$card" OMP_NUM_THREADS=8 \
            .venv/bin/python experiments/benchmark_layers.py --model "models/Qwen3-${size}B-PARO" \
            --rows 1 16 128 --repeats 10 --seed "$seed" \
            --output "results/layers_formal/${name}.json" > "results/layers_formal/${name}.log" 2>&1
    done
done
