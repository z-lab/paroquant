#!/usr/bin/env bash
set -eo pipefail
card=${ASCEND_CARD:-6}
mkdir -p results/models_formal
for size in 4 8; do
    for seed in 917 918 919; do
        for mode in optimized naive dense_fused bf16; do
            model="models/Qwen3-${size}B-PARO"
            if [[ "$mode" == bf16 ]]; then
                model="models/Qwen3-8B"
                if [[ "$size" == 4 ]]; then model="models/Qwen3-4B"; fi
            fi
            name="${size}b_${mode}_seed${seed}"
            source /usr/local/Ascend/cann/set_env.sh
            status=$(npu-smi info)
            printf '%s\n' "$status" > "results/models_formal/${name}_hardware.txt"
            printf '%s\n' "$status" | rg -q "\\| ${card} +910B2C +\\| OK +\\|"
            printf '%s\n' "$status" | rg -q "No running processes found in NPU ${card} +\\|"
            ASCEND_RT_VISIBLE_DEVICES="$card" OMP_NUM_THREADS=8 \
                .venv/bin/python experiments/evaluate_ascend.py --model "$model" --mode "$mode" \
                --new-tokens 8 --seed "$seed" --data data/wikitext2_test.jsonl --prompt-lengths 32 512 2048 \
                --output "results/models_formal/${name}.json" > "results/models_formal/${name}.log" 2>&1
        done
    done
done
