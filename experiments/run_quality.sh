#!/usr/bin/env bash
set -eo pipefail
card=${ASCEND_CARD:-6}
size=${MODEL_SIZE:-4}
mkdir -p results/quality
if (( $# )); then
    modes=("$@")
else
    modes=(optimized dense_fused bf16)
fi
for mode in "${modes[@]}"; do
    model="models/Qwen3-${size}B-PARO"
    if [[ "$mode" == bf16 ]]; then
        model="models/Qwen3-8B"
        if [[ "$size" == 4 ]]; then model="models/Qwen3-4B"; fi
    fi
    for task in perplexity gsm8k; do
        name="${size}b_${mode}_${task}"
        data=data/wikitext2_test.jsonl
        if [[ "$task" == gsm8k ]]; then data=data/gsm8k_test.jsonl; fi
        if [[ "$mode" == optimized && "$task" == perplexity && -f "results/perplexity_optimized_${size}b.json" ]]; then
            cp "results/perplexity_optimized_${size}b.json" "results/quality/${name}.json"
            continue
        fi
        source /usr/local/Ascend/cann/set_env.sh
        status=$(npu-smi info)
        printf '%s\n' "$status" > "results/quality/${name}_hardware.txt"
        printf '%s\n' "$status" | rg -q "\\| ${card} +910B2C +\\| OK +\\|"
        printf '%s\n' "$status" | rg -q "No running processes found in NPU ${card} +\\|"
        ASCEND_RT_VISIBLE_DEVICES="$card" OMP_NUM_THREADS=8 \
            .venv/bin/python experiments/evaluate_ascend.py --model "$model" --mode "$mode" \
            --task "$task" --data "$data" --batch 8 --new-tokens 512 --resume \
            --output "results/quality/${name}.json" > "results/quality/${name}.log" 2>&1
    done
done
