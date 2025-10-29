#!/bin/bash
set -e

model="meta-llama/Meta-Llama-3-8B"
include="full_rotation,scaling,paroquant_no_scaling,paroquant,hadamard"

for layer in 0 15 31; do
    for name in "self_attn.q_proj" "self_attn.k_proj" "self_attn.v_proj" "self_attn.o_proj" "mlp.up_proj" "mlp.gate_proj"; do
        echo "Plotting optimization for layer $layer, linear $name"
        python experiments/plots/plot_rotation_optimization_convergence.py \
            --model $model \
            --layer $layer \
            --linear-name $name \
            --steps 200 \
            --include $include \
            --no-custom-ticks \
            --no-labels \
            --grid \
            --figsize 1.8,1.2
    done
done

