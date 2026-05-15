set -e

model_path=$1
shards=$2

if [ -z $shards ]; then
    shards=1
fi

python3 -m paroquant.cli.optimize \
    --model $model_path \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 5 5 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --skipped-modules \
        "mlp.gate" "mlp.shared_expert_gate" \
        "mlp.shared_expert.up_proj" "mlp.shared_expert.gate_proj" "mlp.shared_expert.down_proj" \
        "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
    --datasets wikitext2 c4 redpajama \
    --val-dataset pileval \
    --train-size 2048 \
    --validation-size 64 \
    --batch-size 16 \
    --gradient-accumulation-steps 1 \
    --seqlen 2048 \
    --cache-shards $shards \
    --output-dir ./output \
    --resume \
    --seed 0
