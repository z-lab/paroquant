set -e

export PYTHONPATH=$(pwd)

model_path=meta-llama/Meta-Llama-3-8B

python3 optimize.py \
    --model $model_path \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --datasets redpajama \
    --val-dataset pileval \
    --train-size 2048 \
    --validation-size 64 \
    --batch-size 16 \
    --seqlen 2048 \
    --cache-shards $shards \
    --output-dir ./output/ablations/redpajama \
    --resume \
    --seed 0
