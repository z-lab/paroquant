set -e

model_path=meta-llama/Meta-Llama-3-8B
shards=$1

if [ -z $shards ]; then
    shards=1
fi

python3 -m paroquant.cli.optimize \
    --model $model_path \
    --params "angles:0.05" \
    --epochs 10 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --datasets wikitext2 c4 redpajama \
    --val-dataset pileval \
    --train-size 2048 \
    --validation-size 64 \
    --batch-size 16 \
    --seqlen 2048 \
    --cache-shards $shards \
    --output-dir ./output/ablations/rotation-only \
    --resume \
    --seed 0
