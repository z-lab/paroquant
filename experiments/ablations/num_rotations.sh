set -e

export PYTHONPATH=$(pwd)

model_path=meta-llama/Meta-Llama-3-8B
shards=$1

if [ -z $shards ]; then
    shards=1
fi

for num in 1 2 4; do
    python3 optimize.py \
        --model $model_path \
        --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
        --epochs 10 10 \
        --group-size 128 \
        --n-bit 4 \
        --num-rotations $num \
        --datasets wikitext2 c4 redpajama \
        --val-dataset pileval \
        --train-size 2048 \
        --validation-size 64 \
        --batch-size 16 \
        --seqlen 2048 \
        --cache-shards $shards \
        --output-dir ./output/ablations/$num-rotations \
        --resume \
        --seed 0
done
