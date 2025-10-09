set -e

export PYTHONPATH=$(pwd)

model_path=$1
shards=$2

if [ -z $shards ]; then
    shards=16
fi

python3 optimize.py \
    --model $model_path \
    --params "channel_scales:0.025,angles:0.025" "weight:5e-6,quantizer:5e-7" \
    --epochs 10 10 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --datasets wikitext2 c4 redpajama \
    --val-dataset pileval \
    --train-size 1024 \
    --validation-size 64 \
    --batch-size 8 \
    --seqlen 2048 \
    --cache-shards $shards \
    --output-dir ./output \
    --resume \
    --seed 0
