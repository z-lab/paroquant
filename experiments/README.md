# Reproduction Scripts

These are the scripts to reproduce most results in the paper. The environment for all the experiments (except for baselines and reasoning tasks) is provided in [`environment.yml`](../environment.yml).

> The docker image for this environment is `ghcr.io/z-lab/paroquant:default`.

We use pseudo-quantized models for all experiments, except for experiments on AWQ where we use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) to run real-quantized models. ParoQuant's pseudo-quantized models used in the experiments can be downloaded from the `pseudo` directory at [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints).

## Optimization

To quantize and optimize a model with ParoQuant:

```
./experiments/optimize/4bit.sh <model> [<num_shards>]
```

`num_shards` is 1 by default and may need to be adjusted for large models to accommodate memory constraints. Run `python3 optimize.py --help` for more details.

We adjust the batch size, learning rate, and number of training samples for LLaMA-3-70B. Please use [`experiments/optimize/4bit_70b.sh`](./optimize/4bit_70b.sh) instead for LLaMA-3-70B.

The optimized checkpoints will be saved to `./output/<model_name>`. To create a Hugging Face model with the checkpoints, use [`scripts/pseudo_quant.py`](../scripts/pseudo_quant.py) for pseudo quantization and [`scripts/real_quant.py`](../scripts/real_quant.py) for real quantization.

## Baselines

Scripts to obtain models quantized by baseline methods presented in the paper are in the [`baselines`](./baselines) directory. These models are used for perplexity and downstream task evaluation. Please refer to each script for its usage.

## Downstream Tasks

To evaluate downstream tasks, use [`tasks/reasoning.sh`](./tasks/reasoning.sh) for reasoning tasks and [`tasks/non_reasoning.sh`](./tasks/non_reasoning.sh) for non-reasoning tasks. Please note that they only support pseudo-quantized and AWQ-quantized models.

To run non-reasoning tasks:

```
./experiments/tasks/non_reasoning.sh <model>
```

We use a separate environment for reasoning tasks. To run reasoning tasks:

```
conda env create -f ./experiments/tasks/reasoning/environment.yml
conda activate paroquant-eval

./experiments/tasks/reasoning.sh <model> <seed> [<task0>, <task1>, ...]
```

> The docker image for this environment is `ghcr.io/z-lab/paroquant:eval-reasoning`.

The seeds we use in our paper are 42 for MMLU-Pro and 42, 0, 1 for other tasks.

## Throughput

Use [`throughput/bench.sh`](./throughput/bench.sh) to benchmark the decoding throughput of a real-quantized ParoQuant model:

```
./experiments/throughput/bench.sh <model>
```

## Ablation Studies

Scripts for the ablation studies are in the [`ablations`](./ablations/) directory. Their usage is similar to the ParoQuant optimization script.

## Plots

We provide the scripts to create the plots in the paper in the [`plots`](./plots/) directory. Some scripts require optimized checkpoints of linear layers, which can be downloaded from [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints).
