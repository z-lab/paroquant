# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

[Paper](https://arxiv.org/abs/2511.10645) |
[Blog](https://paroquant.z-lab.ai) |
[Models](https://huggingface.co/collections/z-lab/paroquant)

ParoQuant is an efficient 4-bit weight-only quantization method that achieves state-of-the-art quantization accuracy while incurring minimal overhead during inference. It currently supports LLaMA and Qwen3 model family.

<img style="width:100%" src="assets/method.svg" alt="ParoQuant Method Diagram">

> [!WARNING] Reproduction
> The [`main`](https://github.com/z-lab/paroquant) branch of this repository is under active development and may break reproducibility. This branch `legacy` is specifically maintained for reproducing the results in the paper.

## Setup

We recommend using the docker image `ghcr.io/z-lab/paroquant:legacy` without manually setting up environment:

```
docker run -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:legacy
```

Please follow the setup instructions below if you'd prefer running on the host.

Clone this repository:

```bash
git clone -b legacy https://github.com/z-lab/paroquant
cd paroquant
```

Install dependencies:

```bash
# use conda (recommended)
conda env create -f environment.yml
conda activate paroquant
pip install ./kernels --no-build-isolation

# or use pip
pip install -r requirements.txt
pip install ./kernels --no-build-isolation
```

You may need to modify [`requirements.txt`](requirements.txt) to match your CUDA version.

## Usage

### Optimization

First, run the optimization script to obtain the optimized checkpoints. The checkpoints will be stored in `output/<model_name>`.

```bash
experiments/optimize/4bit.sh Qwen/Qwen3-8B
```

Then, create a huggingface model with pseudo quantization (*i.e.,* model weights are in FP16 simulating the quantization) or real quantization (*i.e.*, model weights are in INT4):

```bash
# pseudo quantization
python3 scripts/pseudo_quant.py \
    --model Qwen/Qwen3-8B \
    --result-dir output/Qwen3-8B \
    --output-path models/Qwen3-8B-PARO-pseudo

# real quantization
python3 scripts/real_quant.py \
    --model Qwen/Qwen3-8B \
    --result-dir output/Qwen3-8B \
    --output-path models/Qwen3-8B-PARO
```

### Inference

The docker image for interactive inference is `ghcr.io/z-lab/paroquant:chat-legacy`. Install vLLM if you are running on the host:

```bash
pip install vllm==0.15.1
```

To run a real-quantized model with vLLM and open an interactive chat:

```bash
# with docker
docker run --rm -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:chat-legacy --model z-lab/Qwen3-8B-PARO

# without docker
python3 scripts/interactive_gen.py --model z-lab/Qwen3-8B-PARO
```

## Models

The models on [Hugging Face](https://huggingface.co/collections/z-lab/paroquant) have been updated for the newer codebase and is **not** usable with this codebase. You can build the real-quantized models with the checkpoints from [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints). This Hugging Face repository also contains pre-built pseudo-quantized models to facilitate reproduction.

## Reproduction

In the [`experiments`](./experiments/) directory, we provide the original scripts that produce the models, experiment results, and figures in the paper. Please refer to the [README](./experiments/README.md) for more details.

## Docker

We provide four docker images for easy environment setup:

- `ghcr.io/z-lab/paroquant:legacy` for optimization and non-reasoning task evaluation
- `ghcr.io/z-lab/paroquant:chat-legacy` for running the chat app
- `ghcr.io/z-lab/paroquant:chat-cu130-legacy` for running the chat app with CUDA 13.0
- `ghcr.io/z-lab/paroquant:eval-reasoning-legacy` for reasoning task evaluation

Use the following command to create a container and activate an interactive shell:

```
docker run -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:<tag>
```

## Reference

If you find ParoQuant useful or relevant to your research, please kindly cite our paper:

```
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
