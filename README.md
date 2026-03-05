# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

[Paper](https://arxiv.org/abs/2511.10645) |
[Blog](https://paroquant.z-lab.ai) |
[Models](https://huggingface.co/collections/z-lab/paroquant)

ParoQuant is an efficient 4-bit weight-only quantization method that achieves state-of-the-art quantization accuracy while incurring minimal overhead during inference. It currently supports LLaMA and Qwen3 model family.

<img style="width:100%" src="assets/method.svg" alt="ParoQuant Method Diagram">

## Quick Start

Try out ParoQuant models with a single command:

```bash
docker run --pull=always --rm -it --gpus all --ipc=host \
  ghcr.io/z-lab/paroquant:chat \
  --model z-lab/Qwen3-8B-PARO
```

For ARM64 platforms (e.g. NVIDIA DGX Spark), please use `ghcr.io/z-lab/paroquant:chat-cu130` instead.

## Models

We provide pre-quantized 4-bit ParoQuant models listed below:

| Model                        | Hugging Face Path                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Meta-Llama-3-8B              | [`z-lab/Meta-Llama-3-8B-PARO`](https://huggingface.co/z-lab/Meta-Llama-3-8B-PARO)                           |
| Meta-Llama-3-70B             | [`z-lab/Meta-Llama-3-70B-PARO`](https://huggingface.co/z-lab/Meta-Llama-3-70B-PARO)                         |
| Llama-3.1-8B-Instruct        | [`z-lab/Llama-3.1-8B-Instruct-PARO`](https://huggingface.co/z-lab/Llama-3.1-8B-Instruct-PARO)               |
| Llama-2-7b-hf                | [`z-lab/Llama-2-7b-hf-PARO`](https://huggingface.co/z-lab/Llama-2-7b-hf-PARO)                               |
| Qwen3-0.6B                   | [`z-lab/Qwen3-0.6B-PARO`](https://huggingface.co/z-lab/Qwen3-0.6B-PARO)                                     |
| Qwen3-1.7B                   | [`z-lab/Qwen3-1.7B-PARO`](https://huggingface.co/z-lab/Qwen3-1.7B-PARO)                                     |
| Qwen3-4B                     | [`z-lab/Qwen3-4B-PARO`](https://huggingface.co/z-lab/Qwen3-4B-PARO)                                         |
| Qwen3-8B                     | [`z-lab/Qwen3-8B-PARO`](https://huggingface.co/z-lab/Qwen3-8B-PARO)                                         |
| Qwen3-14B                    | [`z-lab/Qwen3-14B-PARO`](https://huggingface.co/z-lab/Qwen3-14B-PARO)                                       |
| Qwen3-0.6B-Base              | [`z-lab/Qwen3-0.6B-Base-PARO`](https://huggingface.co/z-lab/Qwen3-0.6B-Base-PARO)                           |
| Qwen3-1.7B-Base              | [`z-lab/Qwen3-1.7B-Base-PARO`](https://huggingface.co/z-lab/Qwen3-1.7B-Base-PARO)                           |
| Qwen3-4B-Base                | [`z-lab/Qwen3-4B-Base-PARO`](https://huggingface.co/z-lab/Qwen3-4B-Base-PARO)                               |
| Qwen3-8B-Base                | [`z-lab/Qwen3-8B-Base-PARO`](https://huggingface.co/z-lab/Qwen3-8B-Base-PARO)                               |
| Qwen3-14B-Base               | [`z-lab/Qwen3-14B-Base-PARO`](https://huggingface.co/z-lab/Qwen3-14B-Base-PARO)                             |
| Qwen3-4B-Thinking-2507       | [`z-lab/Qwen3-4B-Thinking-2507-PARO`](https://huggingface.co/z-lab/Qwen3-4B-Thinking-2507-PARO)             |
| DeepSeek-R1-Distill-Llama-8B | [`z-lab/DeepSeek-R1-Distill-Llama-8B-PARO`](https://huggingface.co/z-lab/DeepSeek-R1-Distill-Llama-8B-PARO) |

We also provide the original optimization checkpoints and pseudo-quantized models in [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints).

## Setup

We recommend using the docker image `ghcr.io/z-lab/paroquant:latest` without manually setting up environment:

```
docker run -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:latest
```

Please follow the setup instructions below if you'd prefer running on the host.

```bash
git clone https://github.com/z-lab/paroquant
cd paroquant

# GPU — Transformers inference
pip install -e ".[transformers]" --no-build-isolation

# GPU — vLLM inference
pip install -e ".[vllm]" --no-build-isolation

# Apple Silicon — MLX inference (no CUDA build needed)
pip install -e ".[mlx]"

# GPU — optimization & evaluation
pip install -e ".[optim,eval]" --no-build-isolation
```

You may need to adjust the PyTorch version in [`pyproject.toml`](pyproject.toml) to match your CUDA version.

## Inference

Interactive chat with any backend:

```bash
# with docker
docker run --rm -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:chat --model z-lab/Qwen3-8B-PARO

# without docker (defaults to transformers backend)
python3 -m paroquant.cli.chat --model z-lab/Qwen3-8B-PARO

# with vLLM backend
python3 -m paroquant.cli.chat --model z-lab/Qwen3-8B-PARO --backend vllm

# on Apple Silicon
python3 -m paroquant.cli.chat --model z-lab/Qwen3-8B-PARO --backend mlx
```

## Optimization

Run the optimization script to obtain per-layer checkpoints:

```bash
experiments/optimize/4bit.sh Qwen/Qwen3-8B
```

Then convert to a Hugging Face model. Use `--mode real` (default) for INT4 weights or `--mode pseudo` for FP16 weights simulating quantization:

```bash
python3 -m paroquant.cli.convert \
    --model Qwen/Qwen3-8B \
    --result-dir output/Qwen3-8B \
    --output-path models/Qwen3-8B-PARO
```

## Reproduction

In the [`experiments`](./experiments/) directory, we provide the original scripts that produce the models, experiment results, and figures in the paper. Please refer to the [README](./experiments/README.md) for more details.

## Docker

We provide four docker images:

- `ghcr.io/z-lab/paroquant:latest` for optimization and non-reasoning task evaluation
- `ghcr.io/z-lab/paroquant:chat` for running the chat app
- `ghcr.io/z-lab/paroquant:chat-cu130` for running the chat app with CUDA 13.0
- `ghcr.io/z-lab/paroquant:eval-reasoning` for reasoning task evaluation

```
docker run -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:<tag>
```

## Contribution

Contributions are welcome! Please install `pre-commit` to ensure consistent code styles:

```bash
pip install pre-commit
pre-commit install
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
