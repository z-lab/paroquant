# ParoQuant

**Pairwise Rotation Quantization for Efficient Reasoning LLM Inference**

<p align="center">
  <a href="https://arxiv.org/abs/2511.10645"><img src="https://img.shields.io/badge/arXiv-2511.10645-b31b1b.svg" alt="Paper"></a>
  <a href="https://paroquant.z-lab.ai"><img src="https://img.shields.io/badge/Blog-ParoQuant-blue" alt="Blog"></a>
  <a href="https://huggingface.co/collections/z-lab/paroquant"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" alt="Models"></a>
  <a href="https://pypi.org/project/paroquant/"><img src="https://img.shields.io/pypi/v/paroquant" alt="PyPI"></a>
</p>

State-of-the-art INT4 quantization for LLMs. ParoQuant uses learned pairwise rotations to suppress weight outliers, closing the accuracy gap with FP16 while running at near-AWQ speed. Supports NVIDIA GPUs (vLLM, Transformers) and Apple Silicon (MLX).

<p align="center">
  <a href="https://youtu.be/fISG4CkizLM">
    <img src="https://img.youtube.com/vi/fISG4CkizLM/maxresdefault.jpg" width="80%">
  </a>
</p>

## Quick Start

**NVIDIA GPU:**

```bash
pip install "paroquant[vllm]"
python -m paroquant.cli.chat --model z-lab/Qwen3-8B-PARO

# or with Docker
docker run --pull=always --rm -it --gpus all --ipc=host \
  ghcr.io/z-lab/paroquant:chat --model z-lab/Qwen3-8B-PARO
```

**Apple Silicon:**

```bash
pip install "paroquant[mlx]"
python -m paroquant.cli.chat --model z-lab/Qwen3-8B-PARO
```

## Models

All models are available on [Hugging Face](https://huggingface.co/collections/z-lab/paroquant). Swap the model name in the commands above to try any of them.

**Qwen3**

| Model | Checkpoint |
|---|---|
| Qwen3-0.6B | [`z-lab/Qwen3-0.6B-PARO`](https://huggingface.co/z-lab/Qwen3-0.6B-PARO) |
| Qwen3-1.7B | [`z-lab/Qwen3-1.7B-PARO`](https://huggingface.co/z-lab/Qwen3-1.7B-PARO) |
| Qwen3-4B | [`z-lab/Qwen3-4B-PARO`](https://huggingface.co/z-lab/Qwen3-4B-PARO) |
| Qwen3-8B | [`z-lab/Qwen3-8B-PARO`](https://huggingface.co/z-lab/Qwen3-8B-PARO) |
| Qwen3-14B | [`z-lab/Qwen3-14B-PARO`](https://huggingface.co/z-lab/Qwen3-14B-PARO) |

**Qwen3.5**

| Model | Checkpoint |
|---|---|
| Qwen3.5-0.8B | [`z-lab/Qwen3.5-0.8B-PARO`](https://huggingface.co/z-lab/Qwen3.5-0.8B-PARO) |
| Qwen3.5-2B | [`z-lab/Qwen3.5-2B-PARO`](https://huggingface.co/z-lab/Qwen3.5-2B-PARO) |
| Qwen3.5-4B | [`z-lab/Qwen3.5-4B-PARO`](https://huggingface.co/z-lab/Qwen3.5-4B-PARO) |
| Qwen3.5-9B | [`z-lab/Qwen3.5-9B-PARO`](https://huggingface.co/z-lab/Qwen3.5-9B-PARO) |

**Llama**

| Model | Checkpoint |
|---|---|
| Llama-2-7B | [`z-lab/Llama-2-7b-hf-PARO`](https://huggingface.co/z-lab/Llama-2-7b-hf-PARO) |
| Llama-3-8B | [`z-lab/Meta-Llama-3-8B-PARO`](https://huggingface.co/z-lab/Meta-Llama-3-8B-PARO) |
| Llama-3.1-8B-Instruct | [`z-lab/Llama-3.1-8B-Instruct-PARO`](https://huggingface.co/z-lab/Llama-3.1-8B-Instruct-PARO) |

Want a model that's not listed? [Open an issue](https://github.com/z-lab/paroquant/issues/new) and let us know.

## Reproduction

> [!NOTE]
> The main branch of this repository is under active development, and reproducibility is not guaranteed.
> Please use the [`legacy`](https://github.com/z-lab/paroquant/tree/legacy) branch to reproduce results from the paper.

## Installation

```bash
git clone https://github.com/z-lab/paroquant && cd paroquant

pip install -e ".[vllm]"            # vLLM backend (GPU, recommended)
pip install -e ".[transformers]"    # Transformers backend (GPU)
pip install -e ".[mlx]"             # MLX backend (Apple Silicon)
pip install -e ".[optim,eval]"      # Optimization & evaluation
```

Or use Docker: `docker run -it --gpus all --ipc=host ghcr.io/z-lab/paroquant:latest`

## Quantize Your Own Model

```bash
# 1. Optimize rotation parameters
experiments/optimize/4bit.sh Qwen/Qwen3-8B

# 2. Export to HF checkpoint (--mode real for INT4, --mode pseudo for FP16)
python -m paroquant.cli.convert \
  --model Qwen/Qwen3-8B \
  --result-dir output/Qwen3-8B \
  --output-path models/Qwen3-8B-PARO
```

## Docker Images

| Image | Purpose |
|---|---|
| `ghcr.io/z-lab/paroquant:latest` | Optimization & evaluation |
| `ghcr.io/z-lab/paroquant:chat` | Interactive chat |
| `ghcr.io/z-lab/paroquant:chat-cu130` | Interactive chat (CUDA 13.0 / ARM64) |
| `ghcr.io/z-lab/paroquant:eval-reasoning` | Reasoning task evaluation |

## Citation

```bibtex
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
