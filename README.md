# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

[Paper]() | [Models](https://huggingface.co/collections/z-lab/paroquant)

ParoQuant is an efficient 4-bit weight-only quantization method that achieves state-of-the-art quantization accuracy while incurring minimal overhead during inference. It currently supports LLaMA and Qwen3 model family.

## Setup

Clone this repository:

```bash
git clone https://github.com/z-lab/paroquant
cd paroquant
```

Install dependencies:

```bash
# use conda (recommended)
conda env create -f environment.yml
conda activate paroquant
pip install ./kernels

# or use pip
pip install -r requirements.txt
pip install ./kernels
```

## Usage

First, run the optimization script to obtain the optimized checkpoints. The checkpoints will be stored in `output/<model_name>`.

```bash
./experiments/optimize/4bit.sh Qwen/Qwen3-8B
```

Then, create a huggingface model with pseudo quantization (*i.e.,* model weights are in FP16 simulating the quantization) or real quantization (*i.e.*, model weights are in INT4):

```bash
# pseudo quantization
python3 scripts/pseudo_quant.py \
    --model Qwen/Qwen3-8B \
    --result-dir output/Qwen3-8B \
    --output-dir models/Qwen3-8B-PARO-pseudo

# real quantization
python3 scripts/real_quant.py \
    --model Qwen/Qwen3-8B \
    --result-dir output/Qwen3-8B \
    --output-dir models/Qwen3-8B-PARO
```

Pseudo-quantized models can be loaded directly with huggingface. To load real-quantized models with huggingface:

```python
# for LLaMA
from inference_engine.model_executor.models.llama import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("/path/to/quantized/model")

# for Qwen3
from inference_engine.model_executor.models.qwen3 import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained("/path/to/quantized/model")
```

You can also interact with the quantized models directly:

```sh
python3 scripts/interactive_gen.py --model /path/to/quantized/morel --streaming
```

## Models

We provide pre-quantized 4-bit ParoQuant models listed below. These are real-quantized models and can be loaded with the method described above.

| Model                        | Hugging Face Path                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
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

In addition, we provide the original checkpoints and pseudo-quantized models in [`z-lab/paroquant-checkpoints`](https://huggingface.co/z-lab/paroquant-checkpoints) to facilitate reproduction and further research.

## Reproduction

In the [`experiments`](./experiments/) directory, we provide the original scripts that produce the models, experiment results, and figures in the paper. Please refer to the [README](./experiments/README.md) for more details.

## Contribution

Contributions are welcome! Please install `pre-commit` to ensure consistent code styles before commiting any changes:

```bash
pip install pre-commit
pre-commit install
```

## Citation
