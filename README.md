# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

[Paper]() | [Models](https://huggingface.co/collections/z-lab/paroquant)

## Setup

Clone this repository:

```
git clone https://github.com/z-lab/paroquant
cd paroquant
```

Install dependencies:

```
# use conda (recommended)
conda env create -f environment.yml
conda activate paroquant
pip install ./kernels

# or use pip
pip install -r requirements.txt
pip install ./kernels
```

## Usage

ParoQuant currently supports LLaMA-2, LLaMA-3, and Qwen3 model families.

First, run the optimization script to obtain the optimized checkpoints:

```
# <model> is the huggingface path, e.g., Qwen/Qwen3-8B
./experiments/optimize/4bit.sh <model>
```

The checkpoints will be stored in `output/<model_name>` (*e.g.*, `output/Qwen3-8B`). Then, create a huggingface model with pseudo quantization (*i.e.,* model weights are in FP16 simulating the quantization) or real quantization (*i.e.*, model weights are in INT4):

```
# pseudo quantization
# <model> is the huggingface path, e.g., Qwen/Qwen3-8B
# <model_path> is the last component of <model>, e.g., Qwen3-8B
python3 scripts/pseudo_quant.py \
    --model <model>
    --result-dir output/<model_name>
    --output-dir models/<model_name>-Paro

# real quantization
python3 scripts/real_quant.py \
    --model <model>
    --result-dir output/<model_name>
    --output-dir models/<model_name>-Paro
```

Pseudo-quantized models can be loaded directly with huggingface.

To run real-quantized models:

```
python scripts/interactive_gen.py --hf-path <path_to_quantized_model>
```

To load real-quantized models with huggingface:

```
# for LLaMA
from inference_engine.model_executor.models.llama import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("path/to/quantized/model")

# for Qwen3
from inference_engine.model_executor.models.qwen3 import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained("path/to/quantized/model")
```

## Models

We provide pre-quantized 4-bit ParoQuant models listed below. These are real-quantized models and can be loaded with the method described above.

| Model                        | Hugging Face Path                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| Meta-Llama-3-8B              | [z-lab/Meta-Llama-3-8B-PARO](https://huggingface.co/z-lab/Meta-Llama-3-8B-PARO)                           |
| Meta-Llama-3-70B             | [z-lab/Meta-Llama-3-70B-PARO](https://huggingface.co/z-lab/Meta-Llama-3-70B-PARO)                         |
| Llama-3.1-8B-Instruct        | [z-lab/Llama-3.1-8B-Instruct-PARO](https://huggingface.co/z-lab/Llama-3.1-8B-Instruct-PARO)               |
| Llama-2-7b-hf                | [z-lab/Llama-2-7b-hf-PARO](https://huggingface.co/z-lab/Llama-2-7b-hf-PARO)                               |
| Qwen3-0.6B                   | [z-lab/Qwen3-0.6B-PARO](https://huggingface.co/z-lab/Qwen3-0.6B-PARO)                                     |
| Qwen3-0.6B-Base              | [z-lab/Qwen3-0.6B-Base-PARO](https://huggingface.co/z-lab/Qwen3-0.6B-Base-PARO)                           |
| Qwen3-1.7B                   | [z-lab/Qwen3-1.7B-PARO](https://huggingface.co/z-lab/Qwen3-1.7B-PARO)                                     |
| Qwen3-1.7B-Base              | [z-lab/Qwen3-1.7B-Base-PARO](https://huggingface.co/z-lab/Qwen3-1.7B-Base-PARO)                           |
| Qwen3-4B                     | [z-lab/Qwen3-4B-PARO](https://huggingface.co/z-lab/Qwen3-4B-PARO)                                         |
| Qwen3-4B-Base                | [z-lab/Qwen3-4B-Base-PARO](https://huggingface.co/z-lab/Qwen3-4B-Base-PARO)                               |
| Qwen3-8B                     | [z-lab/Qwen3-8B-PARO](https://huggingface.co/z-lab/Qwen3-8B-PARO)                                         |
| Qwen3-8B-Base                | [z-lab/Qwen3-8B-Base-PARO](https://huggingface.co/z-lab/Qwen3-8B-Base-PARO)                               |
| Qwen3-14B                    | [z-lab/Qwen3-14B-PARO](https://huggingface.co/z-lab/Qwen3-14B-PARO)                                       |
| Qwen3-14B-Base               | [z-lab/Qwen3-14B-Base-PARO](https://huggingface.co/z-lab/Qwen3-14B-Base-PARO)                             |
| DeepSeek-R1-Distill-Llama-8B | [z-lab/DeepSeek-R1-Distill-Llama-8B-PARO](https://huggingface.co/z-lab/DeepSeek-R1-Distill-Llama-8B-PARO) |

In addition, we provide the original checkpoints and pseudo-quantized models at https://huggingface.co/z-lab/paroquant-checkpoints to facilitate reproduction and further research.

## Reproduction

In the [`experiments`](./experiments/) directory, we provide the original scripts that produce the models, experiment results, and figures in the paper. Please refer to the [README](./experiments/README.md) for more details.

## Contribution

Contributions are welcome! Please install `pre-commit` to ensure consistent code styles before commiting any changes:

```bash
pip install pre-commit
pre-commit install
```

## Citation
