# Ascend Transformers inference

The Ascend backend loads public Qwen3-4B-PARO and Qwen3-8B-PARO checkpoints
without changing their learned rotations or AWQ tensor format. It fuses channel
scaling and eight group-128 pairwise rotations, retaining FP16 rounding between
stages. A one-time nibble reorder and signed-zero-offset conversion prepares
weights for `torch_npu.npu_weight_quant_batchmatmul`. The prepared buffers are
nonpersistent; serialized checkpoints retain their original tensors.

## Environment

Tested on a 64 GiB Ascend 910B2C with Python 3.11.14, PyTorch 2.9.0,
`torch_npu` 2.9.0.post1, Triton Ascend 3.2.0.dev20260322, CANN 8.5.1,
Transformers 4.57.6, Accelerate 1.15.0, and safetensors 0.7.0. Install the
Ascend runtime packages matching CANN in a project-local environment, then
install this repository without CUDA extras:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps -e .
.venv/bin/pip install transformers==4.57.6 accelerate==1.15.0 safetensors==0.7.0 requests
```

```python
import torch
import torch_npu
from transformers import AutoModelForCausalLM
import paroquant.inference.backends.transformers.quantizer

model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen3-4B-PARO", torch_dtype=torch.float16,
    device_map="npu", attn_implementation="sdpa").eval()
```

Measured records and every GSM8K numeric prediction are available in
[`experiments/ascend_results.json`](experiments/ascend_results.json).

## Measured performance

Three independent processes measure each model, mode, and input length on one
card during an exclusive timing window. The request generates eight tokens:
TTFT includes prefill and the first token, while decode throughput uses the
following seven. The table shows means at 2048 document tokens (2070 with chat
formatting); memory is the largest allocated peak across processes. The raw
evidence also includes 32/512-token inputs, all process records, cold loading,
first-forward preparation, and prompt-specific warmup.

| Model | Path | TTFT (ms) | Decode (tokens/s) | Peak (GiB) |
|---|---|---:|---:|---:|
| 4B | Expand + unfused | 3554.56 | 0.435 | 3.482 |
| 4B | Dense FP16 + fused | 280.60 | 23.854 | 8.142 |
| 4B | Original BF16 | 112.85 | 48.546 | 7.978 |
| 4B | Packed INT4 + fused | 284.61 | 22.656 | 4.857 |
| 8B | Expand + unfused | 5516.80 | 0.259 | 7.218 |
| 8B | Dense FP16 + fused | 401.17 | 24.059 | 15.911 |
| 8B | Original BF16 | 161.36 | 40.615 | 15.760 |
| 8B | Packed INT4 + fused | 395.31 | 21.687 | 9.714 |

Across all tested lengths, packed execution reduces allocated peak memory by
38.9–42.1% relative to the same PARO weights expanded to FP16, with 5.0–18.1%
lower decode throughput. Its decode throughput is 45.6–83.6× that of the naive
per-call expansion path. The original BF16 models are faster. The 4B packed
path at 32 document tokens ranges from 14.46 to 22.68 tokens/s across processes;
all observations remain in the reported mean.

The dense fused reference releases packed weights, zero points, and group
scales after materialization and prepares only its fused rotation data.
Loading peaks retain expansion cost. The production native path retains both
checkpoint weights and its additional packed execution copy.

Six operator paths compare per-stage rotation, rotation fusion, per-call and
cached weight expansion, native W4A16, and tiled Triton dequantization. Query
and down projections use 1, 16, and 128 rows, with ten warmed calls in each of
three independent processes. Rotation fusion lowers native complete-linear latency
in every tested shape. Dense fused execution is fastest in every tested shape;
the tiled Triton matrix path is slower than native in every tested shape.

## Numerical validation

Independent CPU reference calculations cover rotation direction, stagewise
rounding, AWQ nibble order, and groupwise dequantization. Row counts 1, 3, 17,
and 64 cover vector and matrix paths; the largest native relative L2 error is
0.000562. Actual model activations validate every quantized linear against
explicit FP16 weights and unfused rotations: 252 layers per model, maximum
relative L2 errors 0.001092 (4B) and 0.002660 (8B). The 216 measured operator
cases have maximum relative L2 error 0.001028.

## Model quality

All six configurations evaluate the same full WikiText-2 raw test sequence
(299,077 next-token targets) and all 1319 GSM8K test questions. GSM8K uses
greedy non-thinking decoding, batches of eight, and at most 512 new tokens.
Each row reports the complete result; concurrent quality-job timings are not
used as performance evidence.

| Model | Execution | WikiText-2 perplexity | GSM8K correct / 1319 | GSM8K accuracy |
|---|---|---:|---:|---:|
| 4B | Packed INT4 + fused | 13.858716 | 1181 | 89.54% |
| 4B | Dense FP16 + fused | 13.859074 | 1185 | 89.84% |
| 4B | Original BF16 | 13.665473 | 1194 | 90.52% |
| 8B | Packed INT4 + fused | 9.881995 | 1149 | 87.11% |
| 8B | Dense FP16 + fused | 9.882060 | 1147 | 86.96% |
| 8B | Original BF16 | 9.728429 | 1152 | 87.34% |

Native and dense execution use identical learned PARO weights and rotations.
Their perplexities differ by less than 0.00036. The original BF16 checkpoints
have lower perplexity. GSM8K results use the stated 512-token non-thinking
protocol; the table separates execution effects from checkpoint quantization.

## Reproduction

The download scripts obtain the public ModelScope checkpoints and raw test
datasets, writing source revisions alongside them. Use an idle, healthy device;
load the local CANN environment and check `npu-smi info` immediately before
each NPU process. The supplied runners perform those checks on every process
and use `/usr/local/Ascend/cann/set_env.sh` for this host.

```bash
.venv/bin/python experiments/download_models.py --model z-lab/Qwen3-4B-PARO
.venv/bin/python experiments/download_models.py --model z-lab/Qwen3-8B-PARO
.venv/bin/python experiments/download_models.py --model Qwen/Qwen3-4B
.venv/bin/python experiments/download_models.py --model Qwen/Qwen3-8B
.venv/bin/python experiments/download_data.py

source /usr/local/Ascend/cann/set_env.sh
npu-smi info
ASCEND_RT_VISIBLE_DEVICES=0 .venv/bin/python experiments/ascend_correctness.py

ASCEND_CARD=0 bash experiments/run_layers_formal.sh
ASCEND_CARD=0 bash experiments/run_models_formal.sh
ASCEND_CARD=0 MODEL_SIZE=4 bash experiments/run_quality.sh
ASCEND_CARD=0 MODEL_SIZE=8 bash experiments/run_quality.sh
```

The formal runners use seeds 932/933/934 for operators and 917/918/919 for
complete models. Quality uses seed 971, all 299,077 WikiText-2 next-token
targets with context 2048, and all 1319 GSM8K test questions. GSM8K uses
batches of eight, greedy non-thinking generation, and at most 512 new tokens.
The same prompt and numeric answer parser apply to all three quality modes.
Dataset revisions are `b08601e04326c79dfdd32d625aee71d232d685c3` (WikiText)
and `740312add88f781978c0658806c59bc2815b9866` (GSM8K). Saved full batches
resume under the same configuration; an actual interruption/resume preserves
16 saved answers and appends the next batch without duplicate indices.

## Limitations

The measured configuration uses FP16 activations, four-bit weights, eight
rotations, and groups of 128 channels. It uses the native operator high-precision
eager ND path; packed INT4 FRACTAL_NZ requires a separate graph-mode execution
path in the installed operator. These measurements cover one Ascend 910B2C host.
Short fixed-token performance requests and the stated quality protocol do not
establish long-context serving or reasoning-mode performance.
