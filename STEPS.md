# ParoQuant Export Tryout (MLX + GGUF + Ollama + LM Studio)

This guide reproduces a full export of `z-lab/Qwen3.5-9B-PARO` using the
`codex/export-compat` branch.

## 1) Prerequisites

- Apple Silicon macOS (M2 Ultra recommended for full 9B export)
- `uv` installed
- `cmake` installed (`brew install cmake`)
- Optional runtimes for validation:
  - `ollama`
  - LM Studio (desktop app and/or `lms` CLI)
- Hugging Face token with read access to the model

## 2) Clone and checkout

```bash
git clone --recurse-submodules git@github.com:keskinonur/paroquant-export.git
cd paroquant-export
git checkout codex/export-compat
git pull --ff-only origin codex/export-compat
git submodule update --init --recursive
```

## 3) Python environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[export,mlx]"
```

## 4) Build llama.cpp (required for GGUF targets)

```bash
cmake -S ./src/llama.cpp -B ./src/llama.cpp/build -DGGML_METAL=ON
cmake --build ./src/llama.cpp/build -j
```

Quick sanity check:

```bash
test -f ./src/llama.cpp/convert_hf_to_gguf.py
test -x ./src/llama.cpp/build/bin/llama-quantize
```

## 5) Set HF token

Use one of the following:

```bash
export HF_TOKEN=hf_xxx
```

or

```bash
set -a
source .env
set +a
```

Verify token is visible to Python:

```bash
uv run python -c "import os; print(bool(os.getenv('HF_TOKEN')))"
```

Expected output: `True`

## 6) Run exporter

```bash
uv run python -m paroquant.cli.export \
  --model z-lab/Qwen3.5-9B-PARO \
  --output-dir output/qwen3.5-9b-export \
  --targets hf,mlx,gguf,ollama,lmstudio \
  --text-only \
  --gguf-quants Q4_K_M,Q8_0 \
  --llama-cpp-dir ./src/llama.cpp
```

## 7) Confirm generated artifacts

```bash
ls -lah output/qwen3.5-9b-export/gguf
cat output/qwen3.5-9b-export/lmstudio/model.yaml
head -n 80 output/qwen3.5-9b-export/manifest.json
```

Expected GGUF files:

- `model-f16.gguf`
- `model-Q4_K_M.gguf`
- `model-Q8_0.gguf`

## 8) Runtime checks

### 8.1 MLX smoke test

```bash
uv run python -m mlx_lm generate \
  --model output/qwen3.5-9b-export/mlx \
  --prompt "Say hello in one short sentence. Output only the sentence." \
  --max-tokens 64
```

### 8.2 Ollama smoke test

```bash
cd output/qwen3.5-9b-export/ollama
ollama create qwen35-paro -f Modelfile
ollama run qwen35-paro
```

Prompt suggestion:

```text
Say hello in one short sentence. Output only the sentence.
```

Use `/bye` to exit Ollama chat.

### 8.3 LM Studio smoke test (MLX)

If LM Studio uses the default local models directory:

```bash
SRC="$PWD/output/qwen3.5-9b-export/mlx"
DST="$HOME/.lmstudio/models/keskinonur/qwen3.5-9b-paro-mlx"
mkdir -p "$(dirname "$DST")"
ln -sfn "$SRC" "$DST"
```

Then refresh LM Studio `My Models`, load the model, and run:

```text
Strict mode. Output exactly one line of minified JSON with keys hello_en, hello_tr, math_37x24. Constraints: hello_en must be one short English greeting, hello_tr must be one short Turkish greeting, math_37x24 must be the exact integer result of 37*24. No markdown, no extra text, no reasoning tags.
```

Expected:

- valid one-line JSON
- `math_37x24` is `888`

## 9) Common pitfalls

- `HF_TOKEN` missing in the active shell: export fails or download hangs.
- Wrong `--llama-cpp-dir`: GGUF conversion step fails.
- Deprecated MLX invocation warning:
  - prefer `python -m mlx_lm generate` over `python -m mlx_lm.generate`.
- Overly long answers in Ollama:
  - reduce `num_predict`
  - add stricter `SYSTEM` and stop tokens in `Modelfile`
